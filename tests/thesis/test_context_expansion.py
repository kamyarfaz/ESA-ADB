import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from esa_thesis import context_expansion as ce
from esa_thesis.diagnose_channels import aggregate
from esa_thesis.data import RobustChannelScaler

class TinyAE(torch.nn.Module):
    def __init__(self,channels):
        super().__init__();self.scale=torch.nn.Parameter(torch.ones(channels)*.5);self.drop=torch.nn.Dropout(.2)
    def forward(self,x):return self.drop(x)*self.scale[None,:,None,None]


class TinyMasked(TinyAE):
    def __init__(self,channels):
        super().__init__(channels);self.mask_token=torch.nn.Parameter(torch.zeros(1))
    def forward(self,x,phase=0):return super().forward(x)+self.mask_token


class ContextExpansionTests(unittest.TestCase):
    def test_paired_inventory_and_seed_order(self):
        jobs=ce.jobs([33,49])
        self.assertEqual(len(jobs),18)
        self.assertEqual(len({j['name'] for j in jobs}),18)
        for a,b in zip(jobs[::2],jobs[1::2]):
            self.assertEqual(a['method'],b['method']);self.assertEqual(a['seed'],b['seed'])
            self.assertEqual(b['features'],a['features']+['channel_33','channel_49'])
            self.assertEqual(len(a['features']),11 if a['method']=='expanded_short_no_roll' else 8)
        starts=np.arange(100)
        np.testing.assert_array_equal(ce.epoch_order(starts,4,42),ce.epoch_order(starts,4,42))
        self.assertFalse(np.array_equal(ce.epoch_order(starts,4,42),ce.epoch_order(starts,4,43)))

    def test_bounded_aggregation_exact_original_parity(self):
        starts=np.arange(0,4096-ce.config.SEQ_LEN+1,ce.config.SCORE_STRIDE)
        errors=np.random.default_rng(2).lognormal(size=(len(starts),13)).astype(np.float32)
        np.testing.assert_array_equal(ce.aggregate_max(errors,starts,4096),aggregate(errors,starts,4096).max(axis=0))
        with self.assertRaisesRegex(ValueError,'Uncovered'): ce.aggregate_max(errors,starts,4100)

    def test_bounded_scaler_exact_parity(self):
        x=np.random.default_rng(4).normal(size=(51000,13)).astype(np.float32)
        y=np.zeros(len(x),np.int8);y[120:210]=1
        scaler=RobustChannelScaler().fit(x,y)
        np.testing.assert_array_equal(ce.scale_inplace(x.copy(),scaler),scaler.transform(x))

    def test_bounded_inference_matches_existing_scorer(self):
        from esa_thesis.masked_pilot import score_windows
        x=np.random.default_rng(4).normal(size=(512,8)).astype(np.float32)
        model=TinyAE(8)
        np.testing.assert_array_equal(ce.score_bounded(model,x,'cpu',4),score_windows(model,x,False,'cpu',4))

    def test_target_scope_is_selection_objective(self):
        calls=[]
        def sweep(score,**kwargs):
            calls.append(kwargs['evaluator'])
            return pd.DataFrame([dict(val_esa_f05=.8,val_pred_anomaly_rate=0.,val_saturated=False,
                                      val_false_positive_seconds=0.,threshold=1.,merge_gap=0,min_dur=1)])
        with patch.object(ce,'assessment_scopes',return_value={'input_channels_40_47':'target','system_all_channels':'system'}),patch.object(ce,'sweep_thresholds',side_effect=sweep):
            ce.choose_rule(np.ones(10),np.arange(10))
        self.assertEqual(calls,['target'])

    def test_joint_freeze_refuses_missing_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'All18'):ce.freeze_selections(Path(tmp),ce.jobs([33,49]))
            self.assertFalse((Path(tmp)/'selection_complete.json').exists())

    def test_logging_does_not_advance_training_rng(self):
        class Logger:
            def log(self,metrics):np.random.rand(4);torch.rand(4)
        before=ce.capture_rng();ce.safe_log(Logger(),{})
        after=ce.capture_rng()
        self.assertTrue(torch.equal(before['torch'],after['torch']))
        np.testing.assert_array_equal(before['numpy'][1],after['numpy'][1])

    def test_masked_transition_resume(self):
        x=np.random.default_rng(7).normal(size=(288,8)).astype(np.float32)
        def load(*args):return x.copy(),np.zeros(len(x),np.int8),np.arange(len(x))
        def error(model,inputs,masked,phase):return (model(inputs)-inputs).square().mean(dim=(1,2,3))
        def unmask(model,channels,device):
            new=TinyAE(channels)
            new.load_state_dict({k:v for k,v in model.state_dict().items() if k!='mask_token'})
            return new
        def rule(score,*args):
            row=dict(val_esa_f05=.5,val_event_recall=.5,threshold=1.,merge_gap=0,min_dur=1)
            return row,pd.DataFrame([row])
        with tempfile.TemporaryDirectory() as tmp,patch.object(ce,'EPOCHS',10),patch.object(ce,'PRETRAIN_EPOCHS',5),patch.object(ce,'BATCH',1),patch.object(ce,'load_split',side_effect=load),patch.object(ce,'MultivariateAE',TinyAE),patch.object(ce,'MaskedAE',TinyMasked),patch.object(ce,'unmask',side_effect=unmask),patch.object(ce,'reconstruction_error',side_effect=error),patch.object(ce,'score_bounded',return_value=np.array([1.])),patch.object(ce,'choose_rule',side_effect=rule):
            job=ce.jobs([33,49])[6];a,b=Path(tmp)/'a',Path(tmp)/'b'
            ce.train(job,a,'cpu');original=ce.atomic_save
            def interrupt(value,path):
                original(value,path)
                if path.name=='last.pt' and value['epoch']==6 and value['batch']==0:raise RuntimeError('transition interruption')
            with patch.object(ce,'atomic_save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'transition'):ce.train(job,b,'cpu')
            ce.train(job,b,'cpu')
            sa=torch.load(a/'last.pt',weights_only=False);sb=torch.load(b/'last.pt',weights_only=False)
            for key in sa['model']:self.assertTrue(torch.equal(sa['model'][key],sb['model'][key]))
            self.assertEqual(sa['history'],sb['history']);self.assertEqual(sa['scheduler'],sb['scheduler'])
            self.assertEqual(sa['best']['epoch'],10)

    def test_exact_mid_epoch_resume_and_final_transaction(self):
        x=np.random.default_rng(7).normal(size=(288,8)).astype(np.float32)
        def load(*args):return x.copy(),np.zeros(len(x),np.int8),np.arange(len(x))
        def scoring(model,*args):return np.array([float(model.scale.detach().mean())])
        def rule(score,*args):
            row=dict(val_esa_f05=float(score[0]),val_event_recall=.5,threshold=1.,merge_gap=0,min_dur=1)
            return row,pd.DataFrame([row])
        with tempfile.TemporaryDirectory() as tmp,patch.object(ce,'EPOCHS',5),patch.object(ce,'BATCH',1),patch.object(ce,'load_split',side_effect=load),patch.object(ce,'MultivariateAE',TinyAE),patch.object(ce,'score_bounded',side_effect=scoring),patch.object(ce,'choose_rule',side_effect=rule):
            job=ce.jobs([33,49])[0];a,b=Path(tmp)/'a',Path(tmp)/'b'
            ce.train(job,a,'cpu')
            original=ce.atomic_save;tripped=[False]
            def interrupt(value,path):
                original(value,path)
                if path.name=='last.pt' and value['epoch']==1 and not tripped[0]:
                    tripped[0]=True;raise RuntimeError('interrupted')
            with patch.object(ce,'atomic_save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):ce.train(job,b,'cpu')
            ce.train(job,b,'cpu')
            sa=torch.load(a/'last.pt',weights_only=False);sb=torch.load(b/'last.pt',weights_only=False)
            for key in sa['model']:self.assertTrue(torch.equal(sa['model'][key],sb['model'][key]))
            self.assertEqual(sa['history'],sb['history']);self.assertEqual(sa['best'],sb['best'])
            self.assertTrue(torch.equal(sa['rng']['torch'],sb['rng']['torch']))
            self.assertTrue(ce.verify_receipt(b,'trained.json'))
            # A final-epoch checkpoint is enough to reconstruct a lost final history write.
            (b/'trained.json').unlink();(b/'history.csv').unlink()
            ce.train(job,b,'cpu')
            self.assertEqual((a/'history.csv').read_text(),(b/'history.csv').read_text())
            self.assertTrue(ce.verify_receipt(b,'trained.json'))


if __name__=='__main__':unittest.main()
