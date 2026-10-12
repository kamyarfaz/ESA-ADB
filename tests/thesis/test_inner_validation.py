import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from esa_thesis import inner_validation as inner
from esa_thesis.data import RobustChannelScaler


class InnerValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ties_and_fixed_policy(self):
        history = [dict(epoch=i, rule={'val_esa_f05':v}) for i,v in enumerate([0,.7,.7,.1,0],1)]
        decisions = inner.choose_decisions(history)
        self.assertEqual(decisions['calibration_selected']['epoch'],2)
        self.assertEqual(decisions['fixed_epoch5']['epoch'],5)
        self.assertFalse(decisions['fixed_epoch5']['uninformative_calibration'])
        for row in history: row['rule']['val_esa_f05'] = 0
        decisions = inner.choose_decisions(history)
        self.assertEqual(decisions['calibration_selected']['epoch'],1)
        self.assertTrue(decisions['fixed_epoch5']['uninformative_calibration'])
        with self.assertRaises(ValueError): inner.choose_decisions(history[1:])

    def test_validation_refuses_missing_policy_before_read(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p/'frozen_decisions.json').write_text(json.dumps({'calibration_selected':{}}))
            with patch.object(inner,'load_period') as load:
                with self.assertRaises(ValueError): inner.validate_frozen(p,inner.FOLDS['2001'],None,None)
                load.assert_not_called()

    def test_actual_training_resume_and_all_epoch_artifacts(self):
        x = np.random.default_rng(2).normal(size=(272,8)).astype(np.float32)
        scaler = RobustChannelScaler().fit(x,np.zeros(272))
        times = pd.date_range('2001-01-01',periods=272,freq='30s').asi8
        args = SimpleNamespace(seed=42,device='cpu',epochs=2,batch_size=2,resume=False)
        with tempfile.TemporaryDirectory() as td:
            whole,part = Path(td)/'whole',Path(td)/'part'
            with patch.object(inner,'score',return_value=np.zeros(272)):
                inner.train(x,np.array([0,16]),x,times,scaler,whole,args)
            calls = 0
            def interrupted(*args):
                nonlocal calls
                calls += 1
                if calls == 2: raise RuntimeError('interrupted')
                return np.zeros(272)
            with patch.object(inner,'score',side_effect=interrupted):
                with self.assertRaises(RuntimeError): inner.train(x,np.array([0,16]),x,times,scaler,part,args)
            args.resume = True
            with patch.object(inner,'score',return_value=np.zeros(272)):
                inner.train(x,np.array([0,16]),x,times,scaler,part,args)
            a = torch.load(whole/'last.pt',weights_only=False); b = torch.load(part/'last.pt',weights_only=False)
            for key in a['model']: self.assertTrue(torch.equal(a['model'][key],b['model'][key]))
            self.assertEqual([r['loss'] for r in a['history']],[r['loss'] for r in b['history']])
            self.assertEqual(json.loads((whole/'frozen_decisions.json').read_text()),json.loads((part/'frozen_decisions.json').read_text()))
            for epoch in (1,2):
                self.assertTrue((part/f'epoch_{epoch}'/'calibration_candidates.csv').exists())
            (part/'epoch_1'/'calibration_score.npy').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): inner.train(x,np.array([0,16]),x,times,scaler,part,args)

    def test_frozen_policies_use_same_validation_and_shared_epoch_score(self):
        from esa_thesis.models import MultivariateAE
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'epoch_5').mkdir()
            torch.save({'model_state_dict':MultivariateAE(8).state_dict()},out/'epoch_5'/'model.pt')
            decision=dict(epoch=5,rule=dict(threshold=1,merge_gap=0,min_dur=1,val_esa_f05=0),uninformative_calibration=True)
            inner.write_json(out/'frozen_decisions.json',{p:decision for p in inner.POLICIES})
            x=np.zeros((272,8),np.float32);times=pd.date_range('2001-07-01',periods=272,freq='30s').asi8
            scaler=RobustChannelScaler().fit(x,np.zeros(272))
            with patch.object(inner,'load_period',return_value=(x,np.zeros(272),times)) as load,patch.object(inner,'score',return_value=np.zeros(272)) as score:
                rows=inner.validate_frozen(out,inner.FOLDS['2001'],scaler,SimpleNamespace(device='cpu'))
                self.assertEqual(load.call_count,1);self.assertEqual(score.call_count,1)
                self.assertEqual(load.call_args.args[-2:],('2001-07-01','2002-01-01'))
                self.assertEqual(len(rows),4)
            np.testing.assert_array_equal(np.load(out/'calibration_selected_validation_prediction.npy'),np.load(out/'fixed_epoch5_validation_prediction.npy'))

    def test_completed_campaign_resume_reloads_year_labels(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'run'
            def load(path,features,start,end):
                times=pd.date_range(start or '2000-01-01',periods=272,freq='30s').asi8
                return np.zeros((272,8),np.float32),np.zeros(272),times
            rows=[dict(policy=policy,scope=scope,epoch=5,calibration_f05=0,
                       EW_F_0_50=0,FPe=0,EW_recall=0,**{'EW_F_0.50':0})
                  for policy in inner.POLICIES for scope in ('system_all_channels','input_channels_40_47')]
            argv=['inner-validation','--output',str(output),'--device','cpu']
            with patch.object(inner,'verify_inputs',return_value={}),patch.object(inner,'load_period',side_effect=load),patch.object(inner,'train') as train,patch.object(inner,'validate_frozen',return_value=rows),patch('sys.argv',argv):
                inner.main()
                self.assertEqual(train.call_count,6)
            with patch.object(inner,'verify_inputs',return_value={}),patch.object(inner,'load_period',side_effect=load),patch.object(inner,'train') as train,patch('sys.argv',argv+['--resume']):
                inner.main()
                train.assert_not_called()
            self.assertFalse(json.loads((output/'decision.json').read_text())['advancement_gate_passed'])
            self.assertEqual(len(pd.read_csv(output/'results.csv')),24)


if __name__=='__main__': unittest.main()
