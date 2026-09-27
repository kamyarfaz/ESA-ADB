import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.masked_pilot import MaskedAE, patch_mask, reconstruction_error, nominal_sample, train_arm, score_windows, fold_boundaries, assessment_scopes, run_pilot_fold
from esa_thesis.models import MultivariateAE
from esa_thesis.data import RobustChannelScaler
from esa_thesis.evaluation.esa import EventEvaluator

class MaskedPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_overlap_augmentation_changes_hidden_targets_for_offset(self):
        from esa_thesis.augmentation import make_pseudo_anomaly, _overlapping_starts
        import random
        x=torch.arange(config.SEQ_LEN,dtype=torch.float32)[None,None,:,None].repeat(8,8,1,1)
        original=x.clone()
        for phase in range(4):
            mask=patch_mask(16,phase,'cpu').repeat_interleave(16)
            valid=_overlapping_starts(tuple(mask.tolist()),12,84)
            for length, starts in valid.items():
                expected=[s for s in range(257-length) if mask[s:s+length].any()]
                self.assertEqual(starts,expected)
            randint=random.randint
            def offset_only(a,b):
                return 0 if (a,b)==(0,2) else randint(a,b)
            with patch('esa_thesis.augmentation.random.randint',side_effect=offset_only):
                changed=make_pseudo_anomaly(x,target_mask=mask)
            self.assertTrue(((changed-x)[:,:,mask,:].abs().flatten(1).sum(1)>0).all())
        torch.testing.assert_close(x,original)
        with self.assertRaises(ValueError):make_pseudo_anomaly(x,target_mask=torch.zeros(256))

    def test_hidden_targets_cannot_reach_predictions(self):
        torch.manual_seed(42);m=MaskedAE(8).eval()
        x=torch.randn(1,8,config.SEQ_LEN,1);y=x.clone()
        hidden=patch_mask(m.NP,1,'cpu').repeat_interleave(m.PS)
        y[:,:,hidden,:]+=100
        with torch.no_grad():
            torch.testing.assert_close(m(x,1),m(y,1),rtol=0,atol=0)
        # No input gradient is allowed through hidden patch contents.
        x.requires_grad_(True);m(x,1).sum().backward()
        self.assertEqual(x.grad[:,:,hidden,:].abs().sum().item(),0)

    def test_same_base_initialization_and_four_phase_coverage(self):
        torch.manual_seed(5);a=MultivariateAE(8)
        torch.manual_seed(5);b=MaskedAE(8)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        masks=torch.stack([patch_mask(b.NP,p,'cpu') for p in range(4)])
        self.assertTrue(torch.all(masks.sum(0)==1));self.assertTrue(torch.all(masks.sum(1)==b.NP//4))

    def test_hidden_loss_only_and_nominal_sampling(self):
        class Zero:
            NP=config.SEQ_LEN//config.PATCH_SIZE;PS=config.PATCH_SIZE
            def __call__(self,x,phase):return torch.zeros_like(x)
        x=torch.zeros(2,8,config.SEQ_LEN,1)
        hidden=patch_mask(Zero.NP,0,'cpu').repeat_interleave(Zero.PS)
        x[:,:,hidden,:]=2;x[:,:,~hidden,:]=100
        torch.testing.assert_close(reconstruction_error(Zero(),x,True,0),torch.full((2,),4.))
        labels=np.zeros(config.SEQ_LEN*4);labels[config.SEQ_LEN]=1
        starts=nominal_sample(labels,3,42)
        np.testing.assert_array_equal(starts,nominal_sample(labels,3,42))
        for s in starts:self.assertEqual(labels[s:s+config.SEQ_LEN].sum(),0)
        with self.assertRaises(ValueError):nominal_sample(np.ones(config.SEQ_LEN*2),3,42)

    def test_scoring_excludes_visible_targets(self):
        class Probe(torch.nn.Module):
            NP=config.SEQ_LEN//config.PATCH_SIZE;PS=config.PATCH_SIZE
            def forward(self,x,phase):
                hidden=patch_mask(self.NP,phase,x.device).repeat_interleave(self.PS)
                result=x+100
                result[:,:,hidden,:]=x[:,:,hidden,:]+(phase+2)
                return result
        x=np.zeros((config.SEQ_LEN,8),dtype='float32')
        actual=score_windows(Probe(),x,True,'cpu',1)
        np.testing.assert_array_equal(actual,np.full(config.SEQ_LEN,(4+9+16+25)/4))

    def test_training_checkpoint_and_completed_resume(self):
        x=np.random.default_rng(2).normal(size=(config.SEQ_LEN+config.STRIDE,8)).astype('float32')
        scaler=RobustChannelScaler().fit(x,np.zeros(len(x)))
        times=pd.date_range('2003-01-01',periods=len(x),freq='30s').asi8
        labels=pd.DataFrame({'ID':['a'],'StartTime':[pd.Timestamp(times[3])], 'EndTime':[pd.Timestamp(times[5])],'Category':['Anomaly']})
        ev=EventEvaluator(times,labels)
        with tempfile.TemporaryDirectory() as td,patch('esa_thesis.masked_pilot.score_windows',return_value=np.zeros(len(x))):
            out=Path(td)/'masked_overlap'
            first=train_arm('masked_overlap',x,np.array([0,config.STRIDE]),x,ev,scaler,out,1,2,'cpu',42)
            again=train_arm('masked_overlap',x,np.array([0,config.STRIDE]),x,ev,scaler,out,1,2,'cpu',42)
            self.assertEqual(first,again)
            self.assertEqual(json.loads((out/'frozen_rule.json').read_text())['epoch'],1)
            self.assertTrue((out/'model_checkpoint.pt').exists())

    def test_fold_boundaries_and_scope_neutrality(self):
        for year in [2003,2004,2005]:
            self.assertEqual(fold_boundaries(str(year)),(f'{year}-01-01',f'{year}-04-01',f'{year+1}-01-01'))
        with self.assertRaises(ValueError):fold_boundaries('2006')
        times=pd.date_range('2004-04-01',periods=10,freq='30s').asi8
        labels=pd.DataFrame({'ID':['local','outside'],'Channel':['channel_40','channel_50'],
            'StartTime':pd.to_datetime(times[[1,5]]),'EndTime':pd.to_datetime(times[[2,6]]),'Category':['Anomaly','Anomaly']})
        ev=EventEvaluator(times,labels)
        with patch('esa_thesis.masked_pilot.evaluator_for',return_value=ev):scopes=assessment_scopes(times)
        pred=np.zeros(10);pred[5]=1
        self.assertEqual(scopes['system_all_channels'].score(pred)['TPe'],1)
        local=scopes['input_channels_40_47'].score(pred)
        self.assertEqual((local['TPe'],local['FNe'],local['FPe']),(0,1,0))

    def test_2004_orchestration_freezes_both_before_assessment_and_resume_skip(self):
        from esa_thesis.diagnose_channels import TRAIN_SHA
        calls=[]
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'2004'
            args=SimpleNamespace(resume=False,epochs=1,max_windows=3,seed=42,batch_size=2,device='cpu')
            def load(path,features,start,end):
                calls.append((start,end))
                if start=='2004-04-01':
                    self.assertTrue(all((out/name/'frozen_rule.json').exists() for name in ['baseline','masked']))
                x=np.zeros((config.SEQ_LEN,8),dtype='float32')
                t=pd.date_range(start or '2000-01-01',periods=len(x),freq='30s').asi8
                return x,np.zeros(len(x)),t
            def evaluator(t):
                labels=pd.DataFrame({'ID':['a'],'Channel':['channel_40'],'StartTime':[pd.Timestamp(t[3])],
                    'EndTime':[pd.Timestamp(t[5])],'Category':['Anomaly']})
                return EventEvaluator(t,labels)
            def train(name,x,starts,cal,ev,scaler,dest,*rest):
                dest.mkdir();model=MaskedAE(8) if name=='masked' else MultivariateAE(8)
                torch.save({'model_state_dict':model.state_dict()},dest/'model_checkpoint.pt')
                best={'epoch':1,'rule':{'threshold':1.,'merge_gap':0,'min_dur':1,'val_esa_f05':0.}}
                (dest/'frozen_rule.json').write_text(json.dumps(best));return best
            with patch('esa_thesis.masked_pilot.load_period',side_effect=load),patch('esa_thesis.masked_pilot.evaluator_for',side_effect=evaluator),patch('esa_thesis.masked_pilot.train_arm',side_effect=train),patch('esa_thesis.masked_pilot.score_windows',return_value=np.zeros(config.SEQ_LEN)),patch('esa_thesis.masked_pilot.digest',return_value=TRAIN_SHA):
                rows,dual=run_pilot_fold(args,'2004',out)
                self.assertEqual(calls,[(None,'2004-01-01'),('2004-01-01','2004-04-01'),('2004-04-01','2005-01-01')])
                self.assertEqual((len(rows),len(dual)),(2,4))
                args.resume=True;calls.clear();run_pilot_fold(args,'2004',out);self.assertEqual(calls,[])

if __name__=='__main__':unittest.main()
