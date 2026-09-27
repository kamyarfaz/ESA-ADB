import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.data import RobustChannelScaler
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.timing_comparison import TimingWindows,nominal_starts,train_arm,verify_timing,load_timing


class TimingComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_windows_align_metadata_and_exclude_non_nominal(self):
        x=np.arange(320*8,dtype=np.float32).reshape(320,8);context=np.stack([x,x+1],axis=-1)
        ds=TimingWindows(x,context,np.array([16]));values,timing=ds[0]
        np.testing.assert_array_equal(values[:,:,0].numpy().T,x[16:272])
        np.testing.assert_array_equal(timing.numpy().transpose(1,0,2),context[16:272])
        labels=np.zeros(320);labels[0]=1
        for start in nominal_starts(labels):self.assertTrue(np.all(labels[start:start+256]==0))

    def test_rejects_incomplete_timing_and_grid_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'protocol.json').write_text(json.dumps({'status':'incomplete'}))
            with self.assertRaises(ValueError):verify_timing(root)
            np.save(root/'timestamps.npy',np.arange(10))
            with self.assertRaisesRegex(ValueError,'grid differs'):load_timing(root,np.array([3,5]))

    def test_real_training_and_completed_epoch_resume_both_arms(self):
        x=np.random.default_rng(42).normal(size=(272,8)).astype(np.float32)
        context=np.ones((272,8,2),dtype=np.float32);starts=np.array([0,16])
        scaler=RobustChannelScaler().fit(x,np.zeros(len(x)))
        times=pd.date_range('2006-01-01',periods=len(x),freq='30s').asi8
        labels=pd.DataFrame({'ID':['a'],'StartTime':[pd.Timestamp(times[4])],'EndTime':[pd.Timestamp(times[8])],'Category':['Anomaly']})
        ev=EventEvaluator(times,labels);args=SimpleNamespace(seed=42,device='cpu',epochs=1,batch_size=2)
        with tempfile.TemporaryDirectory() as td,patch('esa_thesis.timing_comparison.score',return_value=np.zeros(len(x))):
            for name in ['control','observation_age']:
                out=Path(td)/name
                first=train_arm(name,x,context,starts,x,context,ev,scaler,out,args)
                again=train_arm(name,x,context,starts,x,context,ev,scaler,out,args)
                self.assertEqual(first,again)
                saved=torch.load(out/'model_checkpoint.pt',weights_only=False)
                weight=saved['model_state_dict']['timing_proj.weight']
                if name=='control':self.assertEqual(weight.abs().sum().item(),0)
                else:self.assertGreater(weight.abs().sum().item(),0)

    def test_main_freezes_both_before_assessment_and_reports_every_year(self):
        from esa_thesis.timing_comparison import main
        from esa_thesis.timing_model import TimingAwareAE
        class Evaluator:
            def score(self,pred):return {'EW_F_0.50':0.,'TPe':0,'FPe':0,'FNe':1}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'run';calls=[]
            def load(path,features,start,end):
                calls.append((start,end))
                if start=='2007-01-01':
                    self.assertTrue(all((out/name/'frozen_rule.json').exists() for name in ['control','observation_age']))
                    times=np.concatenate([pd.date_range(f'{y}-01-01',periods=272,freq='30s').asi8 for y in range(2007,2014)])
                else:times=pd.date_range(start or '2000-01-01',periods=272,freq='30s').asi8
                return np.zeros((len(times),8),dtype=np.float32),np.zeros(len(times)),times
            def train(name,x,ctx,starts,val,vctx,ev,scaler,dest,args):
                dest.mkdir();torch.save({'model_state_dict':TimingAwareAE(8).state_dict()},dest/'model_checkpoint.pt')
                rule={'epoch':1,'rule':{'threshold':1.,'merge_gap':0,'min_dur':1,'val_esa_f05':0.}}
                (dest/'frozen_rule.json').write_text(json.dumps(rule));return rule
            with patch('sys.argv',['timing-comparison','--timing-dir',str(root/'timing'),'--output',str(out),'--device','cpu','--epochs','1']), \
                 patch('esa_thesis.timing_comparison.verify_inputs',return_value={}), \
                 patch('esa_thesis.timing_comparison.verify_timing',return_value={}), \
                 patch('esa_thesis.timing_comparison.load_period',side_effect=load), \
                 patch('esa_thesis.timing_comparison.load_timing',side_effect=lambda root,t:np.zeros((len(t),8,2),dtype=np.float32)), \
                 patch('esa_thesis.timing_comparison.evaluator_for',return_value=Evaluator()), \
                 patch('esa_thesis.timing_comparison.train_arm',side_effect=train), \
                 patch('esa_thesis.timing_comparison.score',side_effect=lambda model,x,*args:np.zeros(len(x))), \
                 patch('esa_thesis.timing_comparison.assessment_scopes',return_value={'system_all_channels':Evaluator(),'input_channels_40_47':Evaluator()}):
                main()
            self.assertEqual(calls,[(None,'2006-01-01'),('2006-01-01','2007-01-01'),('2007-01-01','2014-01-01')])
            annual=pd.read_csv(out/'yearly_results.csv')
            self.assertEqual(len(annual),28)
            self.assertEqual(set(annual.year),set(range(2007,2014)))

if __name__=='__main__':unittest.main()
