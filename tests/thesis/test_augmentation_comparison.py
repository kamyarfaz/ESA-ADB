import json
import random
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis.augmentation_comparison import isolated_random,training_starts,train_arm,main
from esa_thesis.data import RobustChannelScaler
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.models import MultivariateAE

class AugmentationComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_isolated_rng_restored_even_on_error(self):
        random.seed(3);np.random.seed(3);torch.manual_seed(3)
        expected=(random.random(),np.random.random(),torch.rand(3))
        random.seed(3);np.random.seed(3);torch.manual_seed(3)
        with self.assertRaises(RuntimeError):
            with isolated_random(99):
                random.random();np.random.random();torch.rand(10);raise RuntimeError()
        self.assertEqual(random.random(),expected[0]);self.assertEqual(np.random.random(),expected[1]);self.assertTrue(torch.equal(torch.rand(3),expected[2]))

    def test_nominal_budget_and_seed(self):
        y=np.zeros(2048);y[800:820]=2
        a=training_starts(y,12,42);np.testing.assert_array_equal(a,training_starts(y,12,42))
        self.assertEqual(len(a),12)
        for start in a:self.assertTrue((y[start:start+256]==0).all())
        with self.assertRaises(ValueError):training_starts(np.ones(512),0,42)

    def test_interrupted_resume_matches_uninterrupted_both_modes(self):
        x=np.random.default_rng(1).normal(size=(272,8)).astype(np.float32)
        scaler=RobustChannelScaler().fit(x,np.zeros(272));starts=np.array([0,16])
        times=pd.date_range('2003-01-01',periods=272,freq='30s').asi8
        ev=EventEvaluator(times,pd.DataFrame({'ID':['a'],'StartTime':[pd.Timestamp(times[10])],'EndTime':[pd.Timestamp(times[20])],'Category':['Anomaly']}))
        args=SimpleNamespace(seed=42,device='cpu',epochs=2,batch_size=2,resume=False)
        with tempfile.TemporaryDirectory() as td:
            for mode in ['legacy','short_coordinated','short_no_roll']:
                whole=Path(td)/(mode+'_whole');part=Path(td)/(mode+'_part')
                args.resume=False
                with patch('esa_thesis.augmentation_comparison.score',return_value=np.zeros(272)):
                    train_arm(mode,x,starts,x,ev,scaler,whole,args)
                calls=0
                def interrupted(*unused):
                    nonlocal calls
                    calls+=1
                    if calls==2:raise RuntimeError('simulated interruption')
                    return np.zeros(272)
                with patch('esa_thesis.augmentation_comparison.score',side_effect=interrupted),self.assertRaises(RuntimeError):
                    train_arm(mode,x,starts,x,ev,scaler,part,args)
                args.resume=True
                with patch('esa_thesis.augmentation_comparison.score',return_value=np.zeros(272)):
                    train_arm(mode,x,starts,x,ev,scaler,part,args)
                a=torch.load(whole/'last.pt',weights_only=False);b=torch.load(part/'last.pt',weights_only=False)
                for key in a['model']:self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)
                self.assertEqual(a['best'],b['best'])
                self.assertEqual([v['loss'] for v in a['history']],[v['loss'] for v in b['history']])

    def test_both_rules_frozen_before_assessment(self):
        self.check_frozen_rules('original',['legacy','short_coordinated'])

    def test_no_roll_rules_frozen_before_assessment(self):
        self.check_frozen_rules('short-no-roll',['short_coordinated','short_no_roll'])

    def check_frozen_rules(self,comparison,arms):
        class Evaluator:
            def score(self,p):return {'EW_F_0.50':0.}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run';calls=[]
            def load(path,features,start,end):
                calls.append((start,end))
                if start=='2003-04-01':self.assertTrue(all((out/'2003'/m/'frozen_rule.json').exists() for m in arms))
                t=pd.date_range(start or '2000-01-01',periods=272,freq='30s').asi8
                return np.zeros((272,8),np.float32),np.zeros(272),t
            def train(mode,x,starts,val,ev,scaler,dest,args):
                dest.mkdir();torch.save({'model_state_dict':MultivariateAE(8).state_dict()},dest/'model_checkpoint.pt')
                d={'epoch':1,'rule':{'threshold':1.,'merge_gap':0,'min_dur':1,'val_esa_f05':0.}}
                (dest/'frozen_rule.json').write_text(json.dumps(d));return d
            with patch('sys.argv',['augmentation-comparison','--comparison',comparison,'--output',str(out),'--folds','2003','--device','cpu']),patch('esa_thesis.augmentation_comparison.verify_inputs',return_value={}),patch('esa_thesis.augmentation_comparison.load_period',side_effect=load),patch('esa_thesis.augmentation_comparison.train_arm',side_effect=train),patch('esa_thesis.augmentation_comparison.evaluator_for',return_value=Evaluator()),patch('esa_thesis.augmentation_comparison.score',return_value=np.zeros(272)),patch('esa_thesis.augmentation_comparison.assessment_scopes',return_value={'system':Evaluator(),'local':Evaluator()}):
                main()
            self.assertEqual(calls,[(None,'2003-01-01'),('2003-01-01','2003-04-01'),('2003-04-01','2004-01-01')])
            self.assertEqual(len(pd.read_csv(out/'results.csv')),4)
            self.assertEqual(json.loads((out/'protocol.json').read_text())['arms'],arms)

if __name__=='__main__':unittest.main()
