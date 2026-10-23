import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch
from esa_thesis import false_alarm_reduction as far
from esa_thesis.masked_pilot import score_windows
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.false_alarm_reduction import detected_ids, transformed, select_candidates

class AlarmRefinementTests(unittest.TestCase):
    def test_inference_resume_and_original_scoring_parity(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x*.5
        class Scaler:
            def transform(self,x):return x
        x=np.random.default_rng(42).normal(size=(512,2)).astype(np.float32)
        model=Model();expected=score_windows(model,x,False,'cpu',1)
        original_inputs=far.inference_inputs
        cpu_inputs=lambda a,b,c:original_inputs(a,b,'cpu')
        loader=lambda *a:(x.copy(),np.zeros(len(x)),np.arange(len(x)))
        with tempfile.TemporaryDirectory() as tmp,patch.object(far,'BATCH',1),patch.object(far,'load_model',return_value=(model,Scaler(),['a','b'],False)),patch.object(far,'load_split',side_effect=loader),patch.object(far,'inference_inputs',side_effect=cpu_inputs):
            folder=Path(tmp);save=far.save_npy;tripped=[False]
            def interrupt(path,values):
                save(path,values)
                if 'window_errors' in path.name and not tripped[0]:
                    tripped[0]=True;raise RuntimeError('interrupted after atomic checkpoint')
            with patch.object(far,'save_npy',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):far.channel_scores('pretrain_finetune',folder,'calibration')
            actual=far.channel_scores('pretrain_finetune',folder,'calibration')
            np.testing.assert_array_equal(actual.max(axis=0),expected)
            np.testing.assert_array_equal(actual,far.channel_scores('pretrain_finetune',folder,'calibration'))

    def test_channel_scaling_and_raw_identity(self):
        x=np.array([[2.,4.,0.],[12.,3.,0.]],dtype=np.float32)
        np.testing.assert_array_equal(transformed(x,'raw',[2,3]),[12,4,0])
        np.testing.assert_array_equal(transformed(x,'channel_q99',[2,3]),[4,2,0])
        self.assertTrue(np.isfinite(transformed(x,'channel_q99',[0,0])).all())
        with self.assertRaises(ValueError):transformed(x,'test_selected',[1,1])

    def test_exact_id_constraint_not_just_equal_count(self):
        times=pd.date_range('2006-01-01',periods=100,freq='30s').asi8
        labels=pd.DataFrame(dict(ID=['a','b'],StartTime=[pd.Timestamp(times[10]),pd.Timestamp(times[50])],EndTime=[pd.Timestamp(times[15]),pd.Timestamp(times[55])],Category=['Anomaly','Anomaly']))
        ev=EventEvaluator(times,labels)
        a=np.zeros(100);a[12]=1;b=np.zeros(100);b[52]=1
        self.assertEqual(detected_ids(ev,a),['a']);self.assertEqual(detected_ids(ev,b),['b'])

    def test_selection_preserves_ids_and_reduces_false_alarms(self):
        times=pd.date_range('2006-01-01',periods=1000,freq='30s').asi8
        labels=pd.DataFrame(dict(ID=['a'],StartTime=[pd.Timestamp(times[100])],EndTime=[pd.Timestamp(times[105])],Category=['Anomaly']))
        ev=EventEvaluator(times,labels);scopes={'system_all_channels':ev,'input_channels_40_47':ev}
        per=np.zeros((1,1000),np.float32);per[0,102]=10;per[0,500]=2
        base=dict(threshold=1.,merge_gap=0,min_dur=1)
        pred=per[0]>1
        diagnosis=dict(rule=base,normalization_q99=[1.],metrics={s:ev.score(pred) for s in scopes},detected_ids={s:['a'] for s in scopes})
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'diagnosis.json').write_text(json.dumps(diagnosis))
            select_candidates(folder,per,times,scopes)
            rule=json.loads((folder/'frozen_rule.json').read_text())['candidate']
            self.assertTrue(rule['preserves_ids']);self.assertTrue(rule['feasible'])
            self.assertEqual(rule['system_detected'],1);self.assertEqual(rule['false_alarm_events'],0)

if __name__=='__main__':unittest.main()
