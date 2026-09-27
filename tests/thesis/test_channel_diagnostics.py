import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis.diagnose_channels import aggregate, training_starts, event_diagnostics, window_errors
from esa_thesis.evaluation.compare_scoring import paired_scores
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis import config

class ChannelDiagnosticTests(unittest.TestCase):
    def test_per_channel_aggregation_matches_existing_scorer(self):
        class Model(torch.nn.Module):
            def forward(self,x): return x*.2
        rng=np.random.default_rng(4)
        x=rng.normal(size=(config.SEQ_LEN+config.SCORE_STRIDE*3,8)).astype('float32')
        starts=np.arange(0,len(x)-config.SEQ_LEN+1,config.SCORE_STRIDE)
        e=window_errors(Model(),x,starts,'cpu',2,'test')
        actual=aggregate(e,starts,len(x)).max(axis=0)
        expected,uncovered=paired_scores(Model(),x,device='cpu',batch_size=2)
        self.assertEqual(uncovered,0)
        np.testing.assert_array_equal(actual,expected['window_mean'])
        with self.assertRaises(ValueError): aggregate(e,starts,len(x)+1)

    def test_sampling_respects_nominal_and_year_boundaries(self):
        times=[]
        for year in range(2000,2006):
            times.extend(pd.date_range(f'{year}-01-01',periods=20,freq='30s').asi8)
        times=np.asarray(times);labels=np.zeros(len(times));labels[2]=1
        with patch('esa_thesis.diagnose_channels.config.SEQ_LEN',4),patch('esa_thesis.diagnose_channels.config.STRIDE',2):
            starts=training_starts(labels,times,3)
            self.assertEqual(len(starts),18)
            earlier=training_starts(labels,times,3,end_year=2004)
            self.assertEqual(len(earlier),12)
            self.assertTrue((pd.DatetimeIndex(times[earlier]).year < 2004).all())
            years=pd.DatetimeIndex(times).year.to_numpy()
            for s in starts:
                self.assertEqual(labels[s:s+4].sum(),0)
                self.assertEqual(years[s],years[s+3])
            with self.assertRaises(ValueError):training_starts(np.ones(len(times)),times,3)

    def test_event_attribution_uses_intervals_not_only_sample_membership(self):
        times=pd.to_datetime([0,30,60,90],unit='s').asi8
        labels=pd.DataFrame({'ID':['hit','miss'],'StartTime':pd.to_datetime([5,65],unit='s'),
            'EndTime':pd.to_datetime([6,66],unit='s'),'Category':['Anomaly','Anomaly']})
        ev=EventEvaluator(times,labels)
        rows=event_diagnostics(ev,np.array([1,0,0,0]),np.array([2.,0.,0.,0.])).set_index('event_id')
        self.assertTrue(rows.loc['hit','detected'])
        self.assertEqual(rows.loc['hit','samples_inside_raw_intervals'],0)
        self.assertFalse(rows.loc['miss','detected'])
        self.assertEqual(rows.detected.sum(),ev.score(np.array([1,0,0,0]))['TPe'])

if __name__=='__main__':unittest.main()
