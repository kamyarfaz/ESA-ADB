import unittest
import numpy as np
import pandas as pd
from esa_thesis.mission2_study import resample, interval_mask, scoring_starts, jobs, BASE
from esa_thesis.context_expansion import aggregate_max
from esa_thesis.evaluation.esa import EventEvaluator

class Mission2PreparationTests(unittest.TestCase):
    def test_causal_split_age_and_no_backfill(self):
        second=10**9
        t=np.array([-1,10,20,650])*second
        v=np.array([99.,1.,2.,3.]);grid=np.array([0,10,15,20,621,650])*second
        x,valid,age,source=resample(t,v,grid,0,651*second)
        np.testing.assert_array_equal(valid,[False,True,True,True,False,True])
        np.testing.assert_allclose(x,[np.nan,1,1,2,np.nan,3],equal_nan=True)
        self.assertEqual(source[2],10*second)
        self.assertEqual(age[0],-1)

    def test_annotation_bins_conservative_and_closed(self):
        ann=pd.DataFrame(dict(StartTime=[pd.Timestamp(19)],EndTime=[pd.Timestamp(20)]))
        np.testing.assert_array_equal(interval_mask(np.array([0,10,20,30]),ann,10),[False,True,True,False])
        np.testing.assert_array_equal(interval_mask(np.array([10,19,19,20,21]),ann),[False,True,True,True,False])

    def test_scoring_tail_coverage(self):
        for n in [256,257,288,289,1000]:
            starts=scoring_starts(n)
            self.assertEqual(starts[-1]+256,n)
            self.assertEqual(len(starts),len(set(starts)))
            values=np.full((len(starts),2),3,np.float32)
            np.testing.assert_array_equal(aggregate_max(values,starts,n),np.full(n,3,np.float32))

    def test_nine_jobs_and_base_scope(self):
        features=BASE+['channel_86','channel_87','channel_88']
        plans=jobs(features)
        self.assertEqual(len(plans),9)
        self.assertEqual(len(set(p['name'] for p in plans)),9)
        self.assertTrue(all(p['features']==BASE for p in plans[:6]))
        self.assertTrue(all(p['features']==features for p in plans[6:]))

    def test_anomaly_only_rare_events_neutral(self):
        t=np.arange(10)*10**9
        labels=pd.DataFrame(dict(ID=['a','r'],StartTime=[pd.Timestamp(t[2]),pd.Timestamp(t[6])],EndTime=[pd.Timestamp(t[3]),pd.Timestamp(t[7])],Category=['Anomaly','Rare Event']))
        prediction=np.array([0,0,0,0,0,0,1,0,0,0])
        primary=EventEvaluator(t,labels).score(prediction)
        secondary=EventEvaluator(t,labels,('Anomaly',)).score(prediction)
        self.assertEqual(primary['TPe'],1);self.assertEqual(secondary['TPe'],0)
        self.assertEqual(secondary['FPe'],0)
        self.assertEqual(primary['nominal_seconds'],secondary['nominal_seconds'])

if __name__=='__main__':unittest.main()
