import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/research'))
from all_channel_correlation_review import stable_matrix, matrices
from verify_all_channel_correlations import independent_pearson


class AllChannelCorrelationTests(unittest.TestCase):
    def test_stability_requires_every_quarter_and_coverage(self):
        a=np.array([[[1,.95],[.95,1]],[[1,.96],[.96,1]]])
        n=np.full(a.shape,500)
        self.assertTrue(stable_matrix(a,n,.9)[0,1])
        n[1,0,1]=499
        self.assertFalse(stable_matrix(a,n,.9)[0,1])
        n[:]=500;a[1,0,1]=-.96
        self.assertFalse(stable_matrix(a,n,.9)[0,1])
        a[1,0,1]=np.nan
        self.assertFalse(stable_matrix(a,n,.9)[0,1])

    def test_constant_missing_and_boundary_changes(self):
        x=np.column_stack([np.arange(300,dtype=float),np.arange(300,dtype=float),np.ones(300),np.full(300,np.nan)])
        x[150:,1]+=100
        q=np.array(['A']*150+['B']*150)
        level,rank,delta,n,dn=matrices(x,q)
        self.assertTrue(np.isnan(level[2]).all())
        self.assertTrue(np.isnan(level[3]).all())
        self.assertEqual(n[0,1],300)
        self.assertEqual(dn[0,1],298)
        self.assertTrue(np.isnan(delta[0,1]))  # Both retained change series are constant.

    def test_independent_replay_handles_pairwise_constant_subset(self):
        rng=np.random.default_rng(73)
        x=rng.normal(size=(1000,3))
        x[:900,1]=0
        x[900:,0]=np.nan
        x[:,2]=2*x[:,0]
        r,n=independent_pearson(x)
        self.assertEqual(n[0,1],900)
        self.assertTrue(np.isnan(r[0,1]))
        self.assertAlmostEqual(r[0,2],1)


if __name__=='__main__':unittest.main()
