import unittest
import numpy as np
from esa_thesis.evaluation.compare_channel_normalization import fit_scale, normalized_max

class NormalizationTests(unittest.TestCase):
    def test_scale_and_zero_floor(self):
        x=np.tile(np.arange(100,dtype=float)[:,None],(1,8));x[:,0]=0
        s=fit_scale(x)
        self.assertEqual(s[0],1e-8)
        np.testing.assert_allclose(s[1:],98.01)

    def test_channel_unit_invariance(self):
        rng=np.random.default_rng(4);x=rng.uniform(.1,3,(200,8));score=rng.uniform(.1,3,(8,20));factors=np.arange(1,9)
        original=normalized_max(score,fit_scale(x))
        np.testing.assert_allclose(original,normalized_max(score*factors[:,None],fit_scale(x*factors)))

    def test_fit_not_affected_by_calibration_and_invalid_values(self):
        x=np.ones((100,8));before=fit_scale(x)
        normalized_max(np.ones((8,10))*1e6,before)
        np.testing.assert_array_equal(before,fit_scale(x))
        for bad in [np.full((100,8),np.nan),np.full((100,8),-1),np.ones((10,8))]:
            with self.assertRaises(ValueError):fit_scale(bad)
        with self.assertRaises(ValueError):normalized_max(np.ones((8,10)),np.zeros(8))

if __name__=='__main__':unittest.main()
