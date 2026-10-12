import unittest
import numpy as np
from esa_thesis.score_change import transform

class ScoreChangeTests(unittest.TestCase):
    def test_prefix_and_streaming_state(self):
        x=np.random.default_rng(42).exponential(size=1000)
        full,state=transform(x)
        left,s=transform(x[:350]);right,t=transform(x[350:],s)
        np.testing.assert_allclose(full,np.r_[left,right],atol=1e-12,rtol=0)
        self.assertAlmostEqual(state,t)
        np.testing.assert_array_equal(left,transform(np.r_[x[:350],np.ones(650)*100])[0][:350])
    def test_constant_and_both_directions(self):
        np.testing.assert_allclose(transform(np.ones(100)*4)[0],0,atol=1e-12)
        s=np.log1p(4.)
        self.assertGreater(transform([9.],s)[0][0],0)
        self.assertGreater(transform([1.],s)[0][0],0)
    def test_invalid_inputs(self):
        for x in ([],[-1],[np.nan],[np.inf]):
            with self.assertRaises(ValueError):transform(x)

if __name__=='__main__':unittest.main()
