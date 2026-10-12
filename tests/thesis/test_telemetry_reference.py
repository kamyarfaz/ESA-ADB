import unittest
import numpy as np
from esa_thesis.telemetry_reference import deviations,fit_scale

class TelemetryReferenceTests(unittest.TestCase):
 def test_stream_and_isolation(self):
  x=np.random.default_rng(1).normal(size=(1000,8));e,s=deviations(x)
  a,t=deviations(x[:500]);b,u=deviations(x[500:],t)
  np.testing.assert_allclose(e,np.vstack([a,b]),atol=1e-12,rtol=0)
  z=x.copy();z[:,7]+=10
  np.testing.assert_array_equal(e[:,:7],deviations(z)[0][:,:7])
  np.testing.assert_array_equal(a,deviations(np.vstack([x[:500],x[500:]*100]))[0][:500])
 def test_scale_excludes_annotations_and_flat_floor(self):
  e=np.array([[0.,1.],[0.,2.],[1e6,1e6]])
  s=fit_scale(e,np.array([True,True,False]));self.assertEqual(s[0],1e-6);self.assertLess(s[1],2.01)
 def test_constant(self):
  np.testing.assert_allclose(deviations(np.ones((100,8)))[0],0,atol=1e-12)

if __name__=='__main__':unittest.main()
