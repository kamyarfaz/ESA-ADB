import unittest,importlib.util
from pathlib import Path
import numpy as np
from esa_thesis.thresholds import postprocess
spec=importlib.util.spec_from_file_location('temporal',Path(__file__).resolve().parents[2]/'scripts/research/run_temporal_postprocessing_20261028.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TemporalTests(unittest.TestCase):
 def test_merge_then_filter_and_edges(self):
  # Two individually short bursts survive after joining; outside zeros stay zero.
  self.assertEqual(postprocess(np.array([0,1,1,0,1,1,0]),1,5).tolist(),[0,1,1,1,1,1,0])
  self.assertEqual(postprocess(np.array([0,1,1,0,1,1,0]),0,3).tolist(),[0]*7)
 def test_gap_and_duration_inclusive_boundaries(self):
  self.assertEqual(postprocess(np.array([1,0,0,1]),2,4).tolist(),[1]*4)
  self.assertEqual(postprocess(np.array([1,0,0,1]),1,2).tolist(),[0]*4)
 def test_selection_never_uses_assessment(self):
  a=dict(feasible=True,local_f05=.8,false_seconds=10,original=True,gap=0,duration=1,assessment_f05=0)
  b=dict(a,original=False,gap=1,assessment_f05=1)
  self.assertEqual(m.choose([b,a]),a)
  self.assertEqual(m.choose([a,dict(b,local_f05=.9,feasible=False)]),a)
  self.assertEqual(m.choose([a,dict(b,false_seconds=9)])['gap'],1)
