import unittest,importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('alarm_evidence',Path(__file__).resolve().parents[2]/'scripts/research/run_alarm_evidence_20261028.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class AlarmEvidenceTests(unittest.TestCase):
 def apply(self,x,**r):return m.transform(np.array(x,float),1,dict(original=False,**r),dict(merge_gap=0,min_dur=1)).tolist()
 def test_short_strong_survives_weak_removed(self):
  self.assertEqual(self.apply([0,1.2,0,3,0,1.2,1.2,1.2],family='strength_duration',duration=3,peak=2),[False,False,False,True,False,True,True,True])
 def test_hysteresis_connectivity_and_strict_threshold(self):
  self.assertEqual(self.apply([.6,1.5,.6,0,.6,2,.6],family='hysteresis',low=.5,high=1.5),[False,False,False,False,True,True,True])
 def test_duration_boundaries(self):
  self.assertEqual(self.apply([2,2,0,2],family='duration',duration=2),[True,True,False,False])
 def test_no_alarm_all_alarm(self):
  self.assertEqual(self.apply([0,0],family='hysteresis',low=.5,high=2),[False,False])
  self.assertEqual(self.apply([3,3],family='hysteresis',low=.5,high=2),[True,True])
 def test_selection_rejects_loss_and_ignores_test(self):
  a=dict(feasible=True,local_f05=.8,false_seconds=1,rule=dict(original=True,index=0),assessment=0)
  b=dict(a,feasible=False,local_f05=1,assessment=1)
  self.assertEqual(m.choose([b,a]),a)
 def test_grid_inventory(self):self.assertEqual(sum(len(m.candidates(f)) for f in m.FAMILIES),74)
