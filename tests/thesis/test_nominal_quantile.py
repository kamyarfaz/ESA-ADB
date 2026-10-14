import unittest
import numpy as np
import pandas as pd
from esa_thesis.nominal_quantile import nominal_threshold, advancement

class NominalQuantileTests(unittest.TestCase):
    def test_closed_annotations_include_neutral_gaps(self):
        t=pd.date_range('2001-01-01',periods=5,freq='30s',tz='UTC')
        a=pd.DataFrame({'StartTime':[t[1],t[4]],'EndTime':[t[2],t[4]],'Category':['Anomaly','Communication Gap']})
        q,n=nominal_threshold(t.asi8,np.array([0,999,999,10,999]),a)
        self.assertEqual(n,2);self.assertAlmostEqual(q,9.99)
    def test_invalid_or_empty_nominal(self):
        t=pd.date_range('2001-01-01',periods=2,freq='30s',tz='UTC')
        a=pd.DataFrame({'StartTime':[t[0]],'EndTime':[t[-1]]})
        with self.assertRaises(ValueError):nominal_threshold(t.asi8,[0,1],a)
        with self.assertRaises(ValueError):nominal_threshold(t.asi8,[0,np.nan],a.iloc[:0])
    def test_gate_rejects_tie_or_more_false_alarm_duration(self):
        rows=[dict(fold=y,scope='system_all_channels',arm=arm,f05_mean=.2,fp_mean=0,false_seconds_mean=0) for y in (2001,2002) for arm in ('control','nominal_q999')]
        f=pd.DataFrame(rows);self.assertFalse(advancement(f)['advance'])
        f.loc[f.arm=='nominal_q999','f05_mean']=.3
        self.assertTrue(advancement(f)['advance'])
        f.loc[(f.fold==2002)&(f.arm=='nominal_q999'),'false_seconds_mean']=1
        self.assertFalse(advancement(f)['advance'])
