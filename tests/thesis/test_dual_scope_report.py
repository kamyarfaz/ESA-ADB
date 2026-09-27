import unittest
import numpy as np
import pandas as pd
from scripts.research.dual_scope_report import scope_evaluator

class ScopeTests(unittest.TestCase):
    def test_other_channel_hits_not_local_but_remain_neutral(self):
        times=pd.date_range('2000-01-01',periods=11,freq='s').asi8
        labels=pd.DataFrame({'ID':['shared','shared','outside'],'Channel':['channel_40','channel_50','channel_50'],
            'StartTime':pd.to_datetime(times[[2,5,8]]),'EndTime':pd.to_datetime(times[[3,6,9]]),'Category':['Anomaly']*3})
        pred=np.zeros(11);pred[5]=1;pred[8]=1
        system=scope_evaluator(times,labels).score(pred)
        local=scope_evaluator(times,labels,['channel_40']).score(pred)
        self.assertEqual((system['TPe'],system['FNe']),(2,0))
        self.assertEqual((local['TPe'],local['FNe']),(0,1))
        self.assertEqual(local['FPe'],0)
        self.assertEqual(local['false_positive_seconds'],0)
        self.assertEqual(local['nominal_seconds'],system['nominal_seconds'])
        self.assertEqual(labels.Category.tolist(),['Anomaly']*3)

if __name__=='__main__':unittest.main()
