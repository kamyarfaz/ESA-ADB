import unittest
import numpy as np
import pandas as pd
from scripts.research.verify_context_expansion_20261113 import process, event_ids
from esa_thesis.evaluation.esa import EventEvaluator


class ContextExpansionVerifierTests(unittest.TestCase):
    def test_join_precedes_duration_filter(self):
        # Two singleton alarms with an allowed two-sample gap become one event.
        mask=np.array([0,1,0,0,1,0,0,0,1,0])
        np.testing.assert_array_equal(process(mask,2,4),[0,1,1,1,1,0,0,0,0,0])

    def test_edges_and_empty_mask(self):
        np.testing.assert_array_equal(process(np.array([1,1,0,0,1]),0,2),[1,1,0,0,0])
        np.testing.assert_array_equal(process(np.zeros(5),64,1),np.zeros(5))
        np.testing.assert_array_equal(process(np.ones(5),0,5),np.ones(5))

    def test_half_open_and_final_closed_ids(self):
        t=pd.date_range('2006-01-01',periods=5,freq='30s')
        ann=pd.DataFrame({'ID':['boundary','final'], 'StartTime':[t[2],t[4]],
                          'EndTime':[t[2],t[4]], 'Category':['Anomaly','Anomaly']})
        ev=EventEvaluator(t.asi8,ann)
        self.assertEqual(event_ids(ev,np.array([1,1,0,0,1])),
                         {'detected':['final'],'missed':['boundary']})


if __name__=='__main__':unittest.main()
