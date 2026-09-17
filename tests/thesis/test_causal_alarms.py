import unittest
import numpy as np
import pandas as pd
from esa_thesis.evaluation.causal_alarms import confirm, event_delays
from esa_thesis.evaluation.esa import EventEvaluator


class CausalAlarmTests(unittest.TestCase):
    def test_confirmation_does_not_backdate(self):
        score = np.array([0, 2, 2, 0, 2, 0])
        covered = np.ones(6, dtype=bool)
        np.testing.assert_array_equal(confirm(score, 1, 2, 2, covered), [0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(confirm(score, 1, 2, 3, covered), [0, 0, 1, 1, 1, 0])
        for end in range(1, 7):
            np.testing.assert_array_equal(confirm(score[:end], 1, 2, 3, covered[:end]),
                                          confirm(score, 1, 2, 3, covered)[:end])

    def test_warmup_does_not_supply_evidence(self):
        np.testing.assert_array_equal(confirm(np.ones(5), 0, 2, 2, [0, 0, 0, 1, 1]), [0, 0, 0, 0, 1])

    def test_delay_and_miss(self):
        times = pd.date_range('2000-01-01', periods=8, freq='30s')
        labels = pd.DataFrame({'ID': ['a', 'b'], 'Category': ['Anomaly']*2,
                               'StartTime': [times[1], times[6]], 'EndTime': [times[4], times[7]]})
        rows = event_delays(EventEvaluator(times, labels), [0, 0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(rows[0]['delay_seconds'], 60)
        self.assertIsNone(rows[1]['delay_seconds'])
        self.assertFalse(rows[1]['detected'])
