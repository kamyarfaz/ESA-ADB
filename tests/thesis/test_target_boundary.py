"""Scientific invariants for target-output pooling and conservative localization."""
import unittest
import numpy as np
import pandas as pd
from esa_thesis.evaluation.esa import EventEvaluator
from scripts.research import target_boundary_common as m
from scripts.research.verify_target_boundary_20261101 import coarse_replay, boundary_replay


class TargetBoundaryTests(unittest.TestCase):
    def test_trim_preserves_enclosing_evidence_without_fragmentation(self):
        proposal = np.array([0,1,1,1,1,1,1,1,0], bool)
        fine = np.array([50,0,0,2,0,2,0,0,50], float)
        rule = dict(original=False, cutoff=1, padding=0)
        actual = m.localize(proposal, fine, rule)
        self.assertEqual(actual.tolist(), [False,False,False,True,True,True,False,False,False])
        self.assertEqual(len(m.segments(actual)), 1)
        self.assertTrue(np.all(actual <= proposal))

    def test_no_crossing_strict_cutoff_and_padding(self):
        proposal = np.array([1,1,1,0,1,1,1,1], bool)
        fine = np.array([1,1,1,20,0,0,2,0], float)
        actual = m.localize(proposal, fine, dict(original=False, cutoff=1, padding=1))
        self.assertEqual(actual.tolist(), [True,True,True,False,False,True,True,True])
        self.assertEqual(len(m.segments(actual)), len(m.segments(proposal)))

    def test_empty_final_and_singleton_components(self):
        rule = dict(original=False, cutoff=1, padding=0)
        for proposal, fine in [([0,0], [5,5]), ([1], [2]), ([0,1,1], [0,0,2])]:
            a = m.localize(np.array(proposal), np.array(fine), rule)
            b = boundary_replay(np.array(proposal), np.array(fine), rule)
            np.testing.assert_array_equal(a, b)

    def test_target_pool_removes_auxiliary_direct_alarm_only(self):
        channel_errors = np.zeros((11, 5))
        channel_errors[10, 1] = 5
        channel_errors[4, 3] = 3
        self.assertEqual((channel_errors.max(0) > 1).tolist(), [False,True,False,True,False])
        self.assertEqual((channel_errors[:8].max(0) > 1).tolist(), [False,False,False,True,False])

    def test_retention_rejects_trim_that_keeps_wrong_core(self):
        times = pd.to_datetime(np.arange(9), unit='s').asi8
        ann = pd.DataFrame(dict(ID=['true'], StartTime=[pd.Timestamp(2,unit='s')],
                                EndTime=[pd.Timestamp(2,unit='s')], Category=['Anomaly']))
        ev = EventEvaluator(times, ann)
        p = np.array([0,1,1,1,1,1,1,0,0], bool)
        q = m.localize(p, np.array([0,0,0,0,0,5,0,0,0]), dict(original=False, cutoff=1, padding=0))
        # Check the ID constraint independently from the alarm-fraction cap.
        self.assertEqual(m.detected_ids(ev, p), ['true'])
        self.assertEqual(m.detected_ids(ev, q), [])

    def test_nominal_percentiles_exclude_neutral_intervals(self):
        times = pd.to_datetime(np.arange(10), unit='s').asi8
        ann = pd.DataFrame(dict(ID=['gap'], StartTime=[pd.Timestamp(3,unit='s')],
                                EndTime=[pd.Timestamp(5,unit='s')], Category=['Communication Gap']))
        mask = m.nominal_mask(EventEvaluator(times, ann))
        self.assertEqual(np.flatnonzero(~mask).tolist(), [3,4,5])
        candidates = m.boundary_candidates(np.where(mask, 1., 100.), mask)
        self.assertEqual(len(candidates), 7)
        self.assertTrue(all(r['cutoff'] == 1 for r in candidates[1:]))

    def test_selector_fallback_ties_and_assessment_irrelevance(self):
        original = dict(feasible=True, local={'EW_F_0.50':.9,'false_positive_seconds':1},
                        rule=dict(index=0, original=True), test_f05=0)
        candidate = dict(original, rule=dict(index=1, original=False), test_f05=1)
        rejected = dict(candidate, feasible=False, local={'EW_F_0.50':1.,'false_positive_seconds':0})
        self.assertEqual(m.choose([rejected, candidate, original]), original)
        self.assertIsNone(m.choose([rejected], allow_empty=True))
        with self.assertRaises(ValueError):
            m.choose([rejected])

    def test_independent_transform_parity_and_fixed_inventory(self):
        self.assertEqual(sum(len(m.family_candidates(f)) for f in m.FAMILIES), 122)
        for seed in range(8):
            rng = np.random.default_rng(seed)
            coarse, fine = rng.uniform(0, 3, 700), rng.uniform(0, 4, 700)
            boundary = dict(original=False, cutoff=2, padding=seed % 3)
            for family in m.FAMILIES:
                for rule in m.family_candidates(family):
                    a = m.localize(m.coarse_transform(coarse, 1, rule), fine, boundary)
                    b = boundary_replay(coarse_replay(coarse, 1, rule), fine, boundary)
                    np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
