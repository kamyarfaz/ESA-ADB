import unittest
import numpy as np
from scripts.research.mission_cross_channel_review import held_values, stable, profile


class MissionCrossChannelReviewTests(unittest.TestCase):
    def test_causal_age_limited_hold(self):
        out = held_values(np.array([10, 20]), np.array([1., 2.]),
                          np.array([5, 10, 19, 20, 26]), [], max_age_ns=5)
        np.testing.assert_allclose(out, [np.nan, 1, np.nan, 2, np.nan], equal_nan=True)

    def test_cannot_carry_value_across_annotation(self):
        out = held_values(np.array([0, 10]), np.array([1., 2.]),
                          np.array([0, 3, 5, 8, 10]), [[3, 5]], max_age_ns=20)
        np.testing.assert_allclose(out, [1, np.nan, np.nan, np.nan, 2], equal_nan=True)

    def test_overlapping_annotations_use_prefix_max_end(self):
        out = held_values(np.array([6, 25]), np.array([1., 2.]),
                          np.array([8, 21, 25]), [[0, 20], [2, 3]], max_age_ns=40)
        np.testing.assert_allclose(out, [np.nan, np.nan, 2], equal_nan=True)

    def test_constants_and_unstable_pairs_are_not_matches(self):
        self.assertNotIn('acf_1', profile(np.ones(400)))
        self.assertFalse(stable([.99, np.nan], .9))
        self.assertFalse(stable([.99, -.99], .9))
        self.assertFalse(stable([.99, .89], .9))
        self.assertTrue(stable([-.99, -.94], .9))

    def test_profile_does_not_difference_across_gaps(self):
        x = np.r_[np.arange(200.), np.nan, 10000 + np.arange(200.)]
        p = profile(x)
        self.assertAlmostEqual(p['log_delta_median'], p['log_delta_p95'])


if __name__ == '__main__':
    unittest.main()
