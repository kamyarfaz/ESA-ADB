import unittest
import numpy as np
from esa_thesis.preprocessing_audit import held_values, restore_candidates


class PreprocessingAuditTests(unittest.TestCase):
    def test_restoration_matches_literal_bin_loop(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            t = np.sort(rng.choice(1000, 100, replace=False))
            labels = rng.integers(0, 4, len(t))
            expected = []
            for b in np.unique(t//30):
                ix = np.flatnonzero(t//30==b)
                pos = ix[np.isin(labels[ix], [1, 2])]
                if len(ix)>1 and labels[ix[-1]]==0 and len(pos):
                    expected.append(((b+1)*30, pos[-1]))
            grid, positions = restore_candidates(t, labels, step_ns=30)
            self.assertEqual(list(zip(grid, positions)), expected)

    def test_exact_grid_point_and_initial_fill(self):
        actual = held_values([2, 30, 33, 65], [1, 5, 9, 2], [0, 30, 60, 90])
        np.testing.assert_array_equal(actual, [1, 5, 9, 2])

    def test_gap_and_last_annotated_do_not_restore(self):
        times, positions = restore_candidates([1, 2, 31, 32, 61, 62],
                                              [3, 0, 0, 1, 2, 0], step_ns=30)
        np.testing.assert_array_equal(times, [90])
        np.testing.assert_array_equal(positions, [4])

    def test_invalid_raw_data_rejected(self):
        with self.assertRaises(ValueError):
            restore_candidates([2, 1], [0, 1])
        with self.assertRaises(ValueError):
            restore_candidates([1, 2], [0, 4])
        with self.assertRaises(ValueError):
            held_values([1, 2], [1, np.nan], [1, 2])


if __name__=='__main__':
    unittest.main()
