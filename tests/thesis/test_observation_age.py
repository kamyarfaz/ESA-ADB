import unittest
import numpy as np
from esa_thesis.observation_age import observation_age


class ObservationAgeTests(unittest.TestCase):
    def test_causal_age_and_initial_unavailability(self):
        native = np.array([5, 35, 245], dtype=np.int64)*10**9
        grid = np.array([0, 30, 35, 60, 240, 245], dtype=np.int64)*10**9
        feature, available, age = observation_age(native, grid, 30)
        np.testing.assert_array_equal(available, [False, True, True, True, True, True])
        np.testing.assert_array_equal(age, [0, 25, 0, 25, 205, 0])
        np.testing.assert_allclose(feature, np.log1p(age/30), rtol=1e-6)
        # A future observation cannot alter features of earlier timestamps.
        shorter = observation_age(native[:-1], grid[:-1], 30)
        np.testing.assert_array_equal(feature[:-1], shorter[0])

    def test_empty_history_and_invalid_inputs(self):
        _, available, age = observation_age(np.array([], dtype=np.int64), np.arange(4), 90)
        self.assertFalse(available.any())
        self.assertFalse(age.any())
        for native in [np.array([2, 1]), np.array([1, 1]), np.array([1.5])]:
            with self.assertRaises(ValueError):
                observation_age(native, np.arange(4), 30)
        with self.assertRaises(ValueError):
            observation_age(np.arange(4), np.arange(4), 0)


if __name__ == '__main__':
    unittest.main()
