from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data import hhmm_to_minutes
from src.evaluate import expected_calibration_error, top_fraction_lift
from src.model import threshold_for_capacity


class CoreTests(unittest.TestCase):
    def test_hhmm_conversion(self) -> None:
        values = pd.Series([0, 5, 930, 2359, None])
        self.assertEqual(hhmm_to_minutes(values).tolist(), [0, 5, 570, 1439, 0])

    def test_capacity_threshold_is_valid_probability(self) -> None:
        score = np.asarray([0.1, 0.2, 0.3, 0.8, 0.9])
        threshold = threshold_for_capacity(score, 0.40)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_lift_rewards_good_ranking(self) -> None:
        y = np.asarray([1, 1, 0, 0])
        score = np.asarray([0.9, 0.8, 0.2, 0.1])
        result = top_fraction_lift(y, score, 0.50)
        self.assertEqual(result["delay_rate"], 1.0)
        self.assertEqual(result["lift"], 2.0)

    def test_calibration_error_range(self) -> None:
        y = np.asarray([0, 0, 1, 1])
        score = np.asarray([0.1, 0.2, 0.8, 0.9])
        ece = expected_calibration_error(y, score, bins=4)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


if __name__ == "__main__":
    unittest.main()
