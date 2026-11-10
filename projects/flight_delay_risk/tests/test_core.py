from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data import DataConfig, hhmm_to_minutes
from src.evaluate import expected_calibration_error, top_fraction_lift
from src.model import threshold_for_capacity


class CoreTests(unittest.TestCase):
    def test_hhmm_conversion(self) -> None:
        values = pd.Series([0, 5, 930, 2359, None])
        converted = hhmm_to_minutes(values)
        self.assertTrue(np.allclose(converted.iloc[:4], [0, 5, 570, 1439]))
        self.assertTrue(np.isnan(converted.iloc[4]))

    def test_invalid_hhmm_values_are_not_coerced_to_midnight(self) -> None:
        values = pd.Series([1260, 2360, 2500, -100, 2400, 930.5])
        converted = hhmm_to_minutes(values)
        self.assertTrue(np.isnan(converted.iloc[[0, 1, 2, 3, 5]]).all())
        self.assertEqual(converted.iloc[4], 0.0)

    def test_data_config_rejects_duplicate_and_non_temporal_months(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(train_months=(1, 1)).validate()
        with self.assertRaises(ValueError):
            DataConfig(train_months=(2, 3), validation_months=(1,), test_months=(4,)).validate()

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
