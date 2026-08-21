from __future__ import annotations

import unittest

import numpy as np

from src.evaluation import classification_metrics, expected_calibration_error, selective_metrics, softmax
from src.parity import parity_report


class EvaluationTests(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self) -> None:
        logits = np.asarray([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
        probabilities = softmax(logits, temperature=1.5)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_temperature_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            softmax(np.asarray([[1.0, 0.0]]), temperature=0.0)

    def test_selective_policy_can_improve_accepted_accuracy(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        probabilities = np.asarray(
            [
                [0.95, 0.05],
                [0.10, 0.90],
                [0.51, 0.49],
                [0.55, 0.45],
            ]
        )
        base = classification_metrics(labels, probabilities)
        policy = selective_metrics(probabilities, labels, threshold=0.80)
        self.assertGreater(policy["selective_accuracy"], base["accuracy"])
        self.assertEqual(policy["coverage"], 0.5)
        self.assertEqual(policy["review_rate"], 0.5)

    def test_ece_is_zero_for_perfectly_calibrated_binary_fixture(self) -> None:
        labels = np.asarray([0, 1])
        probabilities = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(expected_calibration_error(probabilities, labels, bins=5), 0.0)

    def test_parity_report_detects_close_outputs(self) -> None:
        reference = np.asarray([[1.0, 2.0, 3.0]])
        candidate = reference + 1e-7
        result = parity_report(reference, candidate, atol=1e-5, rtol=1e-5)
        self.assertTrue(result["pass"])
        self.assertLess(result["max_abs_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
