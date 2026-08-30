from __future__ import annotations

import unittest

from src.features import FEATURES
from src.inference import build_inference_features


class InferenceFeatureTests(unittest.TestCase):
    def test_request_builds_exact_training_schema(self) -> None:
        frame = build_inference_features(
            [
                {
                    "flight_date": "2026-05-17",
                    "carrier": "aa",
                    "origin": "jfk",
                    "dest": "lax",
                    "crs_dep_minutes": 480,
                    "crs_arr_minutes": 690,
                    "crs_elapsed_minutes": 390,
                    "distance_miles": 2475,
                }
            ]
        )
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(frame.loc[0, "carrier"], "AA")
        self.assertEqual(frame.loc[0, "origin"], "JFK")
        self.assertEqual(frame.loc[0, "dest"], "LAX")
        self.assertEqual(frame.loc[0, "route"], "JFK-LAX")
        self.assertEqual(frame.loc[0, "carrier_route"], "AA|JFK-LAX")
        self.assertEqual(frame.loc[0, "dep_hour"], 8)

    def test_missing_required_field_fails(self) -> None:
        with self.assertRaises(ValueError):
            build_inference_features(
                [
                    {
                        "flight_date": "2026-05-17",
                        "carrier": "AA",
                        "origin": "JFK",
                        "dest": "LAX",
                        "crs_dep_minutes": 480,
                        "crs_arr_minutes": 690,
                        "crs_elapsed_minutes": 390,
                    }
                ]
            )

    def test_empty_batch_fails(self) -> None:
        with self.assertRaises(ValueError):
            build_inference_features([])


if __name__ == "__main__":
    unittest.main()
