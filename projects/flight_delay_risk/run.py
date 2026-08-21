"""Command-line entry point for the flight-delay risk project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DataConfig, clean_flights
from src.evaluate import classification_metrics, expected_calibration_error, top_fraction_lift
from src.features import TARGET, add_risk_features
from src.model import threshold_for_capacity
from src.pipeline import PipelineConfig, run_pipeline


def self_test() -> None:
    raw = pd.DataFrame(
        {
            "FlightDate": pd.date_range("2026-01-01", periods=6),
            "Month": [1] * 6,
            "DayOfWeek": [4, 5, 6, 7, 1, 2],
            "Reporting_Airline": ["AA", "AA", "DL", "DL", "UA", "UA"],
            "Origin": ["JFK", "JFK", "ATL", "ATL", "SFO", "SFO"],
            "Dest": ["LAX", "LAX", "ORD", "ORD", "SEA", "SEA"],
            "CRSDepTime": [800, 900, 1000, 1100, 1200, 1300],
            "CRSArrTime": [1100, 1200, 1300, 1400, 1500, 1600],
            "CRSElapsedTime": [360, 360, 180, 180, 120, 120],
            "Distance": [2475, 2475, 606, 606, 679, 679],
            "ArrDelayMinutes": [0, 22, 7, 45, 16, 0],
            "Cancelled": [0] * 6,
            "Diverted": [0] * 6,
        }
    )
    clean = clean_flights(raw)
    featured = add_risk_features(clean)
    assert featured[TARGET].tolist() == [0, 1, 0, 1, 1, 0]

    y = np.asarray([0, 1, 0, 1, 1, 0])
    score = np.asarray([0.10, 0.80, 0.20, 0.90, 0.70, 0.05])
    threshold = threshold_for_capacity(score, 0.50)
    metrics = classification_metrics(y, score, threshold)
    lift = top_fraction_lift(y, score, 0.50)
    ece = expected_calibration_error(y, score)
    assert metrics["pr_auc"] > metrics["prevalence"]
    assert lift["lift"] >= 1.0
    assert 0.0 <= ece <= 1.0
    print("Flight-delay project self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and verify the 2026 flight-delay risk classifier")
    parser.add_argument("--self-test", action="store_true", help="Run fast offline checks only")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--alert-capacity", type=float, default=0.20)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/bts_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/flight_delay_risk"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    config = PipelineConfig(
        alert_capacity=args.alert_capacity,
        output_dir=args.output_dir,
        data=DataConfig(year=args.year, cache_dir=args.cache_dir),
    )
    result = run_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
