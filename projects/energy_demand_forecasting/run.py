from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation import (
    calibrate_residual_intervals,
    forecast_metrics,
    interval_metrics,
    last_value_baseline,
    seasonal_weekly_baseline,
)

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results" / "verified_metrics.json"


def self_test() -> None:
    X = np.arange(2 * 60, dtype=float).reshape(2, 60, 1) + 100.0
    last = last_value_baseline(X)
    weekly = seasonal_weekly_baseline(X)
    assert last.shape == (2, 14)
    assert weekly.shape == (2, 14)
    assert np.all(last[:, 0] == X[:, -1, 0])
    assert np.all(weekly[:, 0] == X[:, 53, 0])

    actual = np.array([[10.0, 20.0], [12.0, 18.0]])
    predicted = np.array([[11.0, 18.0], [11.0, 20.0]])
    metrics = forecast_metrics(actual, predicted)
    assert metrics["mae"] == 1.5
    radius = calibrate_residual_intervals(actual, predicted, coverage=0.9)
    interval = interval_metrics(actual, predicted, radius)
    assert 0 <= interval["empirical_coverage"] <= 1
    print("Energy forecasting self-test passed.")


def check_evidence() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert report["verification_pass"] is True
    assert report["source_rows"] == 4383
    assert report["forecast_horizon_days"] == 14
    assert report["train_windows"] == 2995
    assert report["validation_windows"] == 644
    assert report["test_windows"] == 645
    model = report["test_metrics"]["tensorflow_lstm"]
    seasonal = report["test_metrics"]["weekly_seasonal"]
    assert model["mae_gwh"] < seasonal["mae_gwh"]
    assert report["mae_improvement_vs_best_baseline_pct"] > 18.0
    assert report["artifact"]["max_reload_prediction_delta"] == 0.0
    print("Retained energy forecast evidence passed.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-evidence", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.check_evidence:
        check_evidence()
    if not args.self_test and not args.check_evidence:
        parser.error("choose --self-test or --check-evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
