import numpy as np

from src.evaluation import (
    calibrate_residual_intervals,
    forecast_metrics,
    interval_metrics,
    last_value_baseline,
    seasonal_weekly_baseline,
)


def test_baseline_shapes_and_alignment():
    X = np.arange(3 * 60, dtype=float).reshape(3, 60, 1) + 1.0
    last = last_value_baseline(X)
    weekly = seasonal_weekly_baseline(X)
    assert last.shape == (3, 14)
    assert weekly.shape == (3, 14)
    assert np.array_equal(weekly[:, 0], X[:, 53, 0])
    assert np.array_equal(weekly[:, 7], X[:, 53, 0])


def test_perfect_forecast_metrics():
    y = np.array([[100.0, 110.0], [120.0, 130.0]])
    metrics = forecast_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape_pct"] == 0.0


def test_validation_calibrated_interval_has_valid_coverage():
    actual_val = np.array([[10.0, 20.0], [11.0, 19.0], [12.0, 18.0]])
    pred_val = np.array([[9.0, 18.0], [10.0, 20.0], [13.0, 17.0]])
    radius = calibrate_residual_intervals(actual_val, pred_val, coverage=0.9)
    test = interval_metrics(actual_val, pred_val, radius)
    assert radius.shape == (2,)
    assert 0.0 <= test["empirical_coverage"] <= 1.0
    assert test["average_width"] >= 0.0
