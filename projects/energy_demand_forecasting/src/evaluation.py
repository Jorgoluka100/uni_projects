from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def forecast_metrics(actual, predicted) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(y.ravel(), p.ravel())),
        "rmse": float(mean_squared_error(y.ravel(), p.ravel()) ** 0.5),
        "mape_pct": float(np.mean(np.abs((y - p) / y)) * 100),
    }


def last_value_baseline(X, horizon: int = 14) -> np.ndarray:
    history = np.asarray(X)
    return np.repeat(history[:, -1, 0, None], horizon, axis=1)


def seasonal_weekly_baseline(X, lookback: int = 60, horizon: int = 14) -> np.ndarray:
    history = np.asarray(X)
    return np.stack([history[:, lookback - 7 + (step % 7), 0] for step in range(horizon)], axis=1)


def calibrate_residual_intervals(actual_validation, predicted_validation, coverage: float = 0.90) -> np.ndarray:
    """Per-horizon absolute-residual quantiles fitted on validation only."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    residual = np.abs(np.asarray(actual_validation) - np.asarray(predicted_validation))
    return np.quantile(residual, coverage, axis=0, method="higher")


def interval_metrics(actual, predicted, radius) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    q = np.asarray(radius, dtype=float)
    lower = p - q
    upper = p + q
    return {
        "empirical_coverage": float(np.mean((y >= lower) & (y <= upper))),
        "average_width": float(np.mean(upper - lower)),
    }
