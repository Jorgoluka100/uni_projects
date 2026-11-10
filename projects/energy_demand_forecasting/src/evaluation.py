from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _paired_arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.size == 0 or y.shape != p.shape:
        raise ValueError("actual and predicted must be non-empty arrays with identical shapes")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("actual and predicted must contain only finite values")
    if np.any(y == 0):
        raise ValueError("actual values must be non-zero for MAPE")
    return y, p


def forecast_metrics(actual, predicted) -> dict[str, float]:
    y, p = _paired_arrays(actual, predicted)
    return {
        "mae": float(mean_absolute_error(y.ravel(), p.ravel())),
        "rmse": float(mean_squared_error(y.ravel(), p.ravel()) ** 0.5),
        "mape_pct": float(np.mean(np.abs((y - p) / y)) * 100),
    }


def _validate_history(X, *, lookback: int, horizon: int) -> np.ndarray:
    history = np.asarray(X, dtype=float)
    if history.ndim != 3 or history.shape[0] == 0 or history.shape[2] < 1:
        raise ValueError("X must be a non-empty 3D array shaped (samples, lookback, features)")
    if lookback < 1 or history.shape[1] < lookback:
        raise ValueError("lookback must be positive and no greater than the supplied history")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not np.isfinite(history).all():
        raise ValueError("X must contain only finite values")
    return history


def last_value_baseline(X, horizon: int = 14) -> np.ndarray:
    history = _validate_history(X, lookback=1, horizon=horizon)
    return np.repeat(history[:, -1, 0, None], horizon, axis=1)


def seasonal_weekly_baseline(X, lookback: int = 60, horizon: int = 14) -> np.ndarray:
    if lookback < 7:
        raise ValueError("weekly seasonal baseline requires a lookback of at least 7 days")
    history = _validate_history(X, lookback=lookback, horizon=horizon)
    return np.stack([history[:, lookback - 7 + (step % 7), 0] for step in range(horizon)], axis=1)


def calibrate_residual_intervals(actual_validation, predicted_validation, coverage: float = 0.90) -> np.ndarray:
    """Per-horizon split-conformal residual radii fitted on validation only."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    y, p = _paired_arrays(actual_validation, predicted_validation)
    if y.ndim != 2 or y.shape[0] < 2:
        raise ValueError("validation arrays must be 2D with at least two forecast origins")
    residual = np.abs(y - p)
    rank = min(residual.shape[0], int(np.ceil((residual.shape[0] + 1) * coverage)))
    return np.partition(residual, rank - 1, axis=0)[rank - 1]


def interval_metrics(actual, predicted, radius) -> dict[str, float]:
    y, p = _paired_arrays(actual, predicted)
    q = np.asarray(radius, dtype=float)
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("radius must contain finite non-negative values")
    try:
        lower = p - q
        upper = p + q
    except ValueError as exc:
        raise ValueError("radius must be broadcastable across the forecast horizon") from exc
    return {
        "empirical_coverage": float(np.mean((y >= lower) & (y <= upper))),
        "average_width": float(np.mean(upper - lower)),
    }
