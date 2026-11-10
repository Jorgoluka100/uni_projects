from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _paired_arrays(actual, predicted, *, positive_actual: bool = False) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.size == 0 or y.shape != p.shape:
        raise ValueError("actual and predicted must be non-empty arrays with identical shapes")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("actual and predicted must contain only finite values")
    if positive_actual and np.any(y <= 0):
        raise ValueError("actual house prices must be strictly positive for percentage metrics")
    return y, p


def regression_metrics(actual, predicted) -> dict[str, float]:
    y, p = _paired_arrays(actual, predicted, positive_actual=True)
    percentage_error = np.abs((y - p) / y)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)),
        "mape_pct": float(np.mean(percentage_error) * 100),
        "within_20pct": float(np.mean(percentage_error <= 0.20)),
    }


class AreaPropertyBaseline:
    """Median baseline fitted on training data only.

    It uses postcode district + property type where available, falls back to
    property type, then to the global training median.
    """

    def fit(self, train: pd.DataFrame) -> "AreaPropertyBaseline":
        required = {"price", "property_type", "postcode_district"}
        missing = sorted(required - set(train.columns))
        if missing or train.empty:
            raise ValueError(f"training data must be non-empty and include {sorted(required)}; missing={missing}")
        if not np.isfinite(train["price"].to_numpy(dtype=float)).all() or (train["price"] <= 0).any():
            raise ValueError("training prices must be finite and strictly positive")
        self.global_median = float(train.price.median())
        self.type_median = train.groupby("property_type").price.median()
        self.area_type_median = train.groupby(["postcode_district", "property_type"]).price.median()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "global_median"):
            raise RuntimeError("fit must be called before predict")
        required = {"property_type", "postcode_district"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"prediction data is missing columns: {missing}")
        keys = pd.MultiIndex.from_frame(frame[["postcode_district", "property_type"]])
        area = np.asarray(self.area_type_median.reindex(keys), dtype=float)
        fallback = frame.property_type.map(self.type_median).fillna(self.global_median).to_numpy(float)
        return np.where(np.isfinite(area), area, fallback)


def conformal_radius(actual, predicted, coverage: float = 0.90) -> float:
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    y, p = _paired_arrays(actual, predicted)
    residual = np.abs(y - p).ravel()
    rank = min(residual.size, math.ceil((residual.size + 1) * coverage))
    return float(np.partition(residual, rank - 1)[rank - 1])


def interval_coverage(
    actual,
    predicted,
    radius: float,
    floor: float = 20_000.0,
    cap: float = 5_000_000.0,
) -> dict[str, float]:
    y, p = _paired_arrays(actual, predicted)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and non-negative")
    if not np.isfinite([floor, cap]).all() or floor >= cap:
        raise ValueError("interval bounds must be finite and floor must be below cap")
    lower = np.clip(p - radius, floor, cap)
    upper = np.clip(p + radius, floor, cap)
    return {
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "average_width_pounds": float(np.mean(upper - lower)),
    }
