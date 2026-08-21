from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)),
        "mape_pct": float(np.mean(np.abs((y - p) / y)) * 100),
        "within_20pct": float(np.mean(np.abs(y - p) / y <= 0.20)),
    }


class AreaPropertyBaseline:
    """Median baseline fitted on training data only.

    It uses postcode district + property type where available, falls back to
    property type, then to the global training median.
    """

    def fit(self, train: pd.DataFrame) -> "AreaPropertyBaseline":
        self.global_median = float(train.price.median())
        self.type_median = train.groupby("property_type").price.median()
        self.area_type_median = train.groupby(["postcode_district", "property_type"]).price.median()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_frame(frame[["postcode_district", "property_type"]])
        area = np.asarray(self.area_type_median.reindex(keys), dtype=float)
        fallback = frame.property_type.map(self.type_median).fillna(self.global_median).to_numpy(float)
        return np.where(np.isfinite(area), area, fallback)


def conformal_radius(actual, predicted, coverage: float = 0.90) -> float:
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    residual = np.abs(np.asarray(actual, float) - np.asarray(predicted, float))
    return float(np.quantile(residual, coverage, method="higher"))


def interval_coverage(actual, predicted, radius: float, floor: float = 20_000.0, cap: float = 5_000_000.0) -> dict[str, float]:
    y = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    lower = np.clip(p - radius, floor, cap)
    upper = np.clip(p + radius, floor, cap)
    return {
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "average_width_pounds": float(np.mean(upper - lower)),
    }
