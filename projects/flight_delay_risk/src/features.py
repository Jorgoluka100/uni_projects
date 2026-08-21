"""Leakage-safe schedule-time feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "delay15"
BASE_FEATURES = [
    "month",
    "day_of_week",
    "carrier",
    "origin",
    "dest",
    "route",
    "crs_dep_minutes",
    "crs_arr_minutes",
    "crs_elapsed_minutes",
    "distance_miles",
    "dep_sin",
    "dep_cos",
]
EXTRA_NUMERIC = ["day_of_month", "dep_hour", "arr_sin", "arr_cos", "is_weekend"]
EXTRA_CATEGORICAL = ["carrier_route", "route_dep_block"]
FEATURES = BASE_FEATURES + EXTRA_NUMERIC + EXTRA_CATEGORICAL
CATEGORICAL_FEATURES = ["carrier", "origin", "dest", "route"] + EXTRA_CATEGORICAL


def add_risk_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the binary target and schedule-time features used by the classifier."""
    required = {
        "FlightDate",
        "delay_minutes",
        "month",
        "day_of_week",
        "carrier",
        "origin",
        "dest",
        "route",
        "crs_dep_minutes",
        "crs_arr_minutes",
        "crs_elapsed_minutes",
        "distance_miles",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature input is missing columns: {missing}")

    out = frame.copy()
    out[TARGET] = out["delay_minutes"].ge(15).astype(int)
    out["day_of_month"] = out["FlightDate"].dt.day.astype(int)
    out["dep_hour"] = (out["crs_dep_minutes"] // 60).astype(int)

    dep_angle = 2 * np.pi * out["crs_dep_minutes"] / (24 * 60)
    arr_angle = 2 * np.pi * out["crs_arr_minutes"] / (24 * 60)
    out["dep_sin"] = np.sin(dep_angle)
    out["dep_cos"] = np.cos(dep_angle)
    out["arr_sin"] = np.sin(arr_angle)
    out["arr_cos"] = np.cos(arr_angle)
    out["is_weekend"] = out["day_of_week"].isin([6, 7]).astype(int)
    out["carrier_route"] = out["carrier"].astype(str) + "|" + out["route"].astype(str)
    out["route_dep_block"] = out["route"].astype(str) + "|h" + out["dep_hour"].astype(str)

    if not set(out[TARGET].unique()).issubset({0, 1}):
        raise AssertionError("Binary target contains values outside {0, 1}")
    return out
