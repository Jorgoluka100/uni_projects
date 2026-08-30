"""Inference-time feature construction and release artifact loading."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .features import FEATURES


def build_inference_features(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Convert schedule-time request records into the exact training feature order."""
    frame = pd.DataFrame(list(records)).copy()
    required = {
        "flight_date",
        "carrier",
        "origin",
        "dest",
        "crs_dep_minutes",
        "crs_arr_minutes",
        "crs_elapsed_minutes",
        "distance_miles",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Inference request is missing fields: {missing}")
    if frame.empty:
        raise ValueError("At least one flight is required")

    date = pd.to_datetime(frame["flight_date"], errors="raise")
    out = pd.DataFrame(index=frame.index)
    out["month"] = date.dt.month.astype(int)
    out["day_of_week"] = (date.dt.dayofweek + 1).astype(int)
    out["carrier"] = frame["carrier"].astype(str).str.strip().str.upper()
    out["origin"] = frame["origin"].astype(str).str.strip().str.upper()
    out["dest"] = frame["dest"].astype(str).str.strip().str.upper()
    out["route"] = out["origin"] + "-" + out["dest"]

    for name in [
        "crs_dep_minutes",
        "crs_arr_minutes",
        "crs_elapsed_minutes",
        "distance_miles",
    ]:
        out[name] = pd.to_numeric(frame[name], errors="raise")

    out["day_of_month"] = date.dt.day.astype(int)
    out["dep_hour"] = (out["crs_dep_minutes"] // 60).astype(int)

    dep_angle = 2 * np.pi * out["crs_dep_minutes"] / (24 * 60)
    arr_angle = 2 * np.pi * out["crs_arr_minutes"] / (24 * 60)
    out["dep_sin"] = np.sin(dep_angle)
    out["dep_cos"] = np.cos(dep_angle)
    out["arr_sin"] = np.sin(arr_angle)
    out["arr_cos"] = np.cos(arr_angle)
    out["is_weekend"] = out["day_of_week"].isin([6, 7]).astype(int)
    out["carrier_route"] = out["carrier"] + "|" + out["route"]
    out["route_dep_block"] = out["route"] + "|h" + out["dep_hour"].astype(str)

    return out[FEATURES]


def load_release_artifacts(
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[CatBoostClassifier, dict[str, object]]:
    """Load the trained CatBoost model and its release metadata."""
    model_file = Path(
        model_path
        or os.getenv(
            "FLIGHT_DELAY_MODEL_PATH",
            "artifacts/flight_delay_risk/flight_delay_catboost.cbm",
        )
    )
    metadata_file = Path(
        metadata_path
        or os.getenv(
            "FLIGHT_DELAY_METADATA_PATH",
            "artifacts/flight_delay_risk/verification.json",
        )
    )
    if not model_file.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Release metadata not found: {metadata_file}")

    model = CatBoostClassifier()
    model.load_model(model_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("verification_pass") is not True:
        raise ValueError("Release metadata is not marked verification_pass=true")
    return model, metadata


def score_records(
    model: CatBoostClassifier,
    records: Iterable[Mapping[str, object]],
) -> np.ndarray:
    features = build_inference_features(records)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
