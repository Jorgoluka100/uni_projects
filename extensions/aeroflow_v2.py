"""AeroFlow AI Engine v2 — leakage-safe 2026 flight-delay intelligence.

This extension keeps the useful idea from the original AeroFlow notebook but fixes
its main methodological weakness: preprocessing/model selection no longer sees the
holdout period. It also replaces the default synthetic-data claim with current,
official U.S. Bureau of Transportation Statistics (BTS) monthly on-time data.

Temporal design
---------------
Train:      January–March 2026
Validation: April 2026
Test:       May 2026 (untouched until final evaluation)

The task predicts arrival-delay minutes for completed, non-diverted flights from
schedule-time information only. Actual departure/arrival, taxi, wheels-off/on and
delay-cause fields are deliberately excluded because they would be unavailable at
schedule time or leak the outcome.

Run from the repository root with Python 3.10+ after installing:
    pip install pandas numpy scikit-learn catboost requests joblib

This file does not ship pre-computed metrics. Only promote metrics to the README or
CV after running the current code end to end and retaining the evidence.
"""

from __future__ import annotations

import io
import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BTS_BASE = "https://transtats.bts.gov/PREZIP"
BTS_TEMPLATE = (
    BTS_BASE
    + "/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2026_{month}.zip"
)

RAW_COLUMNS = {
    "FlightDate",
    "Month",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "ArrDelayMinutes",
    "Cancelled",
    "Diverted",
}

FEATURES = [
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
CAT_FEATURES = ["carrier", "origin", "dest", "route"]
TARGET = "delay_minutes"


@dataclass(frozen=True)
class RunConfig:
    train_months: tuple[int, ...] = (1, 2, 3)
    validation_months: tuple[int, ...] = (4,)
    test_months: tuple[int, ...] = (5,)
    max_rows_per_train_month: int | None = 120_000
    max_rows_validation: int | None = 120_000
    max_rows_test: int | None = 180_000
    seed: int = SEED
    artifact_dir: Path = Path("aeroflow_artifacts")
    request_timeout_seconds: int = 120
    interval_alpha: float = 0.10


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    """Convert BTS HHMM schedule values to minutes after midnight defensively."""
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    hours = (values // 100).clip(0, 23)
    minutes = (values % 100).clip(0, 59)
    return hours * 60 + minutes


def download_bts_month(month: int, timeout: int = 120) -> pd.DataFrame:
    """Download one official BTS 2026 pre-zipped on-time-performance CSV."""
    if month not in range(1, 13):
        raise ValueError(f"month must be 1..12; got {month}")

    url = BTS_TEMPLATE.format(month=month)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found inside {url}")

        with archive.open(csv_names[0]) as handle:
            frame = pd.read_csv(
                handle,
                usecols=lambda name: name.strip() in RAW_COLUMNS,
                low_memory=False,
            )

    frame.columns = [name.strip() for name in frame.columns]
    missing = sorted(RAW_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"BTS month {month} missing expected fields: {missing}")
    return frame


def clean_and_engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw fields and build schedule-time features without target leakage."""
    missing = sorted(RAW_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    data = frame.copy()
    data["FlightDate"] = pd.to_datetime(data["FlightDate"], errors="coerce")

    numeric_columns = [
        "Month",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "Distance",
        "ArrDelayMinutes",
        "Cancelled",
        "Diverted",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Regression target is defined only for completed, non-diverted flights.
    # Cancellation/diversion should be separate production risk models.
    data = data.loc[
        data["FlightDate"].notna()
        & data["ArrDelayMinutes"].notna()
        & data["Cancelled"].fillna(0).eq(0)
        & data["Diverted"].fillna(0).eq(0)
    ].copy()

    data["month"] = data["Month"].astype("Int64")
    data["day_of_week"] = data["DayOfWeek"].astype("Int64")
    data["carrier"] = data["Reporting_Airline"].fillna("UNKNOWN").astype(str)
    data["origin"] = data["Origin"].fillna("UNKNOWN").astype(str)
    data["dest"] = data["Dest"].fillna("UNKNOWN").astype(str)
    data["route"] = data["origin"] + "→" + data["dest"]

    data["crs_dep_minutes"] = hhmm_to_minutes(data["CRSDepTime"])
    data["crs_arr_minutes"] = hhmm_to_minutes(data["CRSArrTime"])
    data["crs_elapsed_minutes"] = data["CRSElapsedTime"].clip(lower=1)
    data["distance_miles"] = data["Distance"].clip(lower=0)

    angle = 2 * np.pi * data["crs_dep_minutes"] / (24 * 60)
    data["dep_sin"] = np.sin(angle)
    data["dep_cos"] = np.cos(angle)
    data[TARGET] = data["ArrDelayMinutes"].clip(lower=0)

    required = FEATURES + [TARGET, "FlightDate"]
    data = data.dropna(subset=required).reset_index(drop=True)

    if data.empty:
        raise ValueError("No usable completed flights remain after cleaning")
    if not data[TARGET].ge(0).all():
        raise AssertionError("Delay target must be non-negative")
    if not data["distance_miles"].ge(0).all():
        raise AssertionError("Distance must be non-negative")
    if not data["crs_elapsed_minutes"].gt(0).all():
        raise AssertionError("Scheduled elapsed time must be positive")

    return data[required]


def deterministic_sample(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed).copy()


def regression_metrics(y_true: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, prediction))),
        "median_absolute_error": float(median_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def route_carrier_baseline(
    train: pd.DataFrame,
    scoring: pd.DataFrame,
    smoothing: float = 50.0,
) -> np.ndarray:
    """Train-only smoothed route/carrier mean; unseen groups fall back to global mean."""
    global_mean = float(train[TARGET].mean())
    stats = (
        train.groupby(["route", "carrier"], observed=True)[TARGET]
        .agg(["mean", "count"])
        .reset_index()
    )
    stats["prediction"] = (
        stats["count"] * stats["mean"] + smoothing * global_mean
    ) / (stats["count"] + smoothing)
    mapping = dict(
        zip(zip(stats["route"], stats["carrier"]), stats["prediction"])
    )
    return np.asarray(
        [mapping.get((route, carrier), global_mean) for route, carrier in zip(scoring["route"], scoring["carrier"])],
        dtype=float,
    )


def temporal_integrity_check(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail fast if the intended month ordering is broken."""
    train_max = train["FlightDate"].max()
    valid_min, valid_max = valid["FlightDate"].min(), valid["FlightDate"].max()
    test_min = test["FlightDate"].min()

    if not train_max < valid_min:
        raise AssertionError(f"Training leaks into validation: {train_max=} {valid_min=}")
    if not valid_max < test_min:
        raise AssertionError(f"Validation leaks into test: {valid_max=} {test_min=}")


def load_temporal_splits(config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = sorted(set(config.train_months + config.validation_months + config.test_months))
    monthly: dict[int, pd.DataFrame] = {}

    for month in months:
        print(f"Downloading and cleaning BTS 2026 month {month}...")
        monthly[month] = clean_and_engineer(
            download_bts_month(month, timeout=config.request_timeout_seconds)
        )
        print(
            f"month={month} rows={len(monthly[month]):,} "
            f"dates={monthly[month]['FlightDate'].min().date()}..{monthly[month]['FlightDate'].max().date()}"
        )

    train = pd.concat(
        [
            deterministic_sample(monthly[m], config.max_rows_per_train_month, config.seed + m)
            for m in config.train_months
        ],
        ignore_index=True,
    )
    valid = pd.concat([monthly[m] for m in config.validation_months], ignore_index=True)
    test = pd.concat([monthly[m] for m in config.test_months], ignore_index=True)

    valid = deterministic_sample(valid, config.max_rows_validation, config.seed + 100)
    test = deterministic_sample(test, config.max_rows_test, config.seed + 200)

    temporal_integrity_check(train, valid, test)
    return train, valid, test


def build_model(config: RunConfig) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAE",
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=8.0,
        random_seed=config.seed,
        random_strength=0.5,
        od_type="Iter",
        od_wait=80,
        verbose=100,
        allow_writing_files=False,
    )


def conformal_interval(
    y_calibration: np.ndarray,
    calibration_prediction: np.ndarray,
    test_prediction: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    residual = np.abs(y_calibration - calibration_prediction)
    radius = float(np.quantile(residual, 1 - alpha, method="higher"))
    lower = np.clip(test_prediction - radius, 0, None)
    upper = test_prediction + radius
    return lower, upper, radius


def run(config: RunConfig = RunConfig()) -> dict[str, object]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    train, valid, test = load_temporal_splits(config)

    x_train, y_train = train[FEATURES].copy(), train[TARGET].copy()
    x_valid, y_valid = valid[FEATURES].copy(), valid[TARGET].copy()
    x_test, y_test = test[FEATURES].copy(), test[TARGET].copy()

    # Baseline 1: global median learned only from train.
    dummy = DummyRegressor(strategy="median")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_test = dummy.predict(np.zeros((len(y_test), 1)))

    # Baseline 2: route/carrier historical mean learned only from train.
    route_test = route_carrier_baseline(train, test)

    baseline_metrics = {
        "global_median": regression_metrics(y_test, dummy_test),
        "train_only_route_carrier_mean": regression_metrics(y_test, route_test),
    }

    model = build_model(config)
    categorical_indices = [FEATURES.index(name) for name in CAT_FEATURES]
    model.fit(
        x_train,
        y_train,
        cat_features=categorical_indices,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
    )

    valid_prediction = np.clip(model.predict(x_valid), 0, None)
    test_prediction = np.clip(model.predict(x_test), 0, None)
    test_metrics = regression_metrics(y_test, test_prediction)

    lower, upper, interval_radius = conformal_interval(
        y_valid.to_numpy(),
        valid_prediction,
        test_prediction,
        config.interval_alpha,
    )
    interval_coverage = float(
        np.mean((y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper))
    )

    actual_delay15 = (y_test.to_numpy() >= 15).astype(int)
    predicted_delay15 = (test_prediction >= 15).astype(int)
    decision_metrics = {
        "precision_at_15_minutes": float(
            precision_score(actual_delay15, predicted_delay15, zero_division=0)
        ),
        "recall_at_15_minutes": float(
            recall_score(actual_delay15, predicted_delay15, zero_division=0)
        ),
        "actual_delay15_rate": float(actual_delay15.mean()),
        "predicted_delay15_rate": float(predicted_delay15.mean()),
    }

    audit = test[["FlightDate", "carrier", "origin", "dest", "route", TARGET]].copy()
    audit["prediction"] = test_prediction
    audit["absolute_error"] = np.abs(audit[TARGET] - audit["prediction"])
    carrier_slices = (
        audit.groupby("carrier", observed=True)
        .agg(
            n=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            actual_mean_delay=(TARGET, "mean"),
            predicted_mean_delay=("prediction", "mean"),
        )
        .query("n >= 500")
        .sort_values("mae", ascending=False)
    )

    # Upper interval -> rounded planning signal. This is decision support, not a promise.
    planning_buffer = np.ceil(upper / 5) * 5

    model_path = config.artifact_dir / "aeroflow_catboost_2026.cbm"
    metadata_path = config.artifact_dir / "aeroflow_metadata.json"
    carrier_slice_path = config.artifact_dir / "aeroflow_carrier_slices.csv"
    model.save_model(model_path)
    carrier_slices.to_csv(carrier_slice_path)

    metadata: dict[str, object] = {
        "data_source": "US DOT Bureau of Transportation Statistics — On-Time Reporting Carrier Performance",
        "data_year": 2026,
        "train_months": list(config.train_months),
        "validation_months": list(config.validation_months),
        "test_months": list(config.test_months),
        "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
        "features": FEATURES,
        "categorical_features": CAT_FEATURES,
        "target": TARGET,
        "baseline_metrics": baseline_metrics,
        "catboost_test_metrics": test_metrics,
        "conformal": {
            "nominal_coverage": 1 - config.interval_alpha,
            "radius_minutes": interval_radius,
            "test_coverage": interval_coverage,
            "mean_width_minutes": float(np.mean(upper - lower)),
        },
        "decision_metrics": decision_metrics,
        "median_planning_buffer_minutes": float(np.median(planning_buffer)),
        "limitations": [
            "Completed non-diverted flights only; cancellation/diversion require separate risk models.",
            "Schedule-time model: no actual departure, taxi, arrival or delay-cause leakage features.",
            "May 2026 is a temporal test month, but future network/weather regimes can drift.",
            "The conformal interval uses a global April residual distribution and can under-cover subgroups.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Reload smoke test proves the saved model is usable.
    reloaded = CatBoostRegressor()
    reloaded.load_model(model_path)
    reloaded_prediction = np.clip(reloaded.predict(x_test.head(250)), 0, None)
    if not np.allclose(reloaded_prediction, test_prediction[:250], atol=1e-8):
        raise AssertionError("Reloaded model does not reproduce the saved model predictions")

    if not np.isfinite(test_prediction).all():
        raise AssertionError("Model produced non-finite predictions")
    if not 0 <= interval_coverage <= 1:
        raise AssertionError("Invalid interval coverage")
    if set(train["FlightDate"]).intersection(set(test["FlightDate"])):
        raise AssertionError("Train/test date overlap detected")

    print("\nAEROFLOW V2 ACCEPTANCE CHECKS PASSED")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    run()
