from __future__ import annotations

import io
import json
import math
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
for folder in (DATA, RESULTS, ARTIFACTS):
    folder.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
RANDOM_STATE = 42
TARGET = "cnt"
LEAKAGE_COLUMNS = {"casual", "registered"}


@dataclass
class SplitSummary:
    train_rows: int
    validation_rows: int
    test_rows: int
    train_end: str
    validation_end: str
    test_end: str


@dataclass
class Metrics:
    mae: float
    rmse: float
    rmsle: float
    r2: float


def download_hourly_data(force: bool = False) -> Path:
    target = DATA / "hour.csv"
    if target.exists() and not force:
        return target
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(name for name in archive.namelist() if name.endswith("hour.csv"))
        target.write_bytes(archive.read(member))
    return target


def load_data(path: Path | None = None) -> pd.DataFrame:
    path = path or download_hourly_data()
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"], errors="raise")
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df.sort_values("timestamp").reset_index(drop=True)


def audit_data(df: pd.DataFrame) -> dict[str, Any]:
    required = {
        "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual",
        "registered", "cnt", "timestamp",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["timestamp"].duplicated().any():
        raise ValueError("Hourly timestamp should be unique")
    if (df[TARGET] < 0).any():
        raise ValueError("Demand cannot be negative")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Rows must be chronologically ordered")
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "target_mean": float(df[TARGET].mean()),
        "target_median": float(df[TARGET].median()),
        "target_max": int(df[TARGET].max()),
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hr"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hr"] / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["mnth"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["mnth"] / 12)
    out["is_weekend"] = out["weekday"].isin([0, 6]).astype(int)
    out["is_commute_hour"] = out["hr"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    out["feels_like_gap"] = out["atemp"] - out["temp"]
    out["weather_discomfort"] = out["hum"] * out["windspeed"]
    out["trend_hour"] = np.arange(len(out), dtype=float)
    out["days_from_start"] = (out["timestamp"] - out["timestamp"].min()).dt.total_seconds() / 86400.0
    return out


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15):
    n = len(df)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()
    if not (train["timestamp"].max() < validation["timestamp"].min() <= validation["timestamp"].max() < test["timestamp"].min()):
        raise ValueError("Chronological split is not strictly ordered")
    summary = SplitSummary(
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        train_end=str(train["timestamp"].max()),
        validation_end=str(validation["timestamp"].max()),
        test_end=str(test["timestamp"].max()),
    )
    return train, validation, test, summary


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        TARGET,
        "instant",
        "dteday",
        "timestamp",
        *LEAKAGE_COLUMNS,
    }
    features = [c for c in df.columns if c not in excluded]
    if LEAKAGE_COLUMNS.intersection(features):
        raise AssertionError("Leakage columns entered model features")
    return features


def make_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    return df[features].copy(), df[TARGET].astype(float).copy()


def metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Metrics:
    truth = np.asarray(y_true, dtype=float)
    pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return Metrics(
        mae=float(mean_absolute_error(truth, pred)),
        rmse=float(math.sqrt(mean_squared_error(truth, pred))),
        rmsle=float(math.sqrt(mean_squared_error(np.log1p(truth), np.log1p(pred)))),
        r2=float(r2_score(truth, pred)),
    )


def seasonal_hour_baseline(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    lookup = train.groupby(["weekday", "hr"])[TARGET].median()
    global_median = float(train[TARGET].median())
    preds = []
    for row in target[["weekday", "hr"]].itertuples(index=False):
        preds.append(float(lookup.get((row.weekday, row.hr), global_median)))
    return np.asarray(preds)


def model_from_params(params: dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mae",
        **params,
    )


def tune_xgboost(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]):
    x_train, y_train = make_xy(train, features)
    x_val, y_val = make_xy(validation, features)
    grid = ParameterGrid(
        {
            "n_estimators": [300, 600],
            "max_depth": [4, 7],
            "learning_rate": [0.03, 0.07],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "min_child_weight": [1, 5],
            "reg_lambda": [1.0],
        }
    )
    trials: list[dict[str, Any]] = []
    best_model: XGBRegressor | None = None
    best_mae = float("inf")
    best_params: dict[str, Any] = {}

    for params in grid:
        model = model_from_params(params)
        model.fit(x_train, y_train, verbose=False)
        pred = model.predict(x_val)
        score = metrics(y_val, pred)
        trial = {**params, **asdict(score)}
        trials.append(trial)
        if score.mae < best_mae:
            best_mae = score.mae
            best_model = model
            best_params = dict(params)

    if best_model is None:
        raise RuntimeError("No XGBoost model trained")
    return best_model, best_params, pd.DataFrame(trials).sort_values("mae")


def refit_best(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], params: dict[str, Any]) -> XGBRegressor:
    combined = pd.concat([train, validation], axis=0).sort_values("timestamp")
    x, y = make_xy(combined, features)
    model = model_from_params(params)
    model.fit(x, y, verbose=False)
    return model


def error_slices(test: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    work = test[["timestamp", "hr", "weekday", "weathersit", TARGET]].copy()
    work["prediction"] = np.maximum(prediction, 0.0)
    work["absolute_error"] = np.abs(work[TARGET] - work["prediction"])
    work["demand_band"] = pd.cut(work[TARGET], bins=[-1, 50, 150, 300, np.inf], labels=["low", "medium", "high", "very_high"])
    slices = []
    for column in ["demand_band", "weathersit", "hr"]:
        grouped = work.groupby(column, observed=True).agg(
            rows=(TARGET, "size"),
            actual_mean=(TARGET, "mean"),
            prediction_mean=("prediction", "mean"),
            mae=("absolute_error", "mean"),
        ).reset_index()
        grouped.insert(0, "slice", column)
        grouped.rename(columns={column: "value"}, inplace=True)
        slices.append(grouped)
    return pd.concat(slices, ignore_index=True)


def feature_importance_table(model: XGBRegressor, features: list[str]) -> pd.DataFrame:
    values = model.feature_importances_
    return pd.DataFrame({"feature": features, "importance": values}).sort_values("importance", ascending=False)


def operational_decisions(prediction: np.ndarray, residual_mae: float) -> pd.DataFrame:
    pred = np.maximum(prediction, 0.0)
    lower = np.maximum(pred - 1.5 * residual_mae, 0.0)
    upper = pred + 1.5 * residual_mae
    band = np.select(
        [pred >= 400, pred >= 250, pred >= 120],
        ["critical", "high", "moderate"],
        default="normal",
    )
    action = np.select(
        [pred >= 400, pred >= 250, pred >= 120],
        ["pre-position bikes and staff", "rebalance inventory", "monitor capacity"],
        default="standard operations",
    )
    return pd.DataFrame(
        {
            "predicted_demand": pred,
            "uncertainty_lower": lower,
            "uncertainty_upper": upper,
            "demand_risk_band": band,
            "recommended_action": action,
        }
    )


def predict_next(model: XGBRegressor, row: pd.DataFrame, features: list[str], validation_mae: float) -> dict[str, Any]:
    value = float(max(model.predict(row[features])[0], 0.0))
    if value >= 400:
        band, action = "critical", "pre-position bikes and staff"
    elif value >= 250:
        band, action = "high", "rebalance inventory"
    elif value >= 120:
        band, action = "moderate", "monitor capacity"
    else:
        band, action = "normal", "standard operations"
    return {
        "predicted_hourly_demand": value,
        "uncertainty_proxy": [max(0.0, value - 1.5 * validation_mae), value + 1.5 * validation_mae],
        "demand_risk_band": band,
        "recommended_action": action,
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    raw = load_data()
    audit = audit_data(raw)
    data = add_features(raw)
    train, validation, test, split_summary = chronological_split(data)
    features = feature_columns(data)

    baseline_pred = seasonal_hour_baseline(train, validation)
    baseline_metrics = metrics(validation[TARGET], baseline_pred)

    _, best_params, trials = tune_xgboost(train, validation, features)
    validation_model = model_from_params(best_params)
    x_train, y_train = make_xy(train, features)
    x_validation, y_validation = make_xy(validation, features)
    validation_model.fit(x_train, y_train, verbose=False)
    validation_pred = validation_model.predict(x_validation)
    validation_metrics = metrics(y_validation, validation_pred)

    final_model = refit_best(train, validation, features, best_params)
    x_test, y_test = make_xy(test, features)
    test_pred = final_model.predict(x_test)
    test_metrics = metrics(y_test, test_pred)

    slices = error_slices(test, test_pred)
    importance = feature_importance_table(final_model, features)
    decisions = operational_decisions(test_pred, validation_metrics.mae)
    predictions = test[["timestamp", TARGET]].reset_index(drop=True).join(decisions)

    trials.to_csv(RESULTS / "tuning_trials.csv", index=False)
    slices.to_csv(RESULTS / "error_slices.csv", index=False)
    importance.to_csv(RESULTS / "feature_importance.csv", index=False)
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)
    joblib.dump({"model": final_model, "features": features, "validation_mae": validation_metrics.mae}, ARTIFACTS / "xgboost_bike_demand.joblib")

    bundle = joblib.load(ARTIFACTS / "xgboost_bike_demand.joblib")
    parity = np.allclose(bundle["model"].predict(x_test.iloc[:20]), final_model.predict(x_test.iloc[:20]))
    if not parity:
        raise RuntimeError("Saved model parity check failed")

    payload = {
        "dataset_audit": audit,
        "split": asdict(split_summary),
        "features": features,
        "best_parameters": best_params,
        "validation_baseline": asdict(baseline_metrics),
        "validation_xgboost": asdict(validation_metrics),
        "test_xgboost": asdict(test_metrics),
        "validation_mae_gain_vs_baseline": float(baseline_metrics.mae - validation_metrics.mae),
        "top_features": importance.head(12).to_dict(orient="records"),
        "example_decision": predict_next(final_model, test.iloc[[0]], features, validation_metrics.mae),
        "limitations": [
            "Historical location-specific data; no live station inventory or current road conditions.",
            "Residual-based uncertainty proxy is not a calibrated predictive interval.",
            "Operational thresholds are illustrative and require real capacity-cost calibration.",
        ],
    }
    save_json(RESULTS / "metrics.json", payload)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
