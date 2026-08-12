"""AeroFlow v3 — operational 15-minute delay-risk classification on official 2026 BTS data.

Why v3 exists
-------------
The v2 schedule-only *regression* is retained as an honest negative result: arrival
delay minutes are zero-inflated and a global zero/median predictor is extremely hard
to beat on MAE without weather or live operational features. Rather than hiding that
finding, this iteration reframes the business decision to something schedule-time
features can support more naturally: **which flights are at elevated risk of arriving
15+ minutes late?**

Data/evaluation remain strict:
- Train: January–March 2026
- Validation/threshold selection: April 2026
- Untouched test: May 2026
- Official US DOT/BTS data only
- Schedule-time features only; no actual departure/arrival or delay-cause leakage
- Validation-only alert threshold, fixed before the May test is opened
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aeroflow_v2 import RunConfig, load_temporal_splits

SEED = 42
TARGET = "delay15"
BASE_FEATURES = [
    "month", "day_of_week", "carrier", "origin", "dest", "route",
    "crs_dep_minutes", "crs_arr_minutes", "crs_elapsed_minutes", "distance_miles",
    "dep_sin", "dep_cos",
]
EXTRA_NUMERIC = [
    "day_of_month", "dep_hour", "arr_sin", "arr_cos", "is_weekend",
]
EXTRA_CATEGORICAL = ["carrier_route", "route_dep_block"]
FEATURES = BASE_FEATURES + EXTRA_NUMERIC + EXTRA_CATEGORICAL
CAT_FEATURES = ["carrier", "origin", "dest", "route"] + EXTRA_CATEGORICAL


@dataclass(frozen=True)
class Config:
    alert_capacity: float = 0.20
    seed: int = SEED
    artifact_dir: Path = Path("aeroflow_delay_risk_artifacts")

    def validate(self) -> None:
        if not 0.01 <= self.alert_capacity <= 0.90:
            raise ValueError("alert_capacity must be in [0.01, 0.90]")


def add_classification_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[TARGET] = out["delay_minutes"].ge(15).astype(int)
    out["day_of_month"] = out["FlightDate"].dt.day.astype(int)
    out["dep_hour"] = (out["crs_dep_minutes"] // 60).astype(int)
    arr_angle = 2 * np.pi * out["crs_arr_minutes"] / (24 * 60)
    out["arr_sin"] = np.sin(arr_angle)
    out["arr_cos"] = np.cos(arr_angle)
    out["is_weekend"] = out["day_of_week"].isin([6, 7]).astype(int)
    out["carrier_route"] = out["carrier"].astype(str) + "|" + out["route"].astype(str)
    out["route_dep_block"] = out["route"].astype(str) + "|h" + out["dep_hour"].astype(str)
    return out


def threshold_for_capacity(scores: np.ndarray, capacity: float) -> float:
    threshold = float(np.quantile(scores, 1.0 - capacity))
    return min(max(threshold, 0.0), 1.0)


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    prevalence = float(y.mean())
    precision = float(precision_score(y, pred, zero_division=0))
    recall = float(recall_score(y, pred, zero_division=0))
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "log_loss": float(log_loss(y, score, labels=[0, 1])),
        "brier": float(brier_score_loss(y, score)),
        "precision": precision,
        "recall": recall,
        "flag_rate": float(pred.mean()),
        "prevalence": prevalence,
        "precision_lift_vs_prevalence": float(precision / prevalence) if prevalence else float("nan"),
    }


def top_fraction_lift(y: np.ndarray, score: np.ndarray, fraction: float) -> dict[str, float]:
    n = max(1, int(round(len(y) * fraction)))
    order = np.argsort(-score, kind="mergesort")[:n]
    rate = float(y[order].mean())
    prevalence = float(y.mean())
    return {
        "fraction": fraction,
        "rows": n,
        "delay_rate": rate,
        "population_prevalence": prevalence,
        "lift": float(rate / prevalence) if prevalence else float("nan"),
    }


def build_model(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.055,
        depth=8,
        l2_leaf_reg=10.0,
        random_seed=seed,
        random_strength=0.4,
        od_type="Iter",
        od_wait=70,
        verbose=100,
        allow_writing_files=False,
    )


def run(config: Config = Config()) -> dict[str, object]:
    config.validate()
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the already audited official-BTS loader and exact Jan–May temporal split.
    base_config = RunConfig(artifact_dir=config.artifact_dir / "unused_regression_artifacts")
    train_raw, valid_raw, test_raw = load_temporal_splits(base_config)
    train = add_classification_features(train_raw)
    valid = add_classification_features(valid_raw)
    test = add_classification_features(test_raw)

    x_train, y_train = train[FEATURES], train[TARGET].to_numpy()
    x_valid, y_valid = valid[FEATURES], valid[TARGET].to_numpy()
    x_test, y_test = test[FEATURES], test[TARGET].to_numpy()

    model = build_model(config.seed)
    cat_indices = [FEATURES.index(name) for name in CAT_FEATURES]
    model.fit(
        x_train,
        y_train,
        cat_features=cat_indices,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
    )

    valid_score = model.predict_proba(x_valid)[:, 1]
    threshold = threshold_for_capacity(valid_score, config.alert_capacity)
    test_score = model.predict_proba(x_test)[:, 1]

    test_metrics = classification_metrics(y_test, test_score, threshold)
    baseline_prevalence = float(y_train.mean())
    baseline_score = np.full(len(y_test), baseline_prevalence, dtype=float)
    baseline = {
        "pr_auc": float(average_precision_score(y_test, baseline_score)),
        "roc_auc": 0.5,
        "log_loss": float(log_loss(y_test, baseline_score, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, baseline_score)),
        "train_prevalence_score": baseline_prevalence,
        "test_prevalence": float(y_test.mean()),
    }

    ranking_lift = {
        "top_10pct": top_fraction_lift(y_test, test_score, 0.10),
        "top_20pct": top_fraction_lift(y_test, test_score, 0.20),
    }

    audit = test[["FlightDate", "carrier", "origin", "dest", "route", "delay_minutes"]].copy()
    audit[TARGET] = y_test
    audit["risk_score"] = test_score
    audit["alert"] = (test_score >= threshold).astype(int)
    carrier_slices = (
        audit.groupby("carrier", observed=True)
        .agg(
            n=(TARGET, "size"),
            prevalence=(TARGET, "mean"),
            mean_score=("risk_score", "mean"),
            alert_rate=("alert", "mean"),
        )
        .query("n >= 500")
        .sort_values("prevalence", ascending=False)
    )

    model_path = config.artifact_dir / "aeroflow_delay15_catboost_2026.cbm"
    slices_path = config.artifact_dir / "carrier_risk_slices.csv"
    metadata_path = config.artifact_dir / "metrics.json"
    model.save_model(model_path)
    carrier_slices.to_csv(slices_path)

    # Fresh model reload parity.
    reloaded = CatBoostClassifier()
    reloaded.load_model(model_path)
    reload_score = reloaded.predict_proba(x_test.head(500))[:, 1]
    if not np.allclose(reload_score, test_score[:500], atol=1e-10):
        raise AssertionError("reloaded classifier changed probabilities")

    if test_metrics["pr_auc"] < test_metrics["prevalence"]:
        raise AssertionError("classifier PR-AUC fell below prevalence baseline")
    if not 0 <= test_metrics["roc_auc"] <= 1:
        raise AssertionError("invalid ROC-AUC")
    if not 0 <= threshold <= 1:
        raise AssertionError("invalid validation-selected threshold")

    metadata: dict[str, object] = {
        "project": "AeroFlow delay-risk v3",
        "data_source": "US DOT Bureau of Transportation Statistics — On-Time Reporting Carrier Performance",
        "data_year": 2026,
        "task": "predict probability of arrival delay >=15 minutes using schedule-time information",
        "train_months": [1, 2, 3],
        "validation_months": [4],
        "test_months": [5],
        "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
        "features": FEATURES,
        "categorical_features": CAT_FEATURES,
        "validation_threshold": threshold,
        "alert_capacity_target": config.alert_capacity,
        "baseline_test": baseline,
        "classifier_test": test_metrics,
        "ranking_lift": ranking_lift,
        "regression_v2_finding": "Schedule-only delay-minute regression did not beat the global-median MAE baseline; retained as a negative result rather than hidden.",
        "limitations": [
            "Completed non-diverted flights only; cancellation/diversion need separate models.",
            "No weather, aircraft rotation, crew, congestion or live operational features are used.",
            "May 2026 is temporally held out, but future network regimes can drift.",
            "The alert threshold is selected on April for a fixed review/operations capacity and requires monitoring after deployment.",
            "This is operational decision support, not a guarantee that a specific flight will be delayed.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("AEROFLOW DELAY-RISK V3 ACCEPTANCE CHECKS PASSED")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    run()
