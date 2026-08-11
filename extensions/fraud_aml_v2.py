"""Financial Fraud / AML v2 — decision-focused, leakage-aware screening.

This extension strengthens the restored notebook without pretending synthetic labels
are evidence about real financial crime. The default demo deliberately generates a
time-ordered synthetic transaction stream so the engineering and evaluation design
can be exercised end to end.

Key upgrades
------------
- explicit transaction/data contract;
- chronological train/validation/test split;
- train-only preprocessing;
- naive prevalence baseline + gradient-boosted model;
- PR-AUC / ROC-AUC plus precision and recall at review-capacity;
- cost-sensitive threshold search on validation only;
- untouched test evaluation;
- monthly performance slices for drift-style diagnostics;
- saved model, threshold, metrics and audit CSV;
- reload smoke test.

This is a portfolio/research demonstration, not an AML system and not a substitute
for regulated monitoring, sanctions screening, transaction monitoring or human
investigation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET = "is_fraud"
TIME_COL = "event_time"
ID_COLS = ["transaction_id", "user_id"]
NUMERIC = [
    "amount",
    "hour",
    "user_24h_txn_count",
    "user_24h_amount",
    "amount_vs_user_median",
]
CATEGORICAL = ["merchant_category", "country", "device_type", "is_international"]
FEATURES = NUMERIC + CATEGORICAL


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    n_transactions: int = 120_000
    review_capacity: float = 0.02
    false_positive_cost: float = 8.0
    false_negative_cost: float = 500.0
    artifact_dir: Path = Path("fraud_aml_artifacts")

    def validate(self) -> None:
        if not 0 < self.review_capacity < 1:
            raise ValueError("review_capacity must be in (0,1)")
        if self.false_positive_cost < 0 or self.false_negative_cost <= 0:
            raise ValueError("costs must be non-negative and FN cost positive")


def generate_synthetic_transactions(config: Config) -> pd.DataFrame:
    """Generate a deterministic time-ordered demo stream.

    The generator intentionally includes slow concept drift so temporal evaluation
    has meaning. It is *not* calibrated to a real bank and must never be described
    as real fraud prevalence.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_transactions
    start = pd.Timestamp("2025-01-01", tz="UTC")
    timestamps = start + pd.to_timedelta(np.arange(n) * 4, unit="m")

    users = rng.integers(10_000, 14_000, n)
    categories = np.array(["groceries", "electronics", "travel", "luxury", "services"])
    countries = np.array(["GB", "FR", "DE", "ES", "US", "AE"])
    devices = np.array(["ios", "android", "web"])

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(n, dtype=np.int64),
            "user_id": users.astype(np.int64),
            "event_time": timestamps,
            "amount": rng.lognormal(mean=4.1, sigma=1.0, size=n),
            "merchant_category": rng.choice(categories, size=n, p=[0.35, 0.18, 0.14, 0.08, 0.25]),
            "country": rng.choice(countries, size=n, p=[0.68, 0.07, 0.06, 0.05, 0.09, 0.05]),
            "device_type": rng.choice(devices, size=n, p=[0.48, 0.40, 0.12]),
        }
    )
    df["hour"] = df[TIME_COL].dt.hour.astype(int)
    df["is_international"] = (df["country"] != "GB").astype(str)

    ordered = df.sort_values([TIME_COL, "transaction_id"]).copy()
    ordered["user_24h_txn_count"] = (
        ordered.set_index(TIME_COL)
        .groupby("user_id")["transaction_id"]
        .rolling("24h", closed="both")
        .count()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )
    ordered["user_24h_amount"] = (
        ordered.set_index(TIME_COL)
        .groupby("user_id")["amount"]
        .rolling("24h", closed="both")
        .sum()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    prior_median = (
        ordered.groupby("user_id", sort=False)["amount"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
    )
    global_median = float(ordered["amount"].median())
    ordered["amount_vs_user_median"] = ordered["amount"] / prior_median.fillna(global_median).clip(lower=1.0)

    month_index = (ordered[TIME_COL].dt.year - 2025) * 12 + ordered[TIME_COL].dt.month - 1
    logit = (
        -5.1
        + 0.85 * np.log1p(ordered["amount"]) / np.log(1000)
        + 1.15 * (ordered["country"] != "GB").astype(float)
        + 0.70 * ordered["hour"].isin([0, 1, 2, 3, 4]).astype(float)
        + 0.95 * (ordered["amount_vs_user_median"] > 4).astype(float)
        + 0.45 * (ordered["merchant_category"] == "luxury").astype(float)
        + 0.05 * month_index.astype(float)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    ordered[TARGET] = rng.binomial(1, prob.clip(0.0005, 0.8)).astype(int)
    return ordered.reset_index(drop=True)


def validate_frame(df: pd.DataFrame) -> None:
    required = set([TIME_COL, TARGET] + ID_COLS + FEATURES)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if df.empty:
        raise ValueError("input is empty")
    if df["transaction_id"].duplicated().any():
        raise ValueError("transaction_id must be unique")
    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("is_fraud must be binary 0/1")
    if not pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
        raise TypeError("event_time must be datetime")
    if (df["amount"] < 0).any():
        raise ValueError("amount cannot be negative")


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values([TIME_COL, "transaction_id"]).reset_index(drop=True)
    train_end = int(len(ordered) * 0.60)
    valid_end = int(len(ordered) * 0.80)
    train = ordered.iloc[:train_end].copy()
    valid = ordered.iloc[train_end:valid_end].copy()
    test = ordered.iloc[valid_end:].copy()
    if not (train[TIME_COL].max() < valid[TIME_COL].min() <= valid[TIME_COL].max() < test[TIME_COL].min()):
        raise AssertionError("temporal split overlap detected")
    return train, valid, test


def make_model(seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=240,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, score))


def evaluate_scores(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": safe_auc(y, score),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "flag_rate": float(pred.mean()),
        "prevalence": float(y.mean()),
    }


def threshold_for_review_capacity(score: np.ndarray, capacity: float) -> float:
    q = float(np.quantile(score, 1.0 - capacity))
    return min(max(q, 0.0), 1.0)


def choose_cost_threshold(
    y_valid: np.ndarray,
    score_valid: np.ndarray,
    fp_cost: float,
    fn_cost: float,
    review_capacity: float,
) -> dict[str, float]:
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 199), score_valid])
    best: dict[str, float] | None = None
    for threshold in candidates:
        pred = score_valid >= threshold
        flag_rate = float(pred.mean())
        if flag_rate > review_capacity * 1.25:
            continue
        fp = int(((pred == 1) & (y_valid == 0)).sum())
        fn = int(((pred == 0) & (y_valid == 1)).sum())
        cost = fp * fp_cost + fn * fn_cost
        row = {"threshold": float(threshold), "cost": float(cost), "flag_rate": flag_rate}
        if best is None or row["cost"] < best["cost"]:
            best = row
    if best is None:
        threshold = threshold_for_review_capacity(score_valid, review_capacity)
        best = {"threshold": threshold, "cost": float("nan"), "flag_rate": float((score_valid >= threshold).mean())}
    return best


def monthly_slices(df: pd.DataFrame, score: np.ndarray, threshold: float) -> pd.DataFrame:
    audit = df[[TIME_COL, TARGET]].copy()
    audit["score"] = score
    audit["month"] = audit[TIME_COL].dt.to_period("M").astype(str)
    rows: list[dict[str, float | str | int]] = []
    for month, group in audit.groupby("month", sort=True):
        y = group[TARGET].to_numpy()
        s = group["score"].to_numpy()
        metrics = evaluate_scores(y, s, threshold)
        rows.append({"month": month, "n": int(len(group)), **metrics})
    return pd.DataFrame(rows)


def run(config: Config = Config()) -> dict[str, object]:
    config.validate()
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_transactions(config)
    validate_frame(df)
    train, valid, test = temporal_split(df)

    model = make_model(config.seed)
    model.fit(train[FEATURES], train[TARGET])

    valid_score = model.predict_proba(valid[FEATURES])[:, 1]
    threshold_choice = choose_cost_threshold(
        valid[TARGET].to_numpy(),
        valid_score,
        config.false_positive_cost,
        config.false_negative_cost,
        config.review_capacity,
    )
    threshold = float(threshold_choice["threshold"])

    test_score = model.predict_proba(test[FEATURES])[:, 1]
    test_metrics = evaluate_scores(test[TARGET].to_numpy(), test_score, threshold)

    prevalence_score = np.full(len(test), float(train[TARGET].mean()))
    baseline_metrics = evaluate_scores(test[TARGET].to_numpy(), prevalence_score, 0.5)

    slices = monthly_slices(test, test_score, threshold)
    audit = test[[TIME_COL, "transaction_id", "user_id", TARGET]].copy()
    audit["score"] = test_score
    audit["flagged_for_review"] = (test_score >= threshold).astype(int)

    model_path = config.artifact_dir / "fraud_screening_pipeline.joblib"
    joblib.dump(model, model_path)
    joblib.dump({"threshold": threshold, "features": FEATURES}, config.artifact_dir / "decision_policy.joblib")
    slices.to_csv(config.artifact_dir / "monthly_slices.csv", index=False)
    audit.to_csv(config.artifact_dir / "test_audit.csv", index=False)

    reloaded = joblib.load(model_path)
    check = reloaded.predict_proba(test[FEATURES].head(25))[:, 1]
    if not np.allclose(check, test_score[:25]):
        raise AssertionError("reloaded model predictions changed")

    metadata = {
        "scope": "synthetic methodology demonstration only",
        "config": {**asdict(config), "artifact_dir": str(config.artifact_dir)},
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "date_ranges": {
            "train": [str(train[TIME_COL].min()), str(train[TIME_COL].max())],
            "valid": [str(valid[TIME_COL].min()), str(valid[TIME_COL].max())],
            "test": [str(test[TIME_COL].min()), str(test[TIME_COL].max())],
        },
        "validation_threshold_choice": threshold_choice,
        "baseline_test": baseline_metrics,
        "model_test": test_metrics,
        "limitations": [
            "Synthetic labels do not establish real-world fraud or AML performance.",
            "Fraud screening and AML investigation are different operational/regulatory problems.",
            "A production system needs institution-specific data, typologies, fairness analysis, investigator feedback and monitoring.",
        ],
    }
    (config.artifact_dir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=120_000)
    parser.add_argument("--artifact-dir", type=Path, default=Path("fraud_aml_artifacts"))
    args = parser.parse_args()
    result = run(Config(n_transactions=args.rows, artifact_dir=args.artifact_dir))
    print(json.dumps(result["model_test"], indent=2))


if __name__ == "__main__":
    main()
