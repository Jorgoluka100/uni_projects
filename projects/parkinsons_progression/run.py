"""Reproducible Parkinson's telemonitoring regression pipeline.

This project deliberately keeps the original university notebook untouched and
adds a stronger validation path beside it. Repeated observations from one person
must never appear in both train and holdout sets, so `subject#` is used only as a
grouping key and is excluded from model features.

This is an educational portfolio project. It is not a clinical diagnostic or
medical decision-support system.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OFFICIAL_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/telemonitoring/parkinsons_updrs.data"
)
TARGET = "total_UPDRS"
GROUP_COLUMN = "subject#"
DEFAULT_EXCLUDED_FEATURES = {GROUP_COLUMN, TARGET, "motor_UPDRS"}

REQUIRED_COLUMNS = {
    "subject#",
    "age",
    "sex",
    "test_time",
    "motor_UPDRS",
    "total_UPDRS",
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE",
}


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    r2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Parkinson's progression regression models."
    )
    parser.add_argument(
        "--data",
        default=OFFICIAL_UCI_URL,
        help="CSV path or URL. Defaults to the official UCI telemonitoring dataset.",
    )
    parser.add_argument(
        "--output",
        default="projects/parkinsons_progression/results/latest_metrics.json",
        help="Where to write the machine-readable evaluation report.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the subject-level holdout split.",
    )
    parser.add_argument(
        "--include-motor-updrs",
        action="store_true",
        help=(
            "Allow motor_UPDRS as an input feature. It is excluded by default so the "
            "main experiment focuses on voice/demographic/test-time measurements."
        ),
    )
    return parser.parse_args()


def load_data(source: str) -> pd.DataFrame:
    """Load a CSV from a local path or HTTP(S) URL."""
    if source.startswith(("http://", "https://")):
        return pd.read_csv(source)

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset contains no rows.")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove exact duplicates and rows that cannot be supervised/grouped.

    Feature-level missing values are retained and imputed inside each model
    pipeline, which prevents information from the holdout set influencing
    preprocessing.
    """
    before = len(df)
    duplicate_rows = int(df.duplicated().sum())
    cleaned = df.drop_duplicates().copy()

    missing_group_or_target = int(cleaned[[GROUP_COLUMN, TARGET]].isna().any(axis=1).sum())
    cleaned = cleaned.dropna(subset=[GROUP_COLUMN, TARGET]).copy()

    if cleaned[GROUP_COLUMN].nunique() < 3:
        raise ValueError("At least three unique subjects are required.")

    report = {
        "raw_rows": int(before),
        "exact_duplicates_removed": duplicate_rows,
        "rows_removed_missing_group_or_target": missing_group_or_target,
        "clean_rows": int(len(cleaned)),
        "unique_subjects": int(cleaned[GROUP_COLUMN].nunique()),
    }
    return cleaned, report


def feature_columns(df: pd.DataFrame, include_motor_updrs: bool) -> list[str]:
    excluded = set(DEFAULT_EXCLUDED_FEATURES)
    if include_motor_updrs:
        excluded.remove("motor_UPDRS")

    candidates = [column for column in df.columns if column not in excluded]
    non_numeric = [column for column in candidates if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"Expected numeric modelling columns, found: {non_numeric}")
    if not candidates:
        raise ValueError("No modelling features remain after exclusions.")
    return candidates


def subject_holdout(
    df: pd.DataFrame,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_idx, test_idx = next(
        splitter.split(df, y=df[TARGET], groups=df[GROUP_COLUMN])
    )

    train_subjects = set(df.iloc[train_idx][GROUP_COLUMN].tolist())
    test_subjects = set(df.iloc[test_idx][GROUP_COLUMN].tolist())
    overlap = train_subjects.intersection(test_subjects)
    if overlap:
        raise AssertionError(f"Subject leakage detected: {sorted(overlap)}")

    return train_idx, test_idx


def build_models(random_state: int) -> dict[str, Pipeline]:
    """Return transparent baseline and candidate scikit-learn pipelines."""
    return {
        "dummy_median": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=3,
                        min_samples_leaf=5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


def grouped_cross_validation(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> dict[str, Any]:
    unique_groups = int(groups.nunique())
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        raise ValueError("Not enough training subjects for grouped cross-validation.")

    fold_metrics: list[Metrics] = []
    splitter = GroupKFold(n_splits=n_splits)

    for fit_idx, validation_idx in splitter.split(X, y, groups):
        fold_model = clone(model)
        fold_model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        predictions = fold_model.predict(X.iloc[validation_idx])
        fold_metrics.append(regression_metrics(y.iloc[validation_idx], predictions))

    def mean(name: str) -> float:
        return float(np.mean([getattr(metric, name) for metric in fold_metrics]))

    def std(name: str) -> float:
        return float(np.std([getattr(metric, name) for metric in fold_metrics], ddof=0))

    return {
        "folds": n_splits,
        "mean": {"mae": mean("mae"), "rmse": mean("rmse"), "r2": mean("r2")},
        "std": {"mae": std("mae"), "rmse": std("rmse"), "r2": std("r2")},
        "per_fold": [asdict(metric) for metric in fold_metrics],
    }


def round_floats(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, digits) for item in value]
    return value


def run_experiment(
    df: pd.DataFrame,
    include_motor_updrs: bool,
    random_state: int,
) -> dict[str, Any]:
    validate_schema(df)
    cleaned, cleaning_report = clean_data(df)
    features = feature_columns(cleaned, include_motor_updrs)

    train_idx, test_idx = subject_holdout(cleaned, random_state)
    train = cleaned.iloc[train_idx].reset_index(drop=True)
    test = cleaned.iloc[test_idx].reset_index(drop=True)

    X_train = train[features]
    y_train = train[TARGET]
    train_groups = train[GROUP_COLUMN]
    X_test = test[features]
    y_test = test[TARGET]

    models = build_models(random_state)
    cv_reports: dict[str, Any] = {}

    for name, model in models.items():
        cv_reports[name] = grouped_cross_validation(
            model,
            X_train,
            y_train,
            train_groups,
        )

    candidate_names = [name for name in models if name != "dummy_median"]
    selected_name = min(
        candidate_names,
        key=lambda name: cv_reports[name]["mean"]["rmse"],
    )

    holdout_reports: dict[str, Any] = {}
    for name in ("dummy_median", selected_name):
        fitted = clone(models[name]).fit(X_train, y_train)
        predictions = fitted.predict(X_test)
        holdout_reports[name] = asdict(regression_metrics(y_test, predictions))

    train_subjects = set(train[GROUP_COLUMN].tolist())
    test_subjects = set(test[GROUP_COLUMN].tolist())

    report: dict[str, Any] = {
        "project": "Parkinson's Telemonitoring Progression",
        "purpose": "educational regression portfolio project; not for clinical use",
        "target": TARGET,
        "group_key": GROUP_COLUMN,
        "feature_policy": {
            "features": features,
            "excluded": sorted(set(cleaned.columns).difference(features + [TARGET])),
            "motor_UPDRS_included": include_motor_updrs,
        },
        "cleaning": cleaning_report,
        "validation": {
            "strategy": "80/20 GroupShuffleSplit by subject, then GroupKFold on training subjects",
            "random_state": random_state,
            "train_rows": int(len(train)),
            "holdout_rows": int(len(test)),
            "train_subjects": int(len(train_subjects)),
            "holdout_subjects": int(len(test_subjects)),
            "subject_overlap": int(len(train_subjects.intersection(test_subjects))),
        },
        "cross_validation": cv_reports,
        "selected_model": selected_name,
        "holdout": holdout_reports,
    }
    return round_floats(report)


def main() -> int:
    args = parse_args()
    df = load_data(args.data)
    report = run_experiment(
        df,
        include_motor_updrs=args.include_motor_updrs,
        random_state=args.random_state,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved evaluation report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
