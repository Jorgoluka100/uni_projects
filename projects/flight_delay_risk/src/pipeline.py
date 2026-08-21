"""End-to-end training, evaluation and evidence generation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from .data import DataConfig, load_temporal_splits
from .evaluate import (
    calibration_table,
    capacity_curve,
    classification_metrics,
    expected_calibration_error,
)
from .features import CATEGORICAL_FEATURES, FEATURES, TARGET, add_risk_features
from .model import ModelConfig, build_model, threshold_for_capacity


@dataclass(frozen=True)
class PipelineConfig:
    alert_capacity: float = 0.20
    output_dir: Path = Path("artifacts/flight_delay_risk")
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()

    def validate(self) -> None:
        if not 0.01 <= self.alert_capacity <= 0.90:
            raise ValueError("alert_capacity must be between 0.01 and 0.90")
        self.data.validate()


def _constant_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    prevalence = float(y_train.mean())
    scores = np.full(len(y_test), prevalence, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y_test, scores)),
        "roc_auc": 0.5,
        "log_loss": float(log_loss(y_test, scores, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, scores)),
        "train_prevalence_score": prevalence,
        "test_prevalence": float(y_test.mean()),
    }


def run_pipeline(config: PipelineConfig = PipelineConfig()) -> dict[str, object]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw, valid_raw, test_raw = load_temporal_splits(config.data)
    train = add_risk_features(train_raw)
    valid = add_risk_features(valid_raw)
    test = add_risk_features(test_raw)

    x_train, y_train = train[FEATURES], train[TARGET].to_numpy()
    x_valid, y_valid = valid[FEATURES], valid[TARGET].to_numpy()
    x_test, y_test = test[FEATURES], test[TARGET].to_numpy()

    model = build_model(config.model)
    categorical_indices = [FEATURES.index(name) for name in CATEGORICAL_FEATURES]
    model.fit(
        x_train,
        y_train,
        cat_features=categorical_indices,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
    )

    validation_scores = model.predict_proba(x_valid)[:, 1]
    threshold = threshold_for_capacity(validation_scores, config.alert_capacity)
    test_scores = model.predict_proba(x_test)[:, 1]

    test_metrics = classification_metrics(y_test, test_scores, threshold)
    test_metrics["expected_calibration_error_10bin"] = expected_calibration_error(y_test, test_scores)
    baseline = _constant_baseline(y_train, y_test)
    curve = capacity_curve(y_test, test_scores)
    calibration = calibration_table(y_test, test_scores)

    audit = test[["FlightDate", "carrier", "origin", "dest", "route", "delay_minutes"]].copy()
    audit[TARGET] = y_test
    audit["risk_score"] = test_scores
    audit["alert"] = (test_scores >= threshold).astype(int)
    carrier_slices = (
        audit.groupby("carrier", observed=True)
        .agg(
            rows=(TARGET, "size"),
            prevalence=(TARGET, "mean"),
            mean_score=("risk_score", "mean"),
            alert_rate=("alert", "mean"),
        )
        .query("rows >= 500")
        .sort_values("prevalence", ascending=False)
    )

    model_path = config.output_dir / "flight_delay_catboost.cbm"
    model.save_model(model_path)
    curve.to_csv(config.output_dir / "capacity_curve.csv", index=False)
    calibration.to_csv(config.output_dir / "calibration_table.csv", index=False)
    carrier_slices.to_csv(config.output_dir / "carrier_risk_slices.csv")

    reloaded = CatBoostClassifier()
    reloaded.load_model(model_path)
    reloaded_scores = reloaded.predict_proba(x_test.head(500))[:, 1]
    reload_match = bool(np.allclose(reloaded_scores, test_scores[:500], atol=1e-10))

    verification_pass = bool(
        reload_match
        and test_metrics["pr_auc"] >= test_metrics["prevalence"]
        and 0.0 <= test_metrics["roc_auc"] <= 1.0
        and 0.0 <= threshold <= 1.0
    )
    if not verification_pass:
        raise AssertionError("One or more release checks failed")

    curve_records = [
        {
            "fraction": float(row.fraction),
            "rows": int(row.rows),
            "delay_rate": float(row.delay_rate),
            "population_prevalence": float(row.population_prevalence),
            "lift": float(row.lift),
        }
        for row in curve.itertuples(index=False)
    ]

    metadata: dict[str, object] = {
        "project": "Flight Delay Prediction and Risk Analysis",
        "verification_pass": verification_pass,
        "data_source": "US DOT Bureau of Transportation Statistics — On-Time Reporting Carrier Performance",
        "data_year": config.data.year,
        "task": "predict arrival delay >=15 minutes from schedule-time information",
        "split": {
            "train_months": list(config.data.train_months),
            "validation_months": list(config.data.validation_months),
            "test_months": list(config.data.test_months),
        },
        "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "validation_threshold": threshold,
        "alert_capacity_target": config.alert_capacity,
        "baseline_test": baseline,
        "classifier_test": test_metrics,
        "capacity_curve": curve_records,
        "model_config": asdict(config.model),
        "release_checks": {
            "saved_model_reload_matches": reload_match,
            "pr_auc_at_least_prevalence": bool(test_metrics["pr_auc"] >= test_metrics["prevalence"]),
            "threshold_in_probability_range": bool(0.0 <= threshold <= 1.0),
        },
        "limitations": [
            "Completed non-diverted flights only; cancellations and diversions need separate models.",
            "No weather, aircraft rotation, crew, congestion or live operational features are used.",
            "May 2026 is temporally held out, but future network regimes can drift.",
            "The alert threshold is selected on April for a fixed review capacity and should be monitored after deployment.",
            "This is operational decision support, not a guarantee that a specific flight will be delayed.",
        ],
    }
    (config.output_dir / "verification.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
