from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


@dataclass
class DatasetAudit:
    rows: int
    columns: int
    target_classes: int
    duplicate_rows: int
    missing_cells: int
    min_class_size: int
    max_class_size: int


@dataclass
class Evaluation:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    log_loss: float
    review_rate: float
    accepted_accuracy: float | None


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    bunch = load_wine(as_frame=True)
    frame = bunch.frame.copy()
    target = frame.pop("target").astype(int)
    names = [str(name) for name in bunch.target_names]
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    return frame, target, names


def audit_dataset(x: pd.DataFrame, y: pd.Series) -> DatasetAudit:
    counts = y.value_counts()
    audit = DatasetAudit(
        rows=len(x),
        columns=x.shape[1],
        target_classes=int(y.nunique()),
        duplicate_rows=int(x.duplicated().sum()),
        missing_cells=int(x.isna().sum().sum()),
        min_class_size=int(counts.min()),
        max_class_size=int(counts.max()),
    )
    if audit.rows < 100:
        raise ValueError("Dataset unexpectedly small")
    if audit.target_classes < 2:
        raise ValueError("Classification requires multiple classes")
    if audit.missing_cells:
        raise ValueError("Built-in benchmark should not contain missing values")
    return audit


def descriptive_profile(x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "feature_means": {k: float(v) for k, v in x.mean().items()},
        "feature_std": {k: float(v) for k, v in x.std().items()},
        "feature_min": {k: float(v) for k, v in x.min().items()},
        "feature_max": {k: float(v) for k, v in x.max().items()},
    }
    return profile


def build_pipeline(n_neighbors: int = 5, weights: str = "distance", p: int = 2) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)),
        ]
    )


def tune_model(x_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            "model__n_neighbors": list(range(3, 22, 2)),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(x_train, y_train)
    return search


def evaluate_confidence_policy(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    truth: np.ndarray,
    threshold: float,
) -> tuple[float, float | None]:
    confidence = probabilities.max(axis=1)
    accepted = confidence >= threshold
    review_rate = float((~accepted).mean())
    accepted_accuracy = None
    if accepted.any():
        accepted_accuracy = float(accuracy_score(truth[accepted], predictions[accepted]))
    return review_rate, accepted_accuracy


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[Evaluation, dict[str, Any]]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    review_rate, accepted_accuracy = evaluate_confidence_policy(
        probabilities,
        predictions,
        y_test.to_numpy(),
        threshold=0.70,
    )
    evaluation = Evaluation(
        accuracy=float(accuracy_score(y_test, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro")),
        log_loss=float(log_loss(y_test, probabilities)),
        review_rate=review_rate,
        accepted_accuracy=accepted_accuracy,
    )
    detail = {
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "confidence": probabilities.max(axis=1).tolist(),
    }
    return evaluation, detail


def scaling_ablation(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scaled = build_pipeline(n_neighbors=7)
    unscaled = KNeighborsClassifier(n_neighbors=7, weights="distance")
    scaled_score = cross_val_score(scaled, x_train, y_train, scoring="f1_macro", cv=cv).mean()
    raw_score = cross_val_score(unscaled, x_train, y_train, scoring="f1_macro", cv=cv).mean()
    return {
        "scaled_macro_f1": float(scaled_score),
        "unscaled_macro_f1": float(raw_score),
        "scaling_gain": float(scaled_score - raw_score),
    }


def feature_importance(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, float | str]]:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="f1_macro",
        n_repeats=25,
        random_state=RANDOM_STATE,
    )
    rows = [
        {"feature": feature, "importance": float(score)}
        for feature, score in zip(x_test.columns, result.importances_mean)
    ]
    return sorted(rows, key=lambda row: float(row["importance"]), reverse=True)


def inspect_neighbours(model: Pipeline, x: pd.DataFrame, row: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    scaler: StandardScaler = model.named_steps["scale"]
    knn: KNeighborsClassifier = model.named_steps["model"]
    x_scaled = scaler.transform(x)
    row_scaled = scaler.transform(row)
    distances, indices = knn.kneighbors(row_scaled, n_neighbors=top_n)
    output: list[dict[str, Any]] = []
    for distance, idx in zip(distances[0], indices[0]):
        output.append({"row_index": int(idx), "distance": float(distance)})
    return output


def predict_one(model: Pipeline, row: pd.DataFrame, target_names: list[str]) -> dict[str, Any]:
    probabilities = model.predict_proba(row)[0]
    prediction = int(model.predict(row)[0])
    confidence = float(probabilities.max())
    return {
        "predicted_class": prediction,
        "predicted_label": target_names[prediction],
        "confidence": confidence,
        "manual_review": bool(confidence < 0.70),
        "class_probabilities": {
            target_names[i]: float(probabilities[i]) for i in range(len(probabilities))
        },
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    x, y, target_names = load_dataset()
    audit = audit_dataset(x, y)
    profile = descriptive_profile(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    search = tune_model(x_train, y_train)
    best_model: Pipeline = search.best_estimator_
    evaluation, evaluation_detail = evaluate_model(best_model, x_test, y_test)
    ablation = scaling_ablation(x_train, y_train)
    importance = feature_importance(best_model, x_test, y_test)
    example = predict_one(best_model, x_test.iloc[[0]], target_names)
    neighbours = inspect_neighbours(best_model, x_train, x_test.iloc[[0]], top_n=5)

    cv_table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_table[
        ["params", "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]
    ].head(20).to_csv(RESULTS / "cv_results.csv", index=False)

    payload = {
        "dataset_audit": asdict(audit),
        "evaluation": asdict(evaluation),
        "best_parameters": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "scaling_ablation": ablation,
        "top_permutation_features": importance[:10],
        "example_prediction": example,
        "example_neighbours": neighbours,
        "confusion_matrix": evaluation_detail["confusion_matrix"],
        "limitations": [
            "Compact benchmark dataset rather than a live production quality stream.",
            "Neighbour distances can become less informative as dimensionality grows.",
            "A production threshold should be chosen from explicit quality-review costs.",
        ],
    }
    save_json(RESULTS / "metrics.json", payload)
    save_json(RESULTS / "dataset_profile.json", profile)
    joblib.dump(best_model, ARTIFACTS / "knn_quality_pipeline.joblib")

    reloaded: Pipeline = joblib.load(ARTIFACTS / "knn_quality_pipeline.joblib")
    original_pred = best_model.predict(x_test)
    reloaded_pred = reloaded.predict(x_test)
    if not np.array_equal(original_pred, reloaded_pred):
        raise RuntimeError("Saved-model parity check failed")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
