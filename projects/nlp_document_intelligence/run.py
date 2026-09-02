from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)
RANDOM_STATE = 42


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    log_loss: float | None
    auto_route_rate: float
    auto_route_accuracy: float | None


def normalise_text(text: str) -> str:
    text = str(text).replace("\x00", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " EMAIL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_corpus():
    kwargs = {
        "remove": ("headers", "footers", "quotes"),
        "shuffle": True,
        "random_state": RANDOM_STATE,
    }
    train = fetch_20newsgroups(subset="train", **kwargs)
    test = fetch_20newsgroups(subset="test", **kwargs)
    x_train = [normalise_text(text) for text in train.data]
    x_test = [normalise_text(text) for text in test.data]
    y_train = np.asarray(train.target, dtype=int)
    y_test = np.asarray(test.target, dtype=int)
    target_names = [str(name) for name in train.target_names]
    return x_train, y_train, x_test, y_test, target_names


def corpus_audit(texts: list[str], labels: np.ndarray, target_names: list[str]) -> dict[str, Any]:
    lengths = np.asarray([len(text.split()) for text in texts], dtype=int)
    empty = int(sum(not text.strip() for text in texts))
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) != len(target_names):
        raise ValueError("Target-name count does not match observed labels")
    if empty / max(len(texts), 1) > 0.10:
        raise ValueError("Unexpectedly high empty-document rate")
    return {
        "documents": int(len(texts)),
        "classes": int(len(target_names)),
        "empty_documents": empty,
        "median_words": float(np.median(lengths)),
        "p95_words": float(np.percentile(lengths, 95)),
        "class_counts": {target_names[int(k)]: int(v) for k, v in zip(unique_labels, counts)},
    }


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        max_features=120000,
        dtype=np.float32,
    )


def build_nb_baseline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("model", MultinomialNB(alpha=0.08)),
        ]
    )


def build_calibrated_svm() -> Pipeline:
    svm = LinearSVC(C=2.0, class_weight=None, random_state=RANDOM_STATE)
    calibrated = CalibratedClassifierCV(svm, method="sigmoid", cv=3)
    return Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("model", calibrated),
        ]
    )


def confidence_policy(probabilities: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    return probabilities.max(axis=1) >= threshold


def evaluate(
    model: Pipeline,
    texts: list[str],
    truth: np.ndarray,
    threshold: float = 0.55,
) -> tuple[Metrics, dict[str, Any], np.ndarray, np.ndarray]:
    prediction = model.predict(texts)
    probabilities = model.predict_proba(texts) if hasattr(model, "predict_proba") else None
    accepted = None
    auto_rate = 1.0
    auto_accuracy = float(accuracy_score(truth, prediction))
    ll = None
    if probabilities is not None:
        accepted = confidence_policy(probabilities, threshold)
        auto_rate = float(accepted.mean())
        auto_accuracy = float(accuracy_score(truth[accepted], prediction[accepted])) if accepted.any() else None
        ll = float(log_loss(truth, probabilities, labels=np.arange(probabilities.shape[1])))
    metrics = Metrics(
        accuracy=float(accuracy_score(truth, prediction)),
        macro_f1=float(f1_score(truth, prediction, average="macro")),
        weighted_f1=float(f1_score(truth, prediction, average="weighted")),
        log_loss=ll,
        auto_route_rate=auto_rate,
        auto_route_accuracy=auto_accuracy,
    )
    detail = {
        "confusion_matrix": confusion_matrix(truth, prediction).tolist(),
        "classification_report": classification_report(truth, prediction, output_dict=True, zero_division=0),
    }
    if probabilities is None:
        probabilities = np.zeros((len(texts), 1), dtype=float)
    if accepted is None:
        accepted = np.ones(len(texts), dtype=bool)
    return metrics, detail, prediction, probabilities


def threshold_sweep(probabilities: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.35, 0.91, 0.05):
        accepted = probabilities.max(axis=1) >= threshold
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "auto_route_rate": float(accepted.mean()),
                "auto_route_accuracy": float(accuracy_score(truth[accepted], prediction[accepted])) if accepted.any() else np.nan,
                "review_rate": float((~accepted).mean()),
                "review_rows": int((~accepted).sum()),
            }
        )
    return pd.DataFrame(rows)


def confusion_pairs(confusion: np.ndarray, target_names: list[str], top_n: int = 20) -> pd.DataFrame:
    rows = []
    matrix = confusion.copy().astype(int)
    np.fill_diagonal(matrix, 0)
    for true_index in range(matrix.shape[0]):
        for pred_index in range(matrix.shape[1]):
            count = int(matrix[true_index, pred_index])
            if count:
                rows.append(
                    {
                        "true_category": target_names[true_index],
                        "predicted_category": target_names[pred_index],
                        "count": count,
                    }
                )
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(top_n)


def category_keywords(model: Pipeline, target_names: list[str], top_n: int = 15) -> dict[str, list[str]]:
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    calibrated: CalibratedClassifierCV = model.named_steps["model"]
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefficient_sets: list[np.ndarray] = []
    for calibrated_classifier in calibrated.calibrated_classifiers_:
        estimator = getattr(calibrated_classifier, "estimator", None)
        if estimator is None:
            estimator = getattr(calibrated_classifier, "base_estimator", None)
        if estimator is not None and hasattr(estimator, "coef_"):
            coefficient_sets.append(estimator.coef_)
    if not coefficient_sets:
        return {name: [] for name in target_names}
    coefficients = np.mean(np.stack(coefficient_sets, axis=0), axis=0)
    if coefficients.shape[0] != len(target_names):
        return {name: [] for name in target_names}
    insights: dict[str, list[str]] = {}
    for index, name in enumerate(target_names):
        top_indices = np.argsort(coefficients[index])[-top_n:][::-1]
        insights[name] = feature_names[top_indices].tolist()
    return insights


def length_error_slices(texts: list[str], truth: np.ndarray, prediction: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "words": [len(text.split()) for text in texts],
            "target": truth,
            "prediction": prediction,
            "confidence": probabilities.max(axis=1),
        }
    )
    frame["correct"] = (frame["target"] == frame["prediction"]).astype(int)
    frame["length_band"] = pd.cut(
        frame["words"],
        bins=[-1, 10, 50, 150, 500, np.inf],
        labels=["very_short", "short", "medium", "long", "very_long"],
    )
    return (
        frame.groupby("length_band", observed=True)
        .agg(rows=("target", "size"), accuracy=("correct", "mean"), mean_confidence=("confidence", "mean"))
        .reset_index()
    )


def predict_document(model: Pipeline, text: str, target_names: list[str], threshold: float = 0.55) -> dict[str, Any]:
    cleaned = normalise_text(text)
    probabilities = model.predict_proba([cleaned])[0]
    label_index = int(np.argmax(probabilities))
    confidence = float(probabilities[label_index])
    top_indices = np.argsort(probabilities)[-3:][::-1]
    return {
        "predicted_category": target_names[label_index],
        "confidence": confidence,
        "decision": "auto_route" if confidence >= threshold else "human_review",
        "top_categories": [
            {"category": target_names[int(i)], "probability": float(probabilities[int(i)])}
            for i in top_indices
        ],
        "cleaned_preview": cleaned[:300],
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    x_train, y_train, x_test, y_test, target_names = load_corpus()
    audit_train = corpus_audit(x_train, y_train, target_names)
    audit_test = corpus_audit(x_test, y_test, target_names)

    baseline = build_nb_baseline()
    baseline.fit(x_train, y_train)
    baseline_metrics, _, _, _ = evaluate(baseline, x_test, y_test, threshold=0.55)

    model = build_calibrated_svm()
    model.fit(x_train, y_train)
    model_metrics, detail, prediction, probabilities = evaluate(model, x_test, y_test, threshold=0.55)

    threshold_table = threshold_sweep(probabilities, y_test, prediction)
    confusion = np.asarray(detail["confusion_matrix"], dtype=int)
    confusion_table = confusion_pairs(confusion, target_names)
    keywords = category_keywords(model, target_names)
    length_slices = length_error_slices(x_test, y_test, prediction, probabilities)

    threshold_table.to_csv(RESULTS / "confidence_thresholds.csv", index=False)
    confusion_table.to_csv(RESULTS / "top_confusions.csv", index=False)
    length_slices.to_csv(RESULTS / "length_error_slices.csv", index=False)
    save_json(RESULTS / "category_keywords.json", keywords)
    joblib.dump({"model": model, "target_names": target_names}, ARTIFACTS / "nlp_document_router.joblib")

    bundle = joblib.load(ARTIFACTS / "nlp_document_router.joblib")
    parity_original = model.predict(x_test[:20])
    parity_loaded = bundle["model"].predict(x_test[:20])
    if not np.array_equal(parity_original, parity_loaded):
        raise RuntimeError("Saved NLP pipeline parity failed")

    example_text = "The graphics driver crashes when I render a 3D scene and the display becomes corrupted."
    payload = {
        "train_audit": audit_train,
        "test_audit": audit_test,
        "naive_bayes_baseline": asdict(baseline_metrics),
        "calibrated_linear_svm": asdict(model_metrics),
        "macro_f1_gain": float(model_metrics.macro_f1 - baseline_metrics.macro_f1),
        "example_prediction": predict_document(model, example_text, target_names),
        "top_confusions": confusion_table.to_dict(orient="records"),
        "limitations": [
            "Historical public benchmark rather than a current enterprise document stream.",
            "Category taxonomy is fixed and does not represent every real routing problem.",
            "Production use requires privacy controls, domain labels, drift monitoring and human escalation for novel text.",
        ],
    }
    save_json(RESULTS / "metrics.json", payload)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
