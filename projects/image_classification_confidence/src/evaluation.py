"""Calibration, uncertainty and selective-prediction evaluation helpers."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable temperature-scaled softmax."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    values = np.asarray(logits, dtype=float) / float(temperature)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> float:
    """Weighted confidence-vs-accuracy gap over equal-width confidence bins."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("probabilities must be [n, classes] and align with labels")
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        include = (confidence > left) & (confidence <= right)
        if not include.any():
            continue
        accuracy = float((prediction[include] == labels[include]).mean())
        mean_confidence = float(confidence[include].mean())
        ece += float(include.mean()) * abs(accuracy - mean_confidence)
    return float(ece)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "negative_log_likelihood": float(
            log_loss(labels, probabilities, labels=list(range(probabilities.shape[1])))
        ),
        "expected_calibration_error": expected_calibration_error(probabilities, labels),
    }


def selective_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate a human-review policy based on calibrated maximum confidence."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accepted = confidence >= threshold
    errors = predictions != labels
    total_errors = int(errors.sum())
    return {
        "threshold": float(threshold),
        "coverage": float(accepted.mean()),
        "review_rate": float((~accepted).mean()),
        "selective_accuracy": (
            float((predictions[accepted] == labels[accepted]).mean())
            if accepted.any()
            else float("nan")
        ),
        "errors_escalated": (
            float(((~accepted) & errors).sum() / total_errors) if total_errors else 0.0
        ),
    }


def bootstrap_metric(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    rounds: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Non-parametric bootstrap interval for a prediction metric."""
    if rounds < 100:
        raise ValueError("rounds should be at least 100 for a useful interval")
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if len(labels) != len(predictions) or not len(labels):
        raise ValueError("labels and predictions must be non-empty and aligned")
    rng = np.random.default_rng(seed)
    values = np.empty(rounds, dtype=float)
    n = len(labels)
    for index in range(rounds):
        sample = rng.integers(0, n, size=n)
        values[index] = metric(labels[sample], predictions[sample])
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "estimate": float(metric(labels, predictions)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "rounds": int(rounds),
    }
