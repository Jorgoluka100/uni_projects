"""Calibration, uncertainty and selective-prediction evaluation helpers."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


def _validate_probabilities_labels(probabilities, labels) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty [n, classes] matrix with at least two classes")
    if y.ndim != 1 or y.size != probs.shape[0]:
        raise ValueError("labels must be a 1D array aligned with probability rows")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("each probability row must sum to one")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.isfinite(y.astype(float)).all() or not np.equal(y.astype(float), np.floor(y.astype(float))).all():
            raise ValueError("labels must be integer class indices")
    y = y.astype(int)
    if np.any((y < 0) | (y >= probs.shape[1])):
        raise ValueError("labels must be valid probability-column indices")
    return probs, y


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable temperature-scaled softmax."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("logits must be a non-empty [n, classes] matrix")
    if not np.isfinite(values).all():
        raise ValueError("logits must contain only finite values")
    values = values / float(temperature)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> float:
    """Weighted confidence-vs-accuracy gap over equal-width confidence bins."""
    probs, y = _validate_probabilities_labels(probabilities, labels)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        include = (confidence > left) & (confidence <= right)
        if not include.any():
            continue
        accuracy = float((prediction[include] == y[include]).mean())
        mean_confidence = float(confidence[include].mean())
        ece += float(include.mean()) * abs(accuracy - mean_confidence)
    return float(ece)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probs, y = _validate_probabilities_labels(probabilities, labels)
    predictions = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "negative_log_likelihood": float(
            log_loss(y, probs, labels=list(range(probs.shape[1])))
        ),
        "expected_calibration_error": expected_calibration_error(probs, y),
    }


def selective_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate a human-review policy based on calibrated maximum confidence."""
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    probs, y = _validate_probabilities_labels(probabilities, labels)
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accepted = confidence >= threshold
    errors = predictions != y
    total_errors = int(errors.sum())
    return {
        "threshold": float(threshold),
        "coverage": float(accepted.mean()),
        "review_rate": float((~accepted).mean()),
        "selective_accuracy": (
            float((predictions[accepted] == y[accepted]).mean())
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
    y = np.asarray(labels)
    pred = np.asarray(predictions)
    if y.ndim != 1 or pred.ndim == 0 or len(y) != len(pred) or not len(y):
        raise ValueError("labels and predictions must be non-empty and aligned")
    rng = np.random.default_rng(seed)
    values = np.empty(rounds, dtype=float)
    n = len(y)
    for index in range(rounds):
        sample = rng.integers(0, n, size=n)
        values[index] = metric(y[sample], pred[sample])
    if not np.isfinite(values).all():
        raise ValueError("bootstrap metric returned non-finite values; check metric and sample design")
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "estimate": float(metric(y, pred)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "rounds": int(rounds),
    }
