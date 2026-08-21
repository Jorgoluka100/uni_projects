"""Evaluation helpers for ranking, calibration and operational capacity."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    prevalence = float(y_true.mean())
    precision = float(precision_score(y_true, predicted, zero_division=0))
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, scores)),
        "precision": precision,
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "flag_rate": float(predicted.mean()),
        "prevalence": prevalence,
        "precision_lift_vs_prevalence": float(precision / prevalence) if prevalence else float("nan"),
    }


def top_fraction_lift(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = max(1, int(round(len(y_true) * fraction)))
    order = np.argsort(-scores, kind="mergesort")[:n]
    rate = float(y_true[order].mean())
    prevalence = float(y_true.mean())
    return {
        "fraction": float(fraction),
        "rows": int(n),
        "delay_rate": rate,
        "population_prevalence": prevalence,
        "lift": float(rate / prevalence) if prevalence else float("nan"),
    }


def capacity_curve(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Show how much delay risk is concentrated in the highest-scored flights."""
    rows = [top_fraction_lift(y_true, scores, fraction) for fraction in (0.05, 0.10, 0.20, 0.30, 0.50)]
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    """Weighted absolute calibration gap across equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, bins - 1)
    error = 0.0
    for idx in range(bins):
        mask = bucket == idx
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(float(scores[mask].mean()) - float(y_true[mask].mean()))
    return float(error)


def calibration_table(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Create a compact reliability table for inspection in the retained artifacts."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, bins - 1)
    rows: list[dict[str, float | int]] = []
    for idx in range(bins):
        mask = bucket == idx
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": idx + 1,
                "rows": int(mask.sum()),
                "mean_predicted_risk": float(scores[mask].mean()),
                "observed_delay_rate": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)
