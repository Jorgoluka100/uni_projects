from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CONTACT_COST_UNITS = 25.0
MISSED_CHURN_COST_UNITS = 200.0


def operating_point(y_true, probability, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(accuracy_score(y, prediction)),
        "alert_rate": float(prediction.mean()),
        "brier_score": float(brier_score_loss(y, p)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def scenario_cost(y_true, probability, threshold: float) -> float:
    metrics = operating_point(y_true, probability, threshold)
    return float(
        CONTACT_COST_UNITS * metrics["false_positives"]
        + MISSED_CHURN_COST_UNITS * metrics["false_negatives"]
    )


def select_cost_threshold(y_true, probability) -> dict[str, float]:
    """Choose an operating threshold from training-only probabilities."""
    p = np.asarray(probability, dtype=float)
    candidates = np.unique(np.r_[0.01, np.round(p, 4), 0.99])
    candidates = candidates[(candidates >= 0.01) & (candidates <= 0.99)]
    rows = []
    for threshold in candidates:
        metrics = operating_point(y_true, p, float(threshold))
        metrics["cost_units"] = scenario_cost(y_true, p, float(threshold))
        rows.append(metrics)
    rows.sort(key=lambda row: (row["cost_units"], -row["recall"], row["threshold"]))
    return rows[0]
