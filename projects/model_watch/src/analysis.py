from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SEVERITY_ORDER = {"green": 0, "amber": 1, "red": 2}


@dataclass(frozen=True)
class MonitoringDecision:
    batch: str
    severity: str
    investigate: bool
    retrain_candidate: bool
    human_approval_required: bool
    reasons: tuple[str, ...]


def calibration_table(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(target) != len(probability):
        raise ValueError("Target and probability lengths must match")
    if bins < 2:
        raise ValueError("Use at least two calibration bins")
    frame = pd.DataFrame({"target": target, "probability": probability})
    frame["bin"] = pd.cut(
        frame["probability"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
    )
    grouped = (
        frame.groupby("bin", observed=True)
        .agg(
            rows=("target", "size"),
            observed_rate=("target", "mean"),
            predicted_rate=("probability", "mean"),
        )
        .reset_index()
    )
    grouped["absolute_calibration_gap"] = (
        grouped["observed_rate"] - grouped["predicted_rate"]
    ).abs()
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def score_decile_table(
    frame: pd.DataFrame,
    probability: np.ndarray,
    target_col: str = "target",
    deciles: int = 10,
) -> pd.DataFrame:
    work = frame[[target_col]].copy().reset_index(drop=True)
    work["probability"] = np.asarray(probability, dtype=float)
    work["score_decile"] = pd.qcut(
        work["probability"],
        q=deciles,
        labels=False,
        duplicates="drop",
    )
    grouped = (
        work.groupby("score_decile", observed=True)
        .agg(
            rows=(target_col, "size"),
            prevalence=(target_col, "mean"),
            mean_score=("probability", "mean"),
            min_score=("probability", "min"),
            max_score=("probability", "max"),
        )
        .reset_index()
        .sort_values("score_decile", ascending=False)
    )
    overall_prevalence = float(work[target_col].mean())
    grouped["lift_vs_population"] = grouped["prevalence"] / max(overall_prevalence, 1e-9)
    return grouped


def feature_drift_table(batch_report: dict) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for feature, metrics in batch_report.get("drift", {}).items():
        rows.append(
            {
                "feature": str(feature),
                "psi": float(metrics.get("psi", np.nan)),
                "ks_stat": float(metrics.get("ks_stat", np.nan)),
                "ks_pvalue": float(metrics.get("ks_pvalue", np.nan)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "psi", "ks_stat", "ks_pvalue"])
    return pd.DataFrame(rows).sort_values(["psi", "ks_stat"], ascending=False)


def batch_trend_table(reports: Iterable[dict]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for index, report in enumerate(reports):
        performance = report.get("performance", {})
        rows.append(
            {
                "batch_order": int(index),
                "batch": str(report.get("batch", index)),
                "rows": int(report.get("rows", 0)),
                "alert": str(report.get("alert", "unknown")),
                "max_psi": float(report.get("max_psi", np.nan)),
                "roc_auc": float(performance.get("roc_auc", np.nan)),
                "pr_auc": float(performance.get("pr_auc", np.nan)),
                "brier": float(performance.get("brier", np.nan)),
                "ece": float(performance.get("ece", np.nan)),
                "auc_drop_vs_reference": float(
                    performance.get("auc_drop_vs_reference", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def subgroup_gap_table(report: dict) -> pd.DataFrame:
    subgroups = report.get("subgroups", {})
    rows: list[dict[str, float | str | int]] = []
    for subgroup, metrics in subgroups.items():
        rows.append(
            {
                "subgroup": str(subgroup),
                "rows": int(metrics.get("rows", 0)),
                "prevalence": float(metrics.get("prevalence", np.nan)),
                "brier": float(metrics.get("brier", np.nan)),
                "mean_score": float(metrics.get("mean_score", np.nan)),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["prevalence_gap_vs_overall"] = table["prevalence"] - np.average(
        table["prevalence"],
        weights=np.maximum(table["rows"], 1),
    )
    table["score_gap_vs_overall"] = table["mean_score"] - np.average(
        table["mean_score"],
        weights=np.maximum(table["rows"], 1),
    )
    return table.sort_values("rows", ascending=False)


def classify_metric_change(
    value: float,
    amber_threshold: float,
    red_threshold: float,
) -> str:
    if value >= red_threshold:
        return "red"
    if value >= amber_threshold:
        return "amber"
    return "green"


def worst_severity(levels: Iterable[str]) -> str:
    levels = list(levels)
    if not levels:
        return "green"
    return max(levels, key=lambda level: SEVERITY_ORDER.get(level, -1))


def decision_from_report(report: dict) -> MonitoringDecision:
    batch = str(report.get("batch", "unknown"))
    alert = str(report.get("alert", "green"))
    reasons = tuple(str(reason) for reason in report.get("reasons", []))
    investigate = alert in {"amber", "red"}
    retrain_candidate = alert == "red"
    return MonitoringDecision(
        batch=batch,
        severity=alert,
        investigate=investigate,
        retrain_candidate=retrain_candidate,
        human_approval_required=True,
        reasons=reasons,
    )


def retraining_policy_table(reports: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for report in reports:
        decision = decision_from_report(report)
        rows.append(
            {
                "batch": decision.batch,
                "severity": decision.severity,
                "investigate": decision.investigate,
                "retrain_candidate": decision.retrain_candidate,
                "human_approval_required": decision.human_approval_required,
                "reasons": " | ".join(decision.reasons),
            }
        )
    return pd.DataFrame(rows)


def consecutive_alerts(
    reports: Iterable[dict],
    minimum_severity: str = "amber",
) -> int:
    threshold = SEVERITY_ORDER[minimum_severity]
    count = 0
    for report in reversed(list(reports)):
        level = str(report.get("alert", "green"))
        if SEVERITY_ORDER.get(level, -1) >= threshold:
            count += 1
        else:
            break
    return count


def operational_recommendation(reports: list[dict]) -> dict[str, object]:
    if not reports:
        return {
            "status": "no_data",
            "action": "collect monitoring batches",
            "auto_retrain": False,
        }
    latest = reports[-1]
    severity = str(latest.get("alert", "green"))
    amber_streak = consecutive_alerts(reports, "amber")
    red_streak = consecutive_alerts(reports, "red")
    if severity == "red" and red_streak >= 2:
        action = "open incident, investigate data/concept drift, prepare retraining candidate"
    elif severity == "red":
        action = "open incident and investigate before any retraining decision"
    elif severity == "amber" and amber_streak >= 2:
        action = "increase review cadence and investigate persistent degradation"
    elif severity == "amber":
        action = "review the next batch and inspect leading drift features"
    else:
        action = "continue standard monitoring cadence"
    return {
        "status": severity,
        "latest_batch": str(latest.get("batch", "unknown")),
        "consecutive_amber_or_worse": int(amber_streak),
        "consecutive_red": int(red_streak),
        "action": action,
        "auto_retrain": False,
        "human_approval_required": True,
    }


def monitoring_scorecard(reports: list[dict]) -> dict[str, object]:
    trend = batch_trend_table(reports)
    decisions = retraining_policy_table(reports)
    worst = worst_severity(trend["alert"].tolist()) if not trend.empty else "green"
    return {
        "batches_monitored": int(len(reports)),
        "worst_severity": worst,
        "red_batches": int((trend["alert"] == "red").sum()) if not trend.empty else 0,
        "amber_batches": int((trend["alert"] == "amber").sum()) if not trend.empty else 0,
        "max_observed_psi": float(trend["max_psi"].max()) if not trend.empty else 0.0,
        "max_auc_drop": float(trend["auc_drop_vs_reference"].max()) if not trend.empty else 0.0,
        "max_ece": float(trend["ece"].max()) if not trend.empty else 0.0,
        "operational_recommendation": operational_recommendation(reports),
        "decision_table": decisions.to_dict(orient="records"),
    }
