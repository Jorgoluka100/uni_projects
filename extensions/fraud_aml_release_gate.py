"""Evidence gate for the synthetic Fraud/AML methodology demonstration."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def finite_01(value: object) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and 0.0 <= x <= 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir
    required = [
        "metrics.json", "fraud_screening_pipeline.joblib", "decision_policy.joblib",
        "monthly_slices.csv", "test_audit.csv",
    ]
    errors = [f"missing {name}" for name in required if not (root / name).is_file()]
    if errors:
        print("\n".join(errors))
        return 1

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    model = payload.get("model_test", {})
    baseline = payload.get("baseline_test", {})
    threshold = payload.get("validation_threshold_choice", {})
    rows = payload.get("rows", {})
    limitations = payload.get("limitations", [])

    if payload.get("scope") != "synthetic methodology demonstration only":
        errors.append("synthetic scope label missing")
    for split in ("train", "valid", "test"):
        if int(rows.get(split, 0)) <= 0:
            errors.append(f"empty {split} split")
    for metric in ("pr_auc", "roc_auc", "precision", "recall", "flag_rate", "prevalence"):
        if not finite_01(model.get(metric)):
            errors.append(f"invalid model_test {metric}")
    if not finite_01(baseline.get("prevalence")):
        errors.append("invalid baseline prevalence")
    if not finite_01(threshold.get("threshold")):
        errors.append("invalid validation-selected threshold")
    if len(limitations) < 3:
        errors.append("limitations incomplete")

    slices = pd.read_csv(root / "monthly_slices.csv")
    audit = pd.read_csv(root / "test_audit.csv")
    if slices.empty or audit.empty:
        errors.append("retained audit evidence is empty")
    if len(audit) != int(rows.get("test", -1)):
        errors.append("test audit row count mismatch")
    if set(audit.get("flagged_for_review", pd.Series(dtype=int)).dropna().unique()) - {0, 1}:
        errors.append("invalid review flag values")

    report = {
        "project": "Fraud/AML v2",
        "verification_pass": not errors,
        "errors": errors,
        "scope": payload.get("scope"),
        "model_test": model,
        "baseline_test": baseline,
        "validation_threshold_choice": threshold,
        "rows": rows,
    }
    (root / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
