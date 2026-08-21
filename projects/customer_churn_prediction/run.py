from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation import operating_point, select_cost_threshold

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results" / "verified_metrics.json"


def self_test() -> None:
    y = np.array([0, 0, 0, 1, 1])
    probability = np.array([0.05, 0.10, 0.30, 0.70, 0.90])
    metrics = operating_point(y, probability, 0.50)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["alert_rate"] == 0.4
    selected = select_cost_threshold(y, probability)
    assert 0.01 <= selected["threshold"] <= 0.99
    assert selected["cost_units"] >= 0
    print("Customer churn self-test passed.")


def check_evidence() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert report["verification_pass"] is True
    assert report["source_rows"] == 3150
    assert report["training_rows"] == 2522
    assert report["holdout_rows"] == 628
    assert report["training_profiles"] == 2229
    assert report["holdout_profiles"] == 565
    assert report["selected_algorithm"] == "Histogram gradient boosting"
    assert report["training_selected_threshold"] == 0.15
    assert report["holdout_metrics"]["average_precision"] == 0.955
    assert report["holdout_metrics"]["recall"] == 0.949
    assert report["holdout_metrics"]["precision"] == 0.7323
    assert report["holdout_metrics"]["alert_rate"] == 0.2022
    assert report["excluded_fields"]["status"].startswith("semantic proxy-risk")
    print("Retained churn evidence passed.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-evidence", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.check_evidence:
        check_evidence()
    if not args.self_test and not args.check_evidence:
        parser.error("choose --self-test or --check-evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
