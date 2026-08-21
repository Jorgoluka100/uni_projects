from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analytics import build_sessions, engagement_funnel
from src.conversion import threshold_metrics

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results" / "verified_metrics.json"


def self_test() -> None:
    from pyspark.sql import SparkSession, functions as F

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("clickstream-project-self-test")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        events = spark.createDataFrame(
            [
                (1, "A", 1, 10.0, 1, "2026-01-01"),
                (1, "B", 1, 12.0, 2, "2026-01-01"),
                (2, "A", 2, 9.0, 1, "2026-01-02"),
            ],
            "session_id long, product_id string, main_category int, price double, page int, event_date string",
        ).withColumn("event_date", F.to_date("event_date")).withColumn("country", F.lit(1))
        sessions = build_sessions(events)
        funnel = engagement_funnel(sessions)
        assert funnel == {"all_sessions": 2, "2+ clicks": 1, "5+ clicks": 0, "10+ clicks": 0}

        scored = spark.createDataFrame(
            [(0.1, 0), (0.4, 0), (0.8, 1), (0.9, 1)],
            "score double, label int",
        )
        metrics = threshold_metrics(scored, 0.5)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
    finally:
        spark.stop()
    print("PySpark clickstream self-test passed.")


def check_evidence() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert report["verification_pass"] is True
    assert report["clickstream_dataset"]["real_events"] == 165474
    assert report["clickstream_dataset"]["real_sessions"] == 24026
    assert report["load_test"]["replicated"] is True
    assert report["load_test"]["used_for_business_metrics"] is False
    assert report["conversion_dataset"]["sessions"] == 12330
    assert report["conversion_dataset"]["test_rows"] == 1801
    assert report["validation_selected_threshold"] == 0.25
    assert report["conversion_test"]["pr_auc"] > report["conversion_dataset"]["conversion_rate"]
    assert report["pipeline_reload_delta"] == 0.0
    print("Retained PySpark clickstream evidence passed.")


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
