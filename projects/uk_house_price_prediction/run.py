from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import AreaPropertyBaseline, conformal_radius, interval_coverage, regression_metrics

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results" / "verified_metrics.json"


def self_test() -> None:
    train = pd.DataFrame(
        {
            "postcode_district": ["E1", "E1", "SW1", "SW1"],
            "property_type": ["F", "F", "T", "T"],
            "price": [300_000, 320_000, 700_000, 760_000],
        }
    )
    test = pd.DataFrame(
        {
            "postcode_district": ["E1", "SW1", "N1"],
            "property_type": ["F", "T", "F"],
            "price": [310_000, 730_000, 330_000],
        }
    )
    baseline = AreaPropertyBaseline().fit(train)
    pred = baseline.predict(test)
    assert np.allclose(pred, [310_000, 730_000, 310_000])
    metrics = regression_metrics(test.price, pred)
    assert metrics["mae"] >= 0
    radius = conformal_radius([100, 200, 300], [90, 190, 250], coverage=0.90)
    assert radius == 50
    covered = interval_coverage([100, 200], [100, 220], radius=30, floor=0, cap=1000)
    assert covered["coverage"] == 1.0
    print("UK house price self-test passed.")


def check_evidence() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert report["verification_pass"] is True
    assert report["model_population"] == 995_059
    assert report["untouched_2026_test_rows"] == 216_564
    model = report["test_metrics"]["catboost"]
    baseline = report["test_metrics"]["area_property_baseline"]
    assert model["mae"] < baseline["mae"]
    assert 0 <= report["test_interval_coverage"] <= 1
    assert report["model_reload_delta"] == 0.0
    print("Retained house-price evidence passed.")


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
