"""Fast verification entry point for the image-classification confidence project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from src.evaluation import (
    bootstrap_metric,
    classification_metrics,
    expected_calibration_error,
    selective_metrics,
    softmax,
)
from src.parity import parity_report


def self_test() -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    logits = np.asarray(
        [
            [4.0, 0.2, 0.1],
            [0.1, 3.0, 0.3],
            [0.2, 0.5, 2.5],
            [2.0, 1.8, 0.2],
            [1.5, 1.6, 0.3],
            [0.2, 2.0, 1.9],
        ],
        dtype=float,
    )
    probabilities = softmax(logits, temperature=1.2)
    metrics = classification_metrics(labels, probabilities)
    policy = selective_metrics(probabilities, labels, threshold=0.60)
    predictions = probabilities.argmax(axis=1)
    interval = bootstrap_metric(
        labels,
        predictions,
        lambda y, p: float(accuracy_score(y, p)),
        rounds=200,
        seed=42,
    )
    ece = expected_calibration_error(probabilities, labels, bins=5)
    parity = parity_report(logits, logits + 1e-7, atol=1e-5, rtol=1e-5)

    assert probabilities.shape == logits.shape
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= policy["coverage"] <= 1.0
    assert 0.0 <= policy["review_rate"] <= 1.0
    assert abs(policy["coverage"] + policy["review_rate"] - 1.0) < 1e-12
    assert 0.0 <= ece <= 1.0
    assert interval["ci95_low"] <= interval["estimate"] <= interval["ci95_high"]
    assert parity["pass"] is True
    print("Image-classification confidence self-test passed.")


def verify_retained(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("verification_pass") is not True:
        raise AssertionError("retained evidence does not report verification_pass=true")
    metrics = payload.get("metrics", {})
    policy = payload.get("selective_policy", {})
    exports = payload.get("export_parity", {})
    required_metrics = {"accuracy", "balanced_accuracy", "macro_f1", "negative_log_likelihood", "ece_after"}
    missing = sorted(required_metrics - set(metrics))
    if missing:
        raise AssertionError(f"retained evidence is missing metrics: {missing}")
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "ece_after"):
        value = float(metrics[name])
        if not 0.0 <= value <= 1.0:
            raise AssertionError(f"{name} is outside [0, 1]")
    for name in ("coverage", "review_rate", "selective_accuracy", "errors_escalated"):
        value = float(policy[name])
        if not 0.0 <= value <= 1.0:
            raise AssertionError(f"selective policy {name} is outside [0, 1]")
    if exports.get("torchscript_pass") is not True or exports.get("onnx_pass") is not True:
        raise AssertionError("one or more retained export parity checks failed")
    reproduced = payload.get("retained_claim_reproduction", {})
    if not reproduced or not all(item.get("match") is True for item in reproduced.values()):
        raise AssertionError("retained headline claims did not reproduce exactly")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify image-classification evaluation utilities and retained evidence")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(__file__).parent / "results" / "verified_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    evidence = verify_retained(args.evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
