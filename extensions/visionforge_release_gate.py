"""VisionForge release-evidence gate.

This script does not train the model or invent performance claims. It validates the
artifact bundle produced by 12_VisionForge_PyTorch_Visual_Inspection.ipynb before
results are promoted into the verified portfolio tier.

Usage:
    python extensions/visionforge_release_gate.py --artifact-dir visionforge_artifacts
    python extensions/visionforge_release_gate.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

REQUIRED_FILES = {
    "manifest.json",
    "model_card.json",
    "model_benchmark.csv",
    "class_metrics.csv",
    "corruption_report.csv",
    "selective_policy.csv",
    "visionforge_state_dict.pt",
    "visionforge_scratch_cnn_state_dict.pt",
    "visionforge_efficientnet.ts",
    "visionforge_efficientnet.onnx",
}

REQUIRED_HEADLINE = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "ece_before",
    "ece_after",
    "temperature",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_01(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def validate(bundle: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []

    missing = sorted(name for name in REQUIRED_FILES if not (bundle / name).is_file())
    if missing:
        errors.append("missing required artifacts: " + ", ".join(missing))
        return errors, warnings, {}

    manifest = load_json(bundle / "manifest.json")
    card = load_json(bundle / "model_card.json")
    metrics = manifest.get("metrics", {})
    if not isinstance(metrics, dict):
        errors.append("manifest.metrics must be an object")
        metrics = {}

    absent_metrics = sorted(REQUIRED_HEADLINE - set(metrics))
    if absent_metrics:
        errors.append("manifest missing headline metrics: " + ", ".join(absent_metrics))

    for name in ("accuracy", "balanced_accuracy", "macro_f1", "ece_before", "ece_after"):
        if name in metrics and not finite_01(metrics[name]):
            errors.append(f"{name} must be finite and in [0, 1]")

    try:
        temperature = float(metrics.get("temperature"))
        if not (math.isfinite(temperature) and 0.05 <= temperature <= 20.0):
            errors.append("temperature must be finite and in [0.05, 20]")
    except (TypeError, ValueError):
        errors.append("temperature must be numeric")

    if finite_01(metrics.get("ece_before")) and finite_01(metrics.get("ece_after")):
        if float(metrics["ece_after"]) > float(metrics["ece_before"]) + 1e-12:
            warnings.append("temperature scaling increased ECE; inspect calibration before promotion")

    # Verify every artifact recorded by the notebook manifest.
    manifest_files = manifest.get("artifacts", [])
    if not isinstance(manifest_files, list) or not manifest_files:
        errors.append("manifest.artifacts is empty")
    else:
        for item in manifest_files:
            if not isinstance(item, dict):
                errors.append("manifest artifact entry must be an object")
                continue
            name = Path(str(item.get("path", ""))).name
            path = bundle / name
            if not path.is_file():
                errors.append(f"manifest references missing artifact: {name}")
                continue
            expected = str(item.get("sha256", ""))
            if expected and sha256(path) != expected:
                errors.append(f"SHA-256 mismatch: {name}")

    benchmark = rows(bundle / "model_benchmark.csv")
    models = {row.get("model", "") for row in benchmark}
    if models != {"ScratchResidualCNN", "EfficientNet-B0"}:
        errors.append("benchmark must contain ScratchResidualCNN and EfficientNet-B0")
    for row in benchmark:
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            if not finite_01(row.get(metric)):
                errors.append(f"invalid benchmark {metric} for {row.get('model', 'unknown')}")

    corruption = rows(bundle / "corruption_report.csv")
    required_corruptions = {"gaussian_noise", "blur", "darkness", "occlusion"}
    if not required_corruptions.issubset({row.get("corruption", "") for row in corruption}):
        errors.append("corruption report must cover noise, blur, darkness and occlusion")

    policy = rows(bundle / "selective_policy.csv")
    required_policy_columns = {"threshold", "coverage", "review_rate", "selective_accuracy", "errors_escalated"}
    if not policy or not required_policy_columns.issubset(policy[0]):
        errors.append("selective_policy.csv is missing required decision-policy columns")

    class_metrics = rows(bundle / "class_metrics.csv")
    if len(class_metrics) < 3:
        errors.append("class_metrics.csv must retain per-class evidence for all three classes")

    intended = str(card.get("intended_use", "")).strip()
    limitations = card.get("known_limitations", [])
    oversight = str(card.get("human_oversight", "")).strip()
    if not intended:
        errors.append("model card missing intended_use")
    if not isinstance(limitations, list) or len(limitations) < 3:
        errors.append("model card must document at least three known limitations")
    if not oversight:
        errors.append("model card missing human_oversight")

    report = {
        "project": "VisionForge",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "headline_metrics": metrics,
        "artifact_count": len(manifest_files) if isinstance(manifest_files, list) else 0,
    }
    return errors, warnings, report


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Minimal valid fixtures. Binary model files are placeholders because this gate
        # validates evidence packaging, while notebook runtime tests validate inference.
        for name in REQUIRED_FILES - {"manifest.json", "model_card.json", "model_benchmark.csv", "class_metrics.csv", "corruption_report.csv", "selective_policy.csv"}:
            (root / name).write_bytes((name + "\n").encode())
        (root / "model_benchmark.csv").write_text(
            "model,accuracy,balanced_accuracy,macro_f1,negative_log_likelihood\n"
            "ScratchResidualCNN,0.60,0.59,0.58,1.0\nEfficientNet-B0,0.80,0.79,0.78,0.5\n",
            encoding="utf-8",
        )
        (root / "class_metrics.csv").write_text("class,precision,recall\na,0.8,0.8\nb,0.8,0.8\nc,0.8,0.8\n", encoding="utf-8")
        (root / "corruption_report.csv").write_text(
            "corruption,severity,accuracy,macro_f1,threshold,coverage,review_rate,selective_accuracy,errors_escalated\n"
            "gaussian_noise,0.1,0.7,0.7,0.7,0.8,0.2,0.8,0.3\nblur,0.3,0.7,0.7,0.7,0.8,0.2,0.8,0.3\n"
            "darkness,0.3,0.7,0.7,0.7,0.8,0.2,0.8,0.3\nocclusion,0.2,0.7,0.7,0.7,0.8,0.2,0.8,0.3\n",
            encoding="utf-8",
        )
        (root / "selective_policy.csv").write_text(
            "threshold,coverage,review_rate,selective_accuracy,errors_escalated\n0.7,0.8,0.2,0.85,0.4\n",
            encoding="utf-8",
        )
        card = {
            "intended_use": "Educational field-image triage prototype",
            "known_limitations": ["small dataset", "domain shift", "calibration drift"],
            "human_oversight": "Low-confidence cases require agronomist review",
        }
        (root / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
        tracked = []
        for path in root.iterdir():
            if path.name not in {"manifest.json"}:
                tracked.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
        manifest = {
            "project": "VisionForge",
            "metrics": {
                "accuracy": 0.80,
                "balanced_accuracy": 0.79,
                "macro_f1": 0.78,
                "negative_log_likelihood": 0.50,
                "ece_before": 0.12,
                "ece_after": 0.08,
                "temperature": 1.2,
            },
            "artifacts": tracked,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        errors, _, _ = validate(root)
        if errors:
            raise AssertionError(errors)
    print("VisionForge release-gate self-test passed.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.artifact_dir is None:
        parser.error("--artifact-dir is required unless --self-test is used")
    errors, warnings, report = validate(args.artifact_dir)
    output = args.artifact_dir / "visionforge_release_report.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"release report: {output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
