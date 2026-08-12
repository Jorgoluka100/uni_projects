"""VisionForge release-evidence gate.

This script does not train the model or invent performance claims. It validates the
artifact bundle produced by 12_VisionForge_PyTorch_Visual_Inspection.ipynb plus the
fresh-process report produced by ``visionforge_verify_v2.py`` before results are
promoted into the verified portfolio tier.

Usage:
    python extensions/visionforge_verify_v2.py --artifact-dir visionforge_artifacts
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
    "verification_metrics.json",
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
    verification = load_json(bundle / "verification_metrics.json")
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

    # Verify every artifact recorded by the notebook manifest. The independent
    # verification report is intentionally produced after the notebook manifest and
    # therefore is checked separately below.
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

    # Independent fresh-process verification gate.
    if verification.get("verification_pass") is not True:
        errors.append("independent verifier did not report verification_pass=true")
    export_parity = verification.get("export_parity", {})
    if not isinstance(export_parity, dict):
        errors.append("verification export_parity must be an object")
    else:
        if export_parity.get("torchscript_pass") is not True:
            errors.append("TorchScript parity did not pass independent verification")
        if export_parity.get("onnx_pass") is not True:
            errors.append("ONNX parity did not pass independent verification")

    reproduced = verification.get("retained_claim_reproduction", {})
    if not isinstance(reproduced, dict) or not reproduced:
        errors.append("independent verification has no retained-claim reproduction checks")
    else:
        required_claims = {"accuracy", "balanced_accuracy", "macro_f1", "negative_log_likelihood"}
        missing_claims = sorted(required_claims - set(reproduced))
        if missing_claims:
            errors.append("independent verification missing claim checks: " + ", ".join(missing_claims))
        for name, item in reproduced.items():
            if not isinstance(item, dict) or item.get("match") is not True:
                errors.append(f"retained claim did not reproduce: {name}")

    verification_metrics = verification.get("metrics", {})
    for ci_name in ("accuracy_ci95", "macro_f1_ci95"):
        interval = verification_metrics.get(ci_name, {}) if isinstance(verification_metrics, dict) else {}
        if not isinstance(interval, dict):
            errors.append(f"{ci_name} missing from independent verification")
            continue
        try:
            estimate = float(interval["estimate"])
            low = float(interval["ci95_low"])
            high = float(interval["ci95_high"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{ci_name} is incomplete")
            continue
        if not (0.0 <= low <= estimate <= high <= 1.0):
            errors.append(f"{ci_name} has invalid bounds")

    report = {
        "project": "VisionForge",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "headline_metrics": metrics,
        "independent_verification": {
            "verification_pass": verification.get("verification_pass"),
            "export_parity": export_parity,
            "claim_checks": reproduced,
        },
        "artifact_count": len(manifest_files) if isinstance(manifest_files, list) else 0,
    }
    return errors, warnings, report


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        binary_files = REQUIRED_FILES - {
            "manifest.json",
            "model_card.json",
            "model_benchmark.csv",
            "class_metrics.csv",
            "corruption_report.csv",
            "selective_policy.csv",
            "verification_metrics.json",
        }
        for name in binary_files:
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
        verification = {
            "verification_pass": True,
            "metrics": {
                "accuracy_ci95": {"estimate": 0.80, "ci95_low": 0.72, "ci95_high": 0.87, "rounds": 2000},
                "macro_f1_ci95": {"estimate": 0.78, "ci95_low": 0.70, "ci95_high": 0.85, "rounds": 2000},
            },
            "retained_claim_reproduction": {
                "accuracy": {"retained": 0.80, "reproduced": 0.80, "abs_delta": 0.0, "match": True},
                "balanced_accuracy": {"retained": 0.79, "reproduced": 0.79, "abs_delta": 0.0, "match": True},
                "macro_f1": {"retained": 0.78, "reproduced": 0.78, "abs_delta": 0.0, "match": True},
                "negative_log_likelihood": {"retained": 0.50, "reproduced": 0.50, "abs_delta": 0.0, "match": True},
            },
            "export_parity": {
                "torchscript_pass": True,
                "torchscript_max_abs_error": 1e-6,
                "onnx_pass": True,
                "onnx_max_abs_error": 1e-6,
                "onnx_error": None,
            },
        }
        (root / "verification_metrics.json").write_text(json.dumps(verification), encoding="utf-8")
        tracked = []
        for path in root.iterdir():
            if path.name not in {"manifest.json", "verification_metrics.json"}:
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
