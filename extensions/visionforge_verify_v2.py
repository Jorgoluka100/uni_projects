"""Independent post-run verifier for VisionForge.

Run this *after* the VisionForge notebook has produced ``visionforge_artifacts``.
The verifier reloads the saved checkpoint in a fresh process, downloads the public
Makerere Beans test split, recomputes test metrics, adds bootstrap confidence
intervals, checks selective prediction, and validates TorchScript/ONNX parity.

It intentionally contains no target performance threshold. Its job is to verify
that retained claims are reproducible, not to manufacture a pass by tuning to test.

Example (Colab/local environment with internet):
    pip install torch torchvision datasets scikit-learn onnxruntime numpy
    python extensions/visionforge_verify_v2.py --artifact-dir visionforge_artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        include = (confidence > left) & (confidence <= right)
        if not include.any():
            continue
        accuracy = float((prediction[include] == labels[include]).mean())
        mean_confidence = float(confidence[include].mean())
        ece += float(include.mean()) * abs(accuracy - mean_confidence)
    return float(ece)


def bootstrap_metric(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    rounds: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    values = np.empty(rounds, dtype=float)
    for index in range(rounds):
        sample = rng.integers(0, n, size=n)
        values[index] = metric(labels[sample], predictions[sample])
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "estimate": float(metric(labels, predictions)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "rounds": int(rounds),
    }


def selective_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accepted = confidence >= threshold
    total_errors = max(int((predictions != labels).sum()), 1)
    return {
        "threshold": float(threshold),
        "coverage": float(accepted.mean()),
        "review_rate": float(1.0 - accepted.mean()),
        "selective_accuracy": float((predictions[accepted] == labels[accepted]).mean()) if accepted.any() else float("nan"),
        "errors_escalated": float((~accepted & (predictions != labels)).sum() / total_errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Imports are deliberately inside main so repository CI can compile this file
    # without installing the heavy computer-vision runtime.
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from PIL import Image
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    artifact_dir = args.artifact_dir.resolve()
    checkpoint_path = artifact_dir / "visionforge_state_dict.pt"
    model_card_path = artifact_dir / "model_card.json"
    torchscript_path = artifact_dir / "visionforge_efficientnet.ts"
    onnx_path = artifact_dir / "visionforge_efficientnet.onnx"
    for path in (checkpoint_path, model_card_path, torchscript_path, onnx_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = [str(value) for value in checkpoint["class_names"]]
    config = dict(checkpoint.get("config", {}))
    image_size = int(config.get("image_size", 224))
    dataset_id = str(config.get("dataset_id", "AI-Lab-Makerere/beans"))
    temperature = float(checkpoint["temperature"])
    threshold = float(checkpoint["confidence_threshold"])
    if len(class_names) != 3:
        raise AssertionError(f"expected 3 classes, got {class_names}")
    if not (0.05 <= temperature <= 20.0):
        raise AssertionError("invalid saved calibration temperature")
    if not (0.0 <= threshold <= 1.0):
        raise AssertionError("invalid saved confidence threshold")

    def build_model(num_classes: int) -> nn.Module:
        network = efficientnet_b0(weights=None)
        in_features = network.classifier[1].in_features
        network.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes),
        )
        return network

    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    dataset = load_dataset(dataset_id)
    if "test" not in dataset:
        raise KeyError(f"{dataset_id} does not expose a test split")
    test_raw = dataset["test"]
    feature_names = set(test_raw.features)
    image_column = "image" if "image" in feature_names else next(
        (name for name, feature in test_raw.features.items() if feature.__class__.__name__.lower() == "image"),
        None,
    )
    label_column = "labels" if "labels" in feature_names else ("label" if "label" in feature_names else None)
    if image_column is None or label_column is None:
        raise KeyError(f"could not resolve image/label columns from {sorted(feature_names)}")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize(int(image_size * 1.12)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    class Adapter(Dataset):
        def __len__(self) -> int:
            return len(test_raw)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            row = test_raw[int(index)]
            image: Image.Image = row[image_column].convert("RGB")
            return transform(image), int(row[label_column])

    loader = DataLoader(
        Adapter(),
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits_parts.append(model(images.to(device)).cpu().numpy())
            label_parts.append(labels.numpy())
    logits = np.concatenate(logits_parts)
    labels = np.concatenate(label_parts)
    probabilities = torch.softmax(torch.from_numpy(logits) / temperature, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)

    accuracy = float(accuracy_score(labels, predictions))
    balanced = float(balanced_accuracy_score(labels, predictions))
    macro_f1 = float(f1_score(labels, predictions, average="macro"))
    nll = float(log_loss(labels, probabilities, labels=list(range(len(class_names)))))
    ece = expected_calibration_error(probabilities, labels)

    acc_ci = bootstrap_metric(
        labels,
        predictions,
        lambda y, p: float(accuracy_score(y, p)),
        rounds=args.bootstrap_rounds,
        seed=args.seed,
    )
    f1_ci = bootstrap_metric(
        labels,
        predictions,
        lambda y, p: float(f1_score(y, p, average="macro", zero_division=0)),
        rounds=args.bootstrap_rounds,
        seed=args.seed + 1,
    )
    policy = selective_metrics(probabilities, labels, threshold)

    # Fresh-process TorchScript and ONNX parity. This catches export artefacts that
    # exist on disk but no longer reproduce the PyTorch model numerically.
    sample = next(iter(loader))[0][: min(2, len(labels))].cpu()
    with torch.inference_mode():
        pytorch_logits = model(sample.to(device)).cpu().numpy()
    scripted = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
    with torch.inference_mode():
        script_logits = scripted(sample).cpu().numpy()
    torchscript_max_abs = float(np.max(np.abs(pytorch_logits - script_logits)))
    torchscript_ok = bool(np.allclose(pytorch_logits, script_logits, atol=1e-4, rtol=1e-3))

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        onnx_logits = session.run(None, {input_name: sample.numpy().astype(np.float32)})[0]
        onnx_max_abs = float(np.max(np.abs(pytorch_logits - onnx_logits)))
        onnx_ok = bool(np.allclose(pytorch_logits, onnx_logits, atol=2e-4, rtol=2e-3))
    except Exception as exc:
        onnx_ok = False
        onnx_max_abs = float("nan")
        onnx_error = f"{type(exc).__name__}: {exc}"
    else:
        onnx_error = None

    model_card = json.loads(model_card_path.read_text(encoding="utf-8"))
    retained = model_card.get("metrics", {})
    claim_checks: dict[str, Any] = {}
    for key, reproduced in {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "negative_log_likelihood": nll,
    }.items():
        if key in retained:
            delta = abs(float(retained[key]) - reproduced)
            claim_checks[key] = {"retained": float(retained[key]), "reproduced": reproduced, "abs_delta": delta, "match": delta <= 1e-6}

    report = {
        "project": "VisionForge",
        "dataset": dataset_id,
        "test_rows": int(len(labels)),
        "class_names": class_names,
        "checkpoint_sha256": sha256(checkpoint_path),
        "device": str(device),
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "macro_f1": macro_f1,
            "negative_log_likelihood": nll,
            "ece_after": ece,
            "accuracy_ci95": acc_ci,
            "macro_f1_ci95": f1_ci,
        },
        "selective_policy": policy,
        "retained_claim_reproduction": claim_checks,
        "export_parity": {
            "torchscript_pass": torchscript_ok,
            "torchscript_max_abs_error": torchscript_max_abs,
            "onnx_pass": onnx_ok,
            "onnx_max_abs_error": onnx_max_abs,
            "onnx_error": onnx_error,
        },
    }
    report["verification_pass"] = bool(
        torchscript_ok
        and onnx_ok
        and all(item.get("match", False) for item in claim_checks.values())
        and math.isfinite(accuracy)
        and math.isfinite(macro_f1)
    )

    output = artifact_dir / "verification_metrics.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"verification report: {output}")
    return 0 if report["verification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
