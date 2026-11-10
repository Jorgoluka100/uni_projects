from pathlib import Path
import ast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "run.py"


def load_module_tree():
    return ast.parse(RUN.read_text(encoding="utf-8"))


def test_training_source_compiles():
    compile(RUN.read_text(encoding="utf-8"), str(RUN), "exec")


def test_cnn_architecture_is_present():
    tree = load_module_tree()
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "CompactCNN" in class_names
    assert "RegularizedCNN" in class_names


def test_core_deep_learning_components_are_visible():
    source = RUN.read_text(encoding="utf-8")
    for token in [
        "nn.Conv2d",
        "nn.BatchNorm2d",
        "nn.Dropout",
        "AdamW",
        "CosineAnnealingLR",
        "ResNet18_Weights.DEFAULT",
        "temperature_scale",
        "selective_prediction_table",
        "confusion_matrix",
    ]:
        assert token in source


def test_compact_cnn_forward_shape():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cnn_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.CompactCNN(num_classes=10)
    batch = torch.randn(4, 3, 32, 32)
    output = model(batch)
    assert output.shape == (4, 10)


def test_regularized_cnn_forward_shape():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cnn_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.RegularizedCNN(num_classes=10)
    batch = torch.randn(2, 3, 32, 32)
    output = model(batch)
    assert output.shape == (2, 10)

def _load_cnn_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cnn_validation_tests", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_and_selective_metrics_reject_invalid_inputs():
    import pytest

    module = _load_cnn_module()
    probabilities = np.full((2, 10), 0.1, dtype=float)
    targets = np.array([0, 1])
    with pytest.raises(ValueError, match="sum to one"):
        module.expected_calibration_error(probabilities * 2, targets)
    with pytest.raises(ValueError, match="bins"):
        module.expected_calibration_error(probabilities, targets, bins=1)
    with pytest.raises(ValueError, match="aligned"):
        module.selective_prediction_table(probabilities, np.array([0]))


def test_temperature_scaling_rejects_invalid_logits_and_temperature():
    import pytest

    module = _load_cnn_module()
    logits = np.zeros((4, 10), dtype=float)
    targets = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="finite"):
        module.temperature_scale(np.full((4, 10), np.nan), targets)
    with pytest.raises(ValueError, match="positive"):
        module.apply_temperature(logits, 0.0)
