from pathlib import Path
import ast

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
