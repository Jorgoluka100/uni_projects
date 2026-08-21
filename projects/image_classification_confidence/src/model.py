"""EfficientNet-B0 architecture used by the retained image classifier."""
from __future__ import annotations


def build_efficientnet_classifier(num_classes: int = 3):
    """Build the exact EfficientNet-B0 classifier head used by the verified checkpoint.

    Heavy PyTorch imports stay inside the function so lightweight CI can test the
    evaluation package without downloading the full computer-vision runtime.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    import torch.nn as nn
    from torchvision.models import efficientnet_b0

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
