"""Train and evaluate CNN image classifiers with confidence routing."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


class CompactCNN(nn.Module):
    """Small from-scratch convolutional baseline."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class RegularizedCNN(nn.Module):
    """Deeper CNN with BatchNorm and Dropout."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.10),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """Transfer-learning comparator adapted to CIFAR-sized images."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_datasets():
    train_full = datasets.CIFAR10(
        root=str(ROOT / "data"), train=True, download=True, transform=TRAIN_TRANSFORM
    )
    train_eval = datasets.CIFAR10(
        root=str(ROOT / "data"), train=True, download=False, transform=EVAL_TRANSFORM
    )
    test_set = datasets.CIFAR10(
        root=str(ROOT / "data"), train=False, download=True, transform=EVAL_TRANSFORM
    )
    indices = np.arange(len(train_full))
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    validation_indices = indices[:5000]
    training_indices = indices[5000:]
    train_set = Subset(train_full, training_indices.tolist())
    validation_set = Subset(train_eval, validation_indices.tolist())
    return train_set, validation_set, test_set


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence < right)
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
    return float(ece)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
    return {"loss": running_loss / seen, "accuracy": correct / seen}


@torch.inference_mode()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    logits_parts = []
    target_parts = []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        logits_parts.append(logits.cpu())
        target_parts.append(labels.cpu())
    logits = torch.cat(logits_parts).numpy()
    targets = torch.cat(target_parts).numpy()
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "loss": running_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "logits": logits,
        "targets": targets,
        "probabilities": probabilities,
        "predictions": predictions,
        "ece": expected_calibration_error(probabilities, targets),
    }


def fit_model(model, train_loader, validation_loader, epochs: int, learning_rate: float):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history = []
    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
        validation_metrics = evaluate(model, validation_loader, criterion)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_ece": validation_metrics["ece"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history), model


def temperature_scale(logits: np.ndarray, targets: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    temperature = torch.ones(1, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=50)

    def closure():
        optimizer.zero_grad()
        safe_temperature = temperature.clamp(0.05, 10.0)
        loss = criterion(logits_tensor / safe_temperature, targets_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(0.05, 10.0).item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = torch.tensor(logits, dtype=torch.float32) / temperature
    return torch.softmax(scaled, dim=1).numpy()


def selective_prediction_table(probabilities: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    rows = []
    for threshold in np.arange(0.50, 0.96, 0.05):
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        accepted_accuracy = float((predictions[accepted] == targets[accepted]).mean()) if accepted.any() else np.nan
        rows.append({
            "threshold": round(float(threshold), 2),
            "coverage": coverage,
            "review_rate": 1.0 - coverage,
            "accepted_accuracy": accepted_accuracy,
            "accepted_images": int(accepted.sum()),
        })
    return pd.DataFrame(rows)


def class_error_table(targets: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    matrix = confusion_matrix(targets, predictions, labels=np.arange(10))
    support = matrix.sum(axis=1)
    correct = np.diag(matrix)
    accuracy = correct / np.maximum(support, 1)
    return pd.DataFrame({
        "class": CLASS_NAMES,
        "support": support,
        "correct": correct,
        "class_accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
    }).sort_values("error_rate", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compact", "regularized", "resnet18"], default="regularized")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    train_set, validation_set, test_set = make_datasets()
    train_loader = make_loader(train_set, args.batch_size, True)
    validation_loader = make_loader(validation_set, args.batch_size, False)
    test_loader = make_loader(test_set, args.batch_size, False)

    if args.model == "compact":
        model = CompactCNN()
    elif args.model == "regularized":
        model = RegularizedCNN()
    else:
        model = build_resnet18()
    model = model.to(DEVICE)

    print("device:", DEVICE)
    print("model:", args.model)
    print("parameters:", parameter_count(model))

    history, model = fit_model(model, train_loader, validation_loader, args.epochs, args.learning_rate)
    history.to_csv(ARTIFACTS / f"{args.model}_history.csv", index=False)

    criterion = nn.CrossEntropyLoss()
    validation_metrics = evaluate(model, validation_loader, criterion)
    test_metrics = evaluate(model, test_loader, criterion)
    temperature = temperature_scale(validation_metrics["logits"], validation_metrics["targets"])
    calibrated_probabilities = apply_temperature(test_metrics["logits"], temperature)
    calibrated_predictions = calibrated_probabilities.argmax(axis=1)

    raw_ece = expected_calibration_error(test_metrics["probabilities"], test_metrics["targets"])
    calibrated_ece = expected_calibration_error(calibrated_probabilities, test_metrics["targets"])

    report = classification_report(
        test_metrics["targets"], calibrated_predictions, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(ARTIFACTS / f"{args.model}_classification_report.csv")

    error_table = class_error_table(test_metrics["targets"], calibrated_predictions)
    error_table.to_csv(ARTIFACTS / f"{args.model}_class_errors.csv", index=False)

    routing_table = selective_prediction_table(calibrated_probabilities, test_metrics["targets"])
    routing_table.to_csv(ARTIFACTS / f"{args.model}_selective_prediction.csv", index=False)

    confidence = calibrated_probabilities.max(axis=1)
    prediction_table = pd.DataFrame({
        "target_id": test_metrics["targets"],
        "target": [CLASS_NAMES[index] for index in test_metrics["targets"]],
        "prediction_id": calibrated_predictions,
        "prediction": [CLASS_NAMES[index] for index in calibrated_predictions],
        "confidence": confidence,
        "correct": calibrated_predictions == test_metrics["targets"],
    })
    prediction_table.to_csv(ARTIFACTS / f"{args.model}_test_predictions.csv", index=False)

    metrics = {
        "model": args.model,
        "parameters": parameter_count(model),
        "epochs": args.epochs,
        "dataset": "CIFAR-10",
        "training_images": len(train_set),
        "validation_images": len(validation_set),
        "test_images": len(test_set),
        "test_accuracy": float(accuracy_score(test_metrics["targets"], calibrated_predictions)),
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "temperature": temperature,
    }
    (ARTIFACTS / f"{args.model}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), ARTIFACTS / f"{args.model}_weights.pt")

    print(json.dumps(metrics, indent=2))
    print("\nHighest-error classes")
    print(error_table.head())
    print("\nConfidence-routing trade-off")
    print(routing_table)


if __name__ == "__main__":
    main()
