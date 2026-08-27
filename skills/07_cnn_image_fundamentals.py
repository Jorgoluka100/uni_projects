"""Compact convolutional neural-network fundamentals in PyTorch."""

import numpy as np
import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(16 * 3 * 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def make_image(label: int, size: int = 12) -> np.ndarray:
    image = np.random.normal(0, 0.08, (size, size)).astype("float32")
    if label == 0:
        image[:, 5:7] += 1.0
    else:
        image[5:7, :] += 1.0
    return image


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    labels = np.array([0, 1] * 500)
    images = np.stack([make_image(label) for label in labels])
    X = torch.tensor(images).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)

    perm = torch.randperm(len(X))
    cut = int(0.8 * len(X))
    train_idx, test_idx = perm[:cut], perm[cut:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = TinyCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(61):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 15 == 0:
            print(f"epoch={epoch:2d} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(dim=1)
        accuracy = (pred == y_test).float().mean().item()
    print(f"test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
