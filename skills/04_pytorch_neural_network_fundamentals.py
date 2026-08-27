"""PyTorch neural-network fundamentals: forward pass, loss, backprop and evaluation."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def main() -> None:
    torch.manual_seed(42)
    n = 1200
    X = torch.randn(n, 2)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1.0).float().unsqueeze(1)

    indices = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = MLP()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(201):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"epoch={epoch:3d} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_test))
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()
    print(f"test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
