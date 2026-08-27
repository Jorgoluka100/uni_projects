"""Long Short-Term Memory sequence-classification fundamentals in PyTorch."""

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 24) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.output(hidden[-1])


def main() -> None:
    torch.manual_seed(42)
    samples, steps, features = 1600, 20, 1
    X = torch.randn(samples, steps, features)
    first_half = X[:, :10, 0].sum(dim=1)
    second_half = X[:, 10:, 0].sum(dim=1)
    y = (second_half > first_half).float().unsqueeze(1)

    perm = torch.randperm(samples)
    cut = int(samples * 0.8)
    train_idx, test_idx = perm[:cut], perm[cut:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = LSTMClassifier()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(151):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 30 == 0:
            print(f"epoch={epoch:3d} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_test))
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()
    print(f"test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
