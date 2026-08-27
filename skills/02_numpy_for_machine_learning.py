"""Focused NumPy operations used in machine learning."""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    X = np.array(
        [
            [1.0, 20.0, 0.2],
            [1.0, 35.0, 0.7],
            [1.0, 50.0, 1.1],
            [1.0, 65.0, 1.5],
        ]
    )
    weights = np.array([-2.0, 0.04, 1.2])
    scores = X @ weights

    features = X[:, 1:]
    standardised = (features - features.mean(axis=0)) / features.std(axis=0)
    probabilities = sigmoid(scores)
    predictions = (probabilities >= 0.5).astype(int)

    print("X shape:", X.shape)
    print("matrix multiplication:", scores)
    print("scaled means:", standardised.mean(axis=0).round(6))
    print("probabilities:", probabilities.round(3))
    print("predictions:", predictions)


if __name__ == "__main__":
    main()
