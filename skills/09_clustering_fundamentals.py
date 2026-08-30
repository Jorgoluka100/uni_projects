"""Focused clustering fundamentals: scaling, k selection and interpretation."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    a = rng.normal([20, 2, 120], [4, 0.5, 18], size=(90, 3))
    b = rng.normal([45, 7, 420], [6, 1.0, 45], size=(100, 3))
    c = rng.normal([75, 12, 850], [8, 1.3, 80], size=(80, 3))
    data = np.vstack([a, b, c])
    return pd.DataFrame(data, columns=["recency_days", "orders", "spend"])


def main() -> None:
    df = build_data()
    X = StandardScaler().fit_transform(df)
    scores: dict[int, float] = {}

    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
        scores[k] = silhouette_score(X, labels)

    best_k = max(scores, key=scores.get)
    final = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    df["cluster"] = final.fit_predict(X)

    print("Silhouette by k:", {k: round(v, 3) for k, v in scores.items()})
    print("Selected k:", best_k)
    print(df.groupby("cluster")[["recency_days", "orders", "spend"]].mean().round(1))


if __name__ == "__main__":
    main()
