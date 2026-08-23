from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

SEED = 42


@dataclass(frozen=True)
class ClusterSelection:
    selected_k: int
    diagnostics: pd.DataFrame


def evaluate_cluster_counts(
    matrix: np.ndarray,
    k_values: range | list[int] = range(2, 11),
    seed: int = SEED,
) -> ClusterSelection:
    """Compare candidate k values with multiple complementary diagnostics."""
    if len(matrix) < 10:
        raise ValueError("Too few customers for meaningful clustering")

    rows: list[dict] = []
    for k in k_values:
        if k >= len(matrix):
            continue
        model = KMeans(n_clusters=k, n_init=50, random_state=seed)
        labels = model.fit_predict(matrix)
        rows.append(
            {
                "k": int(k),
                "silhouette": float(silhouette_score(matrix, labels)),
                "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
                "inertia": float(model.inertia_),
                "smallest_cluster_share": float(pd.Series(labels).value_counts(normalize=True).min()),
            }
        )

    diagnostics = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    if diagnostics.empty:
        raise ValueError("No valid cluster counts were evaluated")

    # Silhouette is the primary selection rule. A minimum cluster-size guard avoids
    # promoting a superficially strong solution dominated by tiny fragments.
    eligible = diagnostics.loc[diagnostics["smallest_cluster_share"] >= 0.02]
    selection_pool = eligible if not eligible.empty else diagnostics
    selected_k = int(selection_pool.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    return ClusterSelection(selected_k=selected_k, diagnostics=diagnostics)


def fit_final_kmeans(matrix: np.ndarray, n_clusters: int, seed: int = SEED) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, n_init=100, random_state=seed)
    labels = model.fit_predict(matrix)
    return model, labels


def cluster_stability(
    matrix: np.ndarray,
    n_clusters: int,
    reference_labels: np.ndarray,
    seeds: tuple[int, ...] = (7, 19, 31, 43, 59, 71, 83, 97),
) -> dict:
    """Estimate sensitivity to KMeans initialization using adjusted Rand index."""
    scores = []
    for seed in seeds:
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        candidate = model.fit_predict(matrix)
        scores.append(float(adjusted_rand_score(reference_labels, candidate)))

    return {
        "metric": "adjusted_rand_index",
        "runs": len(scores),
        "mean": float(np.mean(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "scores": scores,
    }
