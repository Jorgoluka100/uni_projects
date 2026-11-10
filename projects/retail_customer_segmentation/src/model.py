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


def _validated_matrix(matrix) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("matrix must be 2D with at least three rows and one feature")
    if not np.isfinite(values).all():
        raise ValueError("matrix must contain only finite values")
    if np.unique(values, axis=0).shape[0] < 2:
        raise ValueError("matrix must contain at least two distinct observations")
    return values


def evaluate_cluster_counts(
    matrix: np.ndarray,
    k_values: range | list[int] = range(2, 11),
    seed: int = SEED,
) -> ClusterSelection:
    """Compare candidate k values with multiple complementary diagnostics."""
    values = _validated_matrix(matrix)
    if len(values) < 10:
        raise ValueError("Too few customers for meaningful clustering")

    unique_rows = np.unique(values, axis=0).shape[0]
    rows: list[dict] = []
    for k in sorted(set(k_values)):
        if not isinstance(k, (int, np.integer)) or k < 2:
            raise ValueError("candidate cluster counts must be integers >= 2")
        if k >= len(values) or k > unique_rows:
            continue
        model = KMeans(n_clusters=k, n_init=50, random_state=seed)
        labels = model.fit_predict(values)
        distinct_labels = np.unique(labels).size
        if distinct_labels < 2 or distinct_labels >= len(values):
            continue
        rows.append(
            {
                "k": int(k),
                "silhouette": float(silhouette_score(values, labels)),
                "davies_bouldin": float(davies_bouldin_score(values, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(values, labels)),
                "inertia": float(model.inertia_),
                "smallest_cluster_share": float(pd.Series(labels).value_counts(normalize=True).min()),
            }
        )

    diagnostics = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    if diagnostics.empty:
        raise ValueError("No valid cluster counts were evaluated")
    eligible = diagnostics.loc[diagnostics["smallest_cluster_share"] >= 0.02]
    selection_pool = eligible if not eligible.empty else diagnostics
    selected_k = int(selection_pool.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    return ClusterSelection(selected_k=selected_k, diagnostics=diagnostics)


def fit_final_kmeans(matrix: np.ndarray, n_clusters: int, seed: int = SEED) -> tuple[KMeans, np.ndarray]:
    values = _validated_matrix(matrix)
    if not isinstance(n_clusters, (int, np.integer)) or not 2 <= n_clusters < len(values):
        raise ValueError("n_clusters must be an integer between 2 and n_samples - 1")
    if n_clusters > np.unique(values, axis=0).shape[0]:
        raise ValueError("n_clusters cannot exceed the number of distinct observations")
    model = KMeans(n_clusters=n_clusters, n_init=100, random_state=seed)
    labels = model.fit_predict(values)
    if np.unique(labels).size != n_clusters:
        raise ValueError("KMeans returned fewer distinct clusters than requested")
    return model, labels


def cluster_stability(
    matrix: np.ndarray,
    n_clusters: int,
    reference_labels: np.ndarray,
    seeds: tuple[int, ...] = (7, 19, 31, 43, 59, 71, 83, 97),
) -> dict:
    """Estimate sensitivity to KMeans initialization using adjusted Rand index."""
    values = _validated_matrix(matrix)
    labels = np.asarray(reference_labels)
    if labels.ndim != 1 or labels.size != len(values):
        raise ValueError("reference_labels must be 1D and align with matrix rows")
    if np.unique(labels).size != n_clusters or not 2 <= n_clusters < len(values):
        raise ValueError("reference_labels must contain exactly n_clusters valid groups")
    if not seeds:
        raise ValueError("at least one random seed is required")
    scores = []
    for seed in seeds:
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        candidate = model.fit_predict(values)
        scores.append(float(adjusted_rand_score(labels, candidate)))

    return {
        "metric": "adjusted_rand_index",
        "runs": len(scores),
        "mean": float(np.mean(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "scores": scores,
    }
