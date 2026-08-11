"""Movie recommender v2 — temporal ranking evaluation and cold-start fallback.

This extension keeps the restored MovieLens work but makes the evaluation closer to
an actual recommendation decision. Rather than relying mainly on random rating RMSE,
it uses each eligible user's latest positive interaction as a held-out target,
compares a popularity baseline with a latent-factor recommender, and reports
Recall@K / NDCG@K over a reproducible candidate set.

The script can download GroupLens ml-latest-small or use an existing extracted
directory. It does not claim that MovieLens behaviour represents a current streaming
service.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
SEED = 42


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    k: int = 10
    n_components: int = 48
    min_user_ratings: int = 5
    positive_threshold: float = 4.0
    sampled_negatives: int = 100
    artifact_dir: Path = Path("recommender_artifacts")


def download_movielens(target_dir: Path, timeout: int = 120) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = target_dir / "ratings.csv"
    movies_path = target_dir / "movies.csv"
    if ratings_path.exists() and movies_path.exists():
        return target_dir

    response = requests.get(MOVIELENS_URL, timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        prefix = next(
            name.split("/")[0]
            for name in archive.namelist()
            if name.endswith("/ratings.csv")
        )
        for filename in ("ratings.csv", "movies.csv"):
            with archive.open(f"{prefix}/{filename}") as source:
                (target_dir / filename).write_bytes(source.read())
    return target_dir


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    required_r = {"userId", "movieId", "rating", "timestamp"}
    required_m = {"movieId", "title", "genres"}
    if not required_r.issubset(ratings.columns) or not required_m.issubset(movies.columns):
        raise ValueError("MovieLens schema changed or files are not compatible")
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    ratings = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()
    ratings[["userId", "movieId"]] = ratings[["userId", "movieId"]].astype(int)
    if ratings.duplicated(["userId", "movieId", "timestamp"]).any():
        ratings = ratings.drop_duplicates(["userId", "movieId", "timestamp"], keep="last")
    return ratings, movies


def leave_latest_positive_out(
    ratings: pd.DataFrame,
    min_user_ratings: int,
    positive_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"]).copy()
    counts = ordered.groupby("userId").size()
    eligible = counts[counts >= min_user_ratings].index

    positive = ordered.loc[
        ordered["userId"].isin(eligible) & ordered["rating"].ge(positive_threshold)
    ].copy()
    if positive.empty:
        raise ValueError("no eligible positive interactions")
    test_idx = positive.groupby("userId")["timestamp"].idxmax()
    test = ordered.loc[test_idx].copy()
    train = ordered.drop(index=test_idx).copy()

    latest_train = train.groupby("userId")["timestamp"].max()
    merged = test.join(latest_train.rename("latest_train"), on="userId")
    if not (merged["latest_train"] < merged["timestamp"]).all():
        raise AssertionError("temporal user leakage detected")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def popularity_scores(train: pd.DataFrame) -> pd.Series:
    stats = train.groupby("movieId")["rating"].agg(["count", "mean"])
    global_mean = float(train["rating"].mean())
    prior = 20.0
    score = (stats["count"] * stats["mean"] + prior * global_mean) / (stats["count"] + prior)
    return score.sort_values(ascending=False)


class LatentFactorRecommender:
    def __init__(self, n_components: int, seed: int):
        self.n_components = n_components
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "LatentFactorRecommender":
        self.user_ids = np.sort(train["userId"].unique())
        self.movie_ids = np.sort(train["movieId"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}

        rows = train["userId"].map(self.user_to_idx).to_numpy()
        cols = train["movieId"].map(self.movie_to_idx).to_numpy()
        global_mean = float(train["rating"].mean())
        values = train["rating"].to_numpy(dtype=float) - global_mean
        matrix = csr_matrix((values, (rows, cols)), shape=(len(self.user_ids), len(self.movie_ids)))

        max_components = max(2, min(matrix.shape) - 1)
        n_components = min(self.n_components, max_components)
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_.T
        self.global_mean = global_mean
        return self

    def score(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        if user_id not in self.user_to_idx:
            return np.full(len(movie_ids), self.global_mean)
        u = self.user_factors[self.user_to_idx[user_id]]
        out = np.full(len(movie_ids), self.global_mean, dtype=float)
        for i, movie_id in enumerate(movie_ids):
            idx = self.movie_to_idx.get(int(movie_id))
            if idx is not None:
                out[i] += float(np.dot(u, self.item_factors[idx]))
        return out


def candidate_set(
    rng: np.random.Generator,
    train: pd.DataFrame,
    user_id: int,
    target_movie: int,
    all_movies: np.ndarray,
    n_negatives: int,
) -> np.ndarray:
    seen = set(train.loc[train["userId"].eq(user_id), "movieId"].astype(int))
    pool = np.array([m for m in all_movies if m not in seen and m != target_movie], dtype=int)
    n = min(n_negatives, len(pool))
    negatives = rng.choice(pool, size=n, replace=False) if n else np.array([], dtype=int)
    return np.r_[target_movie, negatives].astype(int)


def rank_metrics(rank: int, k: int) -> tuple[float, float]:
    if rank > k:
        return 0.0, 0.0
    return 1.0, 1.0 / math.log2(rank + 1)


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(config.seed)
    all_movies = np.sort(train["movieId"].unique())
    popularity = popularity_scores(train)
    latent = LatentFactorRecommender(config.n_components, config.seed).fit(train)

    rows = []
    for row in test.itertuples(index=False):
        user = int(row.userId)
        target = int(row.movieId)
        candidates = candidate_set(
            rng, train, user, target, all_movies, config.sampled_negatives
        )
        if len(candidates) < 2:
            continue

        pop_score = np.array([float(popularity.get(int(m), -np.inf)) for m in candidates])
        latent_score = latent.score(user, candidates)

        for model_name, scores in (("popularity", pop_score), ("latent_svd", latent_score)):
            order = np.argsort(-scores, kind="mergesort")
            ranked = candidates[order]
            rank = int(np.where(ranked == target)[0][0]) + 1
            recall, ndcg = rank_metrics(rank, config.k)
            rows.append(
                {
                    "userId": user,
                    "target_movieId": target,
                    "model": model_name,
                    "rank": rank,
                    f"recall@{config.k}": recall,
                    f"ndcg@{config.k}": ndcg,
                    "candidate_count": len(candidates),
                }
            )

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("evaluation produced no users")
    summary = {}
    for model, group in detail.groupby("model"):
        summary[model] = {
            "users": int(group["userId"].nunique()),
            f"recall@{config.k}": float(group[f"recall@{config.k}"].mean()),
            f"ndcg@{config.k}": float(group[f"ndcg@{config.k}"].mean()),
            "median_rank": float(group["rank"].median()),
        }
    return detail, summary


def run(data_dir: Path, config: Config = Config()) -> dict[str, object]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    ratings, movies = load_data(data_dir)
    train, test = leave_latest_positive_out(
        ratings,
        config.min_user_ratings,
        config.positive_threshold,
    )
    detail, summary = evaluate(train, test, config)

    detail = detail.merge(
        movies[["movieId", "title"]].rename(columns={"movieId": "target_movieId"}),
        on="target_movieId",
        how="left",
    )
    detail.to_csv(config.artifact_dir / "ranking_evaluation.csv", index=False)
    payload = {
        "config": {**asdict(config), "artifact_dir": str(config.artifact_dir)},
        "source": MOVIELENS_URL,
        "rows": {"ratings": len(ratings), "train": len(train), "held_out_users": len(test)},
        "evaluation": summary,
        "limitations": [
            "MovieLens is an offline benchmark and not current product traffic.",
            "Sampled-negative ranking is cheaper than full-catalog evaluation and must be labelled as such.",
            "Latent factors do not solve new-user cold start; popularity is the explicit fallback.",
        ],
    }
    (config.artifact_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("movielens_data"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=Path("recommender_artifacts"))
    args = parser.parse_args()
    data_dir = args.data_dir if args.no_download else download_movielens(args.data_dir)
    result = run(data_dir, Config(artifact_dir=args.artifact_dir))
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
