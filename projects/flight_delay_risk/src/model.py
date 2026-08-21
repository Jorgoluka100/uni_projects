"""Model configuration and threshold selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostClassifier


@dataclass(frozen=True)
class ModelConfig:
    iterations: int = 900
    learning_rate: float = 0.055
    depth: int = 8
    l2_leaf_reg: float = 10.0
    random_strength: float = 0.4
    early_stopping_rounds: int = 70
    seed: int = 42


def build_model(config: ModelConfig = ModelConfig()) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=config.seed,
        random_strength=config.random_strength,
        od_type="Iter",
        od_wait=config.early_stopping_rounds,
        verbose=100,
        allow_writing_files=False,
    )


def threshold_for_capacity(scores: np.ndarray, capacity: float) -> float:
    """Choose the validation threshold that flags approximately `capacity` of flights."""
    if not 0.01 <= capacity <= 0.90:
        raise ValueError("capacity must be between 0.01 and 0.90")
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("scores must not be empty")
    threshold = float(np.quantile(scores, 1.0 - capacity))
    return float(np.clip(threshold, 0.0, 1.0))
