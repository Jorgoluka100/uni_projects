from __future__ import annotations

import numpy as np

from .data import CATEGORICAL_FEATURES, FEATURES

MODEL_CONFIG = {
    "iterations": 500,
    "depth": 9,
    "learning_rate": 0.08,
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "l2_leaf_reg": 8,
    "random_seed": 42,
    "allow_writing_files": False,
    "thread_count": -1,
    "verbose": False,
}


def build_model():
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**MODEL_CONFIG)


def fit_model(train, validation):
    model = build_model()
    model.fit(
        train[FEATURES],
        np.log1p(train.price),
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(validation[FEATURES], np.log1p(validation.price)),
        early_stopping_rounds=50,
    )
    return model


def predict_price(model, frame):
    return np.expm1(model.predict(frame[FEATURES])).clip(20_000, 5_000_000)
