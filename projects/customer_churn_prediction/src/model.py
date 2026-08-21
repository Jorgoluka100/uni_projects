from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from .features import engineer_features

SEED = 42


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=SEED,
    )


def safe_logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return logit(p).reshape(-1, 1)


def fit_calibrated_model(raw_train, y_train, groups):
    """Fit calibration only on grouped out-of-fold training predictions."""
    X = engineer_features(raw_train)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    template = build_model()
    oof_raw = cross_val_predict(
        template,
        X,
        y_train,
        groups=groups,
        cv=splitter,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    calibrator = LogisticRegression(C=1e6, max_iter=2000, random_state=SEED)
    calibrator.fit(safe_logit(oof_raw), y_train)
    base_model = clone(template).fit(X, y_train)
    return base_model, calibrator, oof_raw


def calibrated_probability(base_model, calibrator, raw_frame) -> np.ndarray:
    raw_probability = base_model.predict_proba(engineer_features(raw_frame))[:, 1]
    calibrated_logit = calibrator.decision_function(safe_logit(raw_probability))
    return expit(calibrated_logit)
