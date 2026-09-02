from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run import audit_dataset, build_pipeline, load_dataset, predict_one


def test_dataset_contract():
    x, y, names = load_dataset()
    audit = audit_dataset(x, y)
    assert audit.rows == len(x)
    assert audit.columns == x.shape[1]
    assert audit.target_classes == 3
    assert len(names) == 3


def test_pipeline_returns_probabilities():
    x, y, names = load_dataset()
    model = build_pipeline(n_neighbors=5)
    model.fit(x, y)
    result = predict_one(model, x.iloc[[0]], names)
    probabilities = np.array(list(result["class_probabilities"].values()))
    assert np.isclose(probabilities.sum(), 1.0)
    assert 0.0 <= result["confidence"] <= 1.0
