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

def test_confidence_policy_rejects_invalid_probabilities_and_alignment():
    import pytest
    from run import evaluate_confidence_policy

    with pytest.raises(ValueError, match="sum to one"):
        evaluate_confidence_policy(np.array([[0.8, 0.8]]), np.array([0]), np.array([0]), 0.5)
    with pytest.raises(ValueError, match="aligned"):
        evaluate_confidence_policy(np.array([[0.8, 0.2]]), np.array([0]), np.array([0, 1]), 0.5)
    with pytest.raises(ValueError, match="threshold"):
        evaluate_confidence_policy(np.array([[0.8, 0.2]]), np.array([0]), np.array([0]), 1.2)


def test_dataset_audit_rejects_non_finite_features():
    import pytest
    from run import audit_dataset

    x = pd.DataFrame({"feature": [1.0, np.inf]})
    y = pd.Series([0, 1])
    with pytest.raises(ValueError, match="finite"):
        audit_dataset(x, y)
