from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run import (
    confidence_policy,
    confusion_pairs,
    corpus_audit,
    length_error_slices,
    normalise_text,
    predict_document,
    threshold_sweep,
)


def test_text_normalisation_redacts_common_contact_patterns():
    # Regression check: common email and URL patterns should not survive preprocessing.
    text = "Email me at person@example.com and see https://example.com/page"
    cleaned = normalise_text(text)
    assert "person@example.com" not in cleaned
    assert "https://example.com/page" not in cleaned
    assert "EMAIL" in cleaned
    assert "URL" in cleaned


def test_confidence_policy_routes_uncertain_rows_to_review():
    # A low top-class probability is escalated rather than automatically routed.
    probabilities = np.array([[0.9, 0.1], [0.52, 0.48], [0.2, 0.8]])
    accepted = confidence_policy(probabilities, threshold=0.60)
    assert accepted.tolist() == [True, False, True]


def test_confidence_policy_rejects_invalid_probability_matrices():
    # These guardrails stop malformed model outputs from silently changing decisions.
    with pytest.raises(ValueError, match="sum to one"):
        confidence_policy(np.array([[0.7, 0.7]]))
    with pytest.raises(ValueError, match="within"):
        confidence_policy(np.array([[1.2, -0.2]]))
    with pytest.raises(ValueError, match="threshold"):
        confidence_policy(np.array([[0.7, 0.3]]), threshold=1.2)
    with pytest.raises(ValueError, match="non-empty"):
        confidence_policy(np.empty((0, 2)))


def test_corpus_audit_reports_class_balance_and_rejects_bad_inputs():
    # A tiny balanced fixture tests the audit without downloading the full benchmark.
    report = corpus_audit(
        ["two useful words", "three useful words", "one word"],
        np.array([0, 1, 0]),
        ["topic_a", "topic_b"],
    )
    assert report["documents"] == 3
    assert report["classes"] == 2
    assert report["class_counts"] == {"topic_a": 2, "topic_b": 1}

    with pytest.raises(ValueError, match="must not be empty"):
        corpus_audit([], np.array([], dtype=int), ["topic_a"])
    with pytest.raises(ValueError, match="aligned"):
        corpus_audit(["one", "two"], np.array([0]), ["topic_a"])
    with pytest.raises(ValueError, match="cover"):
        corpus_audit(["one", "two"], np.array([0, 2]), ["topic_a", "topic_b"])


def test_threshold_sweep_reports_a_stable_coverage_tradeoff():
    probabilities = np.array([[0.95, 0.05], [0.60, 0.40], [0.52, 0.48], [0.10, 0.90]])
    truth = np.array([0, 0, 1, 1])
    prediction = np.array([0, 0, 0, 1])
    result = threshold_sweep(probabilities, truth, prediction)

    assert list(result.columns) == [
        "threshold", "auto_route_rate", "auto_route_accuracy", "review_rate", "review_rows"
    ]
    assert result["auto_route_rate"].is_monotonic_decreasing
    assert np.allclose(result["auto_route_rate"] + result["review_rate"], 1.0)

    with pytest.raises(ValueError, match="aligned"):
        threshold_sweep(probabilities, truth[:-1], prediction[:-1])


def test_confusion_pairs_keep_columns_when_there_are_no_errors():
    # Stable empty outputs are easier to consume in dashboards and automated reports.
    perfect = np.diag([5, 7])
    result = confusion_pairs(perfect, ["topic_a", "topic_b"])
    assert result.empty
    assert list(result.columns) == ["true_category", "predicted_category", "count"]

    with pytest.raises(ValueError, match="square"):
        confusion_pairs(np.zeros((2, 3)), ["topic_a", "topic_b"])


def test_length_error_slices_require_aligned_inputs():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.9, 0.1]])
    result = length_error_slices(
        ["short text", "a medium length document", "another short item"],
        np.array([0, 1, 0]),
        np.array([0, 1, 1]),
        probabilities,
    )
    assert int(result["rows"].sum()) == 3

    with pytest.raises(ValueError, match="aligned"):
        length_error_slices(["one"], np.array([0, 1]), np.array([0, 1]), probabilities[:2])


def test_predict_document_returns_a_reviewable_decision():
    class DummyModel:
        def predict_proba(self, texts):
            return np.array([[0.72, 0.28]])

    result = predict_document(DummyModel(), "A clear example document", ["topic_a", "topic_b"])
    assert result["predicted_category"] == "topic_a"
    assert result["decision"] == "auto_route"
    assert len(result["top_categories"]) == 2

    with pytest.raises(ValueError, match="non-empty"):
        predict_document(DummyModel(), " ", ["topic_a", "topic_b"])
