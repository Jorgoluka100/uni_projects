import numpy as np

from src.evaluation import operating_point, scenario_cost, select_cost_threshold


def test_operating_point_reports_decision_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = operating_point(y, p, 0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["alert_rate"] == 0.5


def test_missing_churn_costs_more_than_extra_review():
    y = np.array([0, 1])
    p = np.array([0.4, 0.4])
    conservative = scenario_cost(y, p, 0.5)
    review_both = scenario_cost(y, p, 0.3)
    assert review_both < conservative


def test_threshold_is_selected_from_training_probabilities():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.02, 0.10, 0.35, 0.60, 0.75, 0.95])
    selected = select_cost_threshold(y, p)
    assert 0.01 <= selected["threshold"] <= 0.99
    assert selected["cost_units"] >= 0
