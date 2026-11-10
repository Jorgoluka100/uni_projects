import numpy as np
import pandas as pd

from src.evaluation import AreaPropertyBaseline, conformal_radius, regression_metrics


def test_area_property_baseline_fallbacks():
    train = pd.DataFrame({
        "postcode_district": ["E1", "E1", "SW1"],
        "property_type": ["F", "F", "T"],
        "price": [300000, 320000, 700000],
    })
    test = pd.DataFrame({
        "postcode_district": ["E1", "N1", "N1"],
        "property_type": ["F", "F", "D"],
    })
    prediction = AreaPropertyBaseline().fit(train).predict(test)
    assert np.allclose(prediction, [310000, 310000, 320000])


def test_regression_metrics_are_exact_for_perfect_prediction():
    metrics = regression_metrics([100000, 200000], [100000, 200000])
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["mape_pct"] == 0.0
    assert metrics["within_20pct"] == 1.0


def test_conformal_radius_uses_validation_errors_only():
    radius = conformal_radius([100, 200, 300, 400], [100, 180, 260, 450], coverage=0.75)
    assert radius == 50.0

def test_regression_metrics_reject_zero_or_non_finite_prices():
    import pytest

    with pytest.raises(ValueError, match="strictly positive"):
        regression_metrics([100.0, 0.0], [90.0, 10.0])
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([100.0, np.nan], [90.0, 10.0])


def test_conformal_radius_uses_finite_sample_rank():
    # For n=4 and 90% coverage, ceil((n+1)*0.9)=5, capped at n.
    assert conformal_radius([1, 2, 3, 4], [0, 0, 0, 0], coverage=0.9) == 4.0


def test_area_baseline_requires_fit_and_valid_training_prices():
    import pytest

    baseline = AreaPropertyBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        baseline.predict(pd.DataFrame({"postcode_district": ["E1"], "property_type": ["F"]}))
    with pytest.raises(ValueError, match="strictly positive"):
        baseline.fit(pd.DataFrame({"postcode_district": ["E1"], "property_type": ["F"], "price": [0]}))
