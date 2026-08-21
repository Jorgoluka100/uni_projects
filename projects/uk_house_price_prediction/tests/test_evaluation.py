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
