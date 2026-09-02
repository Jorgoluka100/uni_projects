from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


PROJECT = Path(__file__).resolve().parents[1]


def test_run_script_compiles() -> None:
    source = (PROJECT / "run.py").read_text(encoding="utf-8")
    compile(source, str(PROJECT / "run.py"), "exec")


def test_required_portfolio_evidence_is_in_source() -> None:
    source = (PROJECT / "run.py").read_text(encoding="utf-8")
    required = [
        "fetch_ucirepo(id=242)",
        "DummyRegressor",
        "LinearRegression",
        "RidgeCV",
        "LassoCV",
        "cross_val_score",
        "learning_curve",
        "linear_residual",
        "bootstrap",
        "scenario_analysis.csv",
        "metrics.json",
    ]
    for token in required:
        assert token in source, token


def test_linear_regression_recovers_a_simple_signal() -> None:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=300)
    x2 = rng.normal(size=300)
    y = 4.0 + 2.5 * x1 - 1.2 * x2 + rng.normal(scale=0.05, size=300)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    assert r2_score(y, pred) > 0.99
    assert abs(model.coef_[0] - 2.5) < 0.05
    assert abs(model.coef_[1] + 1.2) < 0.05


def test_high_load_screening_rule() -> None:
    training_loads = pd.Series([10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    threshold = float(training_loads.quantile(0.75))
    predictions = np.array([15.0, threshold - 0.01, threshold, threshold + 5.0])
    decisions = np.where(predictions >= threshold, "HIGH LOAD - REVIEW", "LOW / NORMAL LOAD")
    assert decisions.tolist() == [
        "LOW / NORMAL LOAD",
        "LOW / NORMAL LOAD",
        "HIGH LOAD - REVIEW",
        "HIGH LOAD - REVIEW",
    ]
