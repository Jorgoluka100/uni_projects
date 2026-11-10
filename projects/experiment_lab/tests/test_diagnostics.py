from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.diagnostics import (
    effect_estimate,
    numeric_balance_table,
    power_curve,
    standardized_mean_difference,
)


def test_constant_but_different_experiment_arms_are_not_reported_as_balanced():
    smd = standardized_mean_difference(np.array([5.0, 5.0]), np.array([1.0, 1.0]))
    assert np.isinf(smd)

    frame = pd.DataFrame({
        "treatment": [0, 0, 1, 1],
        "pre_metric": [1.0, 1.0, 5.0, 5.0],
    })
    result = numeric_balance_table(frame, "treatment", ["pre_metric"])
    assert not bool(result.loc[0, "passed"])


def test_effect_estimate_rejects_non_binary_and_tiny_arms():
    with pytest.raises(ValueError, match="binary"):
        effect_estimate([1, 2, 3, 4], [0, 0, 1, 2], "overall")
    with pytest.raises(ValueError, match="at least two"):
        effect_estimate([1, 2, 3], [0, 0, 1], "overall")
    with pytest.raises(ValueError, match="confidence"):
        effect_estimate([1, 2, 3, 4], [0, 0, 1, 1], "overall", confidence=1.0)


def test_power_curve_uses_integer_arm_sizes_and_validates_inputs():
    curve = power_curve(10.0, [100, 101], target_effect=2.5)
    assert curve["total_rows"].tolist() == [100, 101]
    assert curve.loc[1, "achieved_power"] > 0
    with pytest.raises(ValueError, match="at least four"):
        power_curve(10.0, [3])

from run import cuped_adjust, mean_effect, power_and_mde, simulate, stratified_bootstrap


def test_simulation_uses_balanced_random_assignment_for_odd_samples():
    frame = simulate(101, 42, 2.5)
    assert abs(int(frame["treatment"].sum()) - int((frame["treatment"] == 0).sum())) == 1


def test_experiment_estimators_reject_degenerate_inputs():
    with pytest.raises(ValueError, match="positive sample variance"):
        cuped_adjust(np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4))
    with pytest.raises(ValueError, match="at least two"):
        mean_effect(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 1]))
    with pytest.raises(ValueError, match="at least 100"):
        stratified_bootstrap(np.arange(10.0), np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), 10, 42)


def test_power_calculation_validates_and_handles_odd_sample_sizes():
    result = power_and_mde(10.0, 101)
    assert result["mde_outcome_units"] > 0
    with pytest.raises(ValueError, match="alpha"):
        power_and_mde(10.0, 100, alpha=1.0)
