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
