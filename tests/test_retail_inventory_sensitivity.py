from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extensions"))

from retail_inventory_sensitivity import inventory_sensitivity


def sample_policy() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    return pd.DataFrame(
        [
            {"date": date, "StockCode": "SKU-A", "demand": demand,
             "forecast": 2.0, "residual_std": 1.0, "rolling_mean_7": 2.0}
            for date, demand in zip(dates, [1, 2, 3, 4, 5])
        ]
        + [
            {"date": date, "StockCode": "SKU-B", "demand": demand,
             "forecast": 1.5, "residual_std": 0.5, "rolling_mean_7": 1.5}
            for date, demand in zip(dates, [2, 1, 2, 1, 2])
        ]
    )


def test_sensitivity_covers_lead_time_and_service_factor_grid():
    result = inventory_sensitivity(
        sample_policy(),
        lead_times=(2, 3),
        service_z_values=(0.5, 1.0, 1.5),
    )

    assert len(result) == 6
    assert set(result["lead_time_days"]) == {2, 3}
    assert set(result["service_z"]) == {0.5, 1.0, 1.5}
    assert set(result.loc[result["lead_time_days"] == 2, "simulated_windows"]) == {8}
    assert set(result.loc[result["lead_time_days"] == 3, "simulated_windows"]) == {6}


def test_higher_safety_factor_increases_reorder_target_and_does_not_reduce_service():
    result = inventory_sensitivity(
        sample_policy(),
        lead_times=(2,),
        service_z_values=(0.5, 1.0, 1.5),
    ).sort_values("service_z")

    assert result["average_reorder_target_units"].is_monotonic_increasing
    assert result["recommended_service_level"].is_monotonic_increasing


def test_sensitivity_rejects_missing_duplicate_and_invalid_data():
    frame = sample_policy()
    with pytest.raises(ValueError, match="missing required columns"):
        inventory_sensitivity(frame.drop(columns=["forecast"]))
    with pytest.raises(ValueError, match="at most one row"):
        inventory_sensitivity(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    invalid = frame.copy()
    invalid.loc[0, "demand"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        inventory_sensitivity(invalid)


def test_sensitivity_rejects_invalid_scenarios_and_short_history():
    with pytest.raises(ValueError, match="positive integers"):
        inventory_sensitivity(sample_policy(), lead_times=(0,))
    with pytest.raises(ValueError, match="positive finite"):
        inventory_sensitivity(sample_policy(), service_z_values=(float("nan"),))
    with pytest.raises(ValueError, match="non-negative"):
        inventory_sensitivity(sample_policy(), stockout_cost_per_unit=-1)
    with pytest.raises(ValueError, match="not enough consecutive"):
        inventory_sensitivity(sample_policy(), lead_times=(10,))
