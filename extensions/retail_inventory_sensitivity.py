"""Lead-time and service-factor sensitivity for the retail replenishment notebook.

The helper is deliberately separate from the original notebook so the decision
logic can be unit-tested without downloading the UCI transaction dataset.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "date",
    "StockCode",
    "demand",
    "forecast",
    "residual_std",
    "rolling_mean_7",
)


def _positive_integer_values(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    if any(isinstance(value, bool) or int(value) != value or int(value) <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(int(value) for value in result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(int(value) for value in result)


def _positive_finite_values(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive finite values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def inventory_sensitivity(
    policy: pd.DataFrame,
    lead_times: Iterable[int] = (3, 7, 14),
    service_z_values: Iterable[float] = (1.282, 1.645, 1.96),
    stockout_cost_per_unit: float = 4.0,
    holding_cost_per_excess_unit: float = 0.05,
) -> pd.DataFrame:
    """Compare replenishment policies across lead-time and safety-factor scenarios.

    For each SKU/date, the realised lead-time demand is calculated retrospectively
    from that date forward. The forecast and residual standard deviation are
    treated as information available at the decision date. Cost rates and z-values
    are scenario assumptions, not observed company costs or guaranteed service.
    """
    if not isinstance(policy, pd.DataFrame):
        raise TypeError("policy must be a pandas DataFrame")
    missing = sorted(set(REQUIRED_COLUMNS) - set(policy.columns))
    if missing:
        raise ValueError(f"policy is missing required columns: {missing}")
    if policy.empty:
        raise ValueError("policy must not be empty")

    lead_times = _positive_integer_values(lead_times, "lead_times")
    service_z_values = _positive_finite_values(service_z_values, "service_z_values")
    costs = (float(stockout_cost_per_unit), float(holding_cost_per_excess_unit))
    if any(not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError("cost assumptions must be finite and non-negative")

    frame = policy.loc[:, REQUIRED_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["StockCode"].isna().any():
        raise ValueError("date and StockCode must be complete")

    numeric_columns = ["demand", "forecast", "residual_std", "rolling_mean_7"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("demand, forecast and inventory inputs must be finite")
    if (frame[numeric_columns] < 0).any().any():
        raise ValueError("demand, forecast, residual_std and rolling_mean_7 must be non-negative")
    if frame.duplicated(["StockCode", "date"]).any():
        raise ValueError("policy must contain at most one row per SKU and date")

    frame = frame.sort_values(["StockCode", "date"]).reset_index(drop=True)
    rows: list[dict[str, float | int]] = []

    for lead_time in lead_times:
        # Reverse rolling windows make each row's target the demand from its date forward.
        frame["actual_lead_time_demand"] = (
            frame.groupby("StockCode", sort=False)["demand"]
            .transform(
                lambda values: values.iloc[::-1]
                .rolling(lead_time, min_periods=lead_time)
                .sum()
                .iloc[::-1]
            )
        )
        eligible = frame.dropna(subset=["actual_lead_time_demand"]).copy()
        if eligible.empty:
            raise ValueError(f"not enough consecutive observations for lead_time={lead_time}")

        actual = eligible["actual_lead_time_demand"].to_numpy(dtype=float)
        baseline_target = (
            eligible["rolling_mean_7"].clip(lower=0).to_numpy(dtype=float) * lead_time
        )
        baseline_shortage = np.maximum(actual - baseline_target, 0.0)
        baseline_excess = np.maximum(baseline_target - actual, 0.0)
        baseline_cost = float(
            baseline_shortage.sum() * costs[0] + baseline_excess.sum() * costs[1]
        )
        baseline_service = float((baseline_shortage == 0).mean())

        for z_value in service_z_values:
            # The square-root lead-time rule assumes approximately independent daily errors.
            safety_stock = (
                z_value
                * eligible["residual_std"].to_numpy(dtype=float)
                * math.sqrt(lead_time)
            )
            recommended_target = (
                eligible["forecast"].to_numpy(dtype=float) * lead_time + safety_stock
            )
            shortage = np.maximum(actual - recommended_target, 0.0)
            excess = np.maximum(recommended_target - actual, 0.0)
            recommended_cost = float(shortage.sum() * costs[0] + excess.sum() * costs[1])

            rows.append(
                {
                    "lead_time_days": lead_time,
                    "service_z": z_value,
                    "simulated_windows": int(len(eligible)),
                    "baseline_service_level": baseline_service,
                    "recommended_service_level": float((shortage == 0).mean()),
                    "recommended_stockout_window_rate": float((shortage > 0).mean()),
                    "baseline_cost_assumption": baseline_cost,
                    "recommended_cost_assumption": recommended_cost,
                    "cost_change_vs_baseline": (
                        recommended_cost / baseline_cost - 1.0
                        if baseline_cost > 0
                        else float("nan")
                    ),
                    "average_reorder_target_units": float(recommended_target.mean()),
                    "shortage_units": float(shortage.sum()),
                    "excess_units": float(excess.sum()),
                }
            )

    return pd.DataFrame(rows)
