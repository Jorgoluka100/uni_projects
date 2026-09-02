from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class EffectEstimate:
    segment: str
    treatment_rows: int
    control_rows: int
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class BalanceCheck:
    variable: str
    treatment_mean: float
    control_mean: float
    standardized_difference: float
    passed: bool


def standardized_mean_difference(
    treatment_values: np.ndarray,
    control_values: np.ndarray,
) -> float:
    treatment_values = np.asarray(treatment_values, dtype=float)
    control_values = np.asarray(control_values, dtype=float)
    treatment_variance = float(np.var(treatment_values, ddof=1))
    control_variance = float(np.var(control_values, ddof=1))
    pooled_sd = float(np.sqrt((treatment_variance + control_variance) / 2.0))
    if pooled_sd == 0.0:
        return 0.0
    return float((treatment_values.mean() - control_values.mean()) / pooled_sd)


def numeric_balance_table(
    frame: pd.DataFrame,
    treatment_col: str,
    numeric_columns: Iterable[str],
    threshold: float = 0.10,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    treatment_mask = frame[treatment_col].to_numpy() == 1
    for column in numeric_columns:
        values = frame[column].to_numpy(dtype=float)
        treated = values[treatment_mask]
        control = values[~treatment_mask]
        smd = standardized_mean_difference(treated, control)
        check = BalanceCheck(
            variable=column,
            treatment_mean=float(treated.mean()),
            control_mean=float(control.mean()),
            standardized_difference=smd,
            passed=bool(abs(smd) < threshold),
        )
        rows.append(asdict(check))
    return pd.DataFrame(rows)


def categorical_balance_table(
    frame: pd.DataFrame,
    treatment_col: str,
    categorical_column: str,
) -> pd.DataFrame:
    counts = (
        frame.groupby([treatment_col, categorical_column], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    totals = counts.groupby(treatment_col)["rows"].transform("sum")
    counts["share"] = counts["rows"] / totals
    pivot = counts.pivot(
        index=categorical_column,
        columns=treatment_col,
        values="share",
    ).fillna(0.0)
    for expected in (0, 1):
        if expected not in pivot.columns:
            pivot[expected] = 0.0
    output = pivot.rename(columns={0: "control_share", 1: "treatment_share"}).reset_index()
    output["absolute_share_gap"] = (
        output["treatment_share"] - output["control_share"]
    ).abs()
    return output.sort_values("absolute_share_gap", ascending=False)


def effect_estimate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    segment: str,
    confidence: float = 0.95,
) -> EffectEstimate:
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    treated = outcome[treatment == 1]
    control = outcome[treatment == 0]
    if len(treated) < 2 or len(control) < 2:
        raise ValueError("Both experiment arms need at least two observations")
    estimate = float(treated.mean() - control.mean())
    standard_error = float(
        np.sqrt(
            treated.var(ddof=1) / len(treated)
            + control.var(ddof=1) / len(control)
        )
    )
    alpha = 1.0 - confidence
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    return EffectEstimate(
        segment=segment,
        treatment_rows=int(len(treated)),
        control_rows=int(len(control)),
        estimate=estimate,
        standard_error=standard_error,
        ci_low=float(estimate - critical * standard_error),
        ci_high=float(estimate + critical * standard_error),
    )


def segment_effect_table(
    frame: pd.DataFrame,
    segment_col: str = "segment",
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    overall = effect_estimate(
        frame[outcome_col].to_numpy(),
        frame[treatment_col].to_numpy(),
        segment="overall",
    )
    rows.append(asdict(overall))
    for segment, group in frame.groupby(segment_col, observed=True):
        estimate = effect_estimate(
            group[outcome_col].to_numpy(),
            group[treatment_col].to_numpy(),
            segment=str(segment),
        )
        rows.append(asdict(estimate))
    return pd.DataFrame(rows)


def bootstrap_effect_distribution(
    outcome: np.ndarray,
    treatment: np.ndarray,
    rounds: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    if rounds < 100:
        raise ValueError("Use at least 100 bootstrap rounds")
    rng = np.random.default_rng(seed)
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    treated_index = np.where(treatment == 1)[0]
    control_index = np.where(treatment == 0)[0]
    draws = np.empty(rounds, dtype=float)
    for index in range(rounds):
        treated_sample = rng.choice(treated_index, size=len(treated_index), replace=True)
        control_sample = rng.choice(control_index, size=len(control_index), replace=True)
        draws[index] = float(
            outcome[treated_sample].mean() - outcome[control_sample].mean()
        )
    return draws


def randomization_inference_pvalue(
    outcome: np.ndarray,
    treatment: np.ndarray,
    permutations: int = 2000,
    seed: int = 42,
) -> float:
    if permutations < 100:
        raise ValueError("Use at least 100 permutations")
    rng = np.random.default_rng(seed)
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    observed = abs(effect_estimate(outcome, treatment, "observed").estimate)
    more_extreme = 0
    for _ in range(permutations):
        shuffled = rng.permutation(treatment)
        candidate = abs(effect_estimate(outcome, shuffled, "permuted").estimate)
        if candidate >= observed:
            more_extreme += 1
    return float((more_extreme + 1) / (permutations + 1))


def bootstrap_summary(draws: np.ndarray) -> dict[str, float]:
    draws = np.asarray(draws, dtype=float)
    return {
        "mean": float(draws.mean()),
        "std": float(draws.std(ddof=1)),
        "ci_low_95": float(np.quantile(draws, 0.025)),
        "ci_high_95": float(np.quantile(draws, 0.975)),
        "probability_positive": float((draws > 0.0).mean()),
    }


def power_curve(
    outcome_sd: float,
    total_sample_sizes: Iterable[int],
    alpha: float = 0.05,
    target_effect: float = 2.5,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    for total_rows in total_sample_sizes:
        per_arm = max(int(total_rows) / 2.0, 2.0)
        standard_error = float(outcome_sd * np.sqrt(2.0 / per_arm))
        noncentrality = float(target_effect / standard_error)
        achieved_power = float(
            1.0
            - norm.cdf(critical - noncentrality)
            + norm.cdf(-critical - noncentrality)
        )
        minimum_detectable_effect = float(
            (critical + norm.ppf(0.80)) * standard_error
        )
        rows.append(
            {
                "total_rows": int(total_rows),
                "target_effect": float(target_effect),
                "achieved_power": achieved_power,
                "mde_at_80pct_power": minimum_detectable_effect,
            }
        )
    return pd.DataFrame(rows)


def guardrail_sensitivity_table(
    guardrail_estimate: float,
    guardrail_ci_low: float,
    effect_ci_low: float,
    margins: Iterable[float] = (-0.05, -0.10, -0.20, -0.30, -0.50),
) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for margin in margins:
        guardrail_pass = bool(guardrail_ci_low > margin)
        primary_pass = bool(effect_ci_low > 0.0)
        decision = "ship" if guardrail_pass and primary_pass else "hold"
        rows.append(
            {
                "non_inferiority_margin": float(margin),
                "guardrail_estimate": float(guardrail_estimate),
                "guardrail_ci_low": float(guardrail_ci_low),
                "primary_ci_low": float(effect_ci_low),
                "guardrail_pass": guardrail_pass,
                "primary_pass": primary_pass,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics(
    frame: pd.DataFrame,
    adjusted_outcome: np.ndarray,
    seed: int = 42,
) -> dict[str, object]:
    numeric_balance = numeric_balance_table(
        frame,
        treatment_col="treatment",
        numeric_columns=["pre_metric"],
    )
    categorical_balance = categorical_balance_table(
        frame,
        treatment_col="treatment",
        categorical_column="segment",
    )
    segment_effects = segment_effect_table(frame)
    draws = bootstrap_effect_distribution(
        adjusted_outcome,
        frame["treatment"].to_numpy(),
        rounds=2000,
        seed=seed,
    )
    permutation_pvalue = randomization_inference_pvalue(
        adjusted_outcome,
        frame["treatment"].to_numpy(),
        permutations=1500,
        seed=seed + 1,
    )
    return {
        "numeric_balance": numeric_balance.to_dict(orient="records"),
        "categorical_balance": categorical_balance.to_dict(orient="records"),
        "segment_effects": segment_effects.to_dict(orient="records"),
        "bootstrap": bootstrap_summary(draws),
        "randomization_inference_pvalue": permutation_pvalue,
    }
