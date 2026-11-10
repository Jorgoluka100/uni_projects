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


def _validate_experiment_arrays(outcome, treatment) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment)
    if y.ndim != 1 or t.ndim != 1 or y.size == 0 or y.shape != t.shape:
        raise ValueError("outcome and treatment must be non-empty aligned 1D arrays")
    if not np.isfinite(y).all():
        raise ValueError("outcome must contain only finite values")
    if not np.isin(t, [0, 1]).all():
        raise ValueError("treatment must contain only binary 0/1 assignments")
    t = t.astype(int)
    if np.count_nonzero(t == 1) < 2 or np.count_nonzero(t == 0) < 2:
        raise ValueError("both experiment arms need at least two observations")
    return y, t


def standardized_mean_difference(
    treatment_values: np.ndarray,
    control_values: np.ndarray,
) -> float:
    treated = np.asarray(treatment_values, dtype=float)
    control = np.asarray(control_values, dtype=float)
    if treated.ndim != 1 or control.ndim != 1 or min(treated.size, control.size) < 2:
        raise ValueError("each experiment arm needs at least two numeric observations")
    if not np.isfinite(treated).all() or not np.isfinite(control).all():
        raise ValueError("balance values must be finite")
    treatment_variance = float(np.var(treated, ddof=1))
    control_variance = float(np.var(control, ddof=1))
    pooled_sd = float(np.sqrt((treatment_variance + control_variance) / 2.0))
    difference = float(treated.mean() - control.mean())
    if pooled_sd == 0.0:
        # Two constant but different arms are maximally imbalanced, not perfectly balanced.
        return 0.0 if np.isclose(difference, 0.0) else float(np.copysign(np.inf, difference))
    return float(difference / pooled_sd)


def numeric_balance_table(
    frame: pd.DataFrame,
    treatment_col: str,
    numeric_columns: Iterable[str],
    threshold: float = 0.10,
) -> pd.DataFrame:
    columns = list(numeric_columns)
    required = {treatment_col, *columns}
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing:
        raise ValueError(f"balance frame must be non-empty and include {sorted(required)}; missing={missing}")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("balance threshold must be finite and positive")
    y, assignment = _validate_experiment_arrays(
        np.zeros(len(frame), dtype=float), frame[treatment_col].to_numpy()
    )
    treatment_mask = assignment == 1
    rows: list[dict[str, float | str | bool]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain only finite numeric values")
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
    required = {treatment_col, categorical_column}
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing:
        raise ValueError(f"categorical balance frame must be non-empty and include {sorted(required)}; missing={missing}")
    _, assignment = _validate_experiment_arrays(
        np.zeros(len(frame), dtype=float), frame[treatment_col].to_numpy()
    )
    work = frame[[treatment_col, categorical_column]].copy()
    work[treatment_col] = assignment
    work[categorical_column] = work[categorical_column].astype("string").fillna("<MISSING>")
    counts = (
        work.groupby([treatment_col, categorical_column], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    totals = counts.groupby(treatment_col)["rows"].transform("sum")
    counts["share"] = counts["rows"] / totals
    pivot = counts.pivot(index=categorical_column, columns=treatment_col, values="share").fillna(0.0)
    for expected in (0, 1):
        if expected not in pivot.columns:
            pivot[expected] = 0.0
    output = pivot.rename(columns={0: "control_share", 1: "treatment_share"}).reset_index()
    output["absolute_share_gap"] = (output["treatment_share"] - output["control_share"]).abs()
    return output.sort_values("absolute_share_gap", ascending=False)


def effect_estimate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    segment: str,
    confidence: float = 0.95,
) -> EffectEstimate:
    y, assignment = _validate_experiment_arrays(outcome, treatment)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    treated = y[assignment == 1]
    control = y[assignment == 0]
    estimate = float(treated.mean() - control.mean())
    standard_error = float(np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)))
    alpha = 1.0 - confidence
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    return EffectEstimate(
        segment=str(segment),
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
    required = {segment_col, treatment_col, outcome_col}
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing or frame[list(required)].isna().any().any():
        raise ValueError(f"segment analysis requires non-empty complete columns: {sorted(required)}")
    rows: list[dict[str, float | str | int]] = []
    overall = effect_estimate(frame[outcome_col].to_numpy(), frame[treatment_col].to_numpy(), segment="overall")
    rows.append(asdict(overall))
    for segment, group in frame.groupby(segment_col, observed=True):
        estimate = effect_estimate(group[outcome_col].to_numpy(), group[treatment_col].to_numpy(), segment=str(segment))
        rows.append(asdict(estimate))
    return pd.DataFrame(rows)


def bootstrap_effect_distribution(
    outcome: np.ndarray,
    treatment: np.ndarray,
    rounds: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    y, assignment = _validate_experiment_arrays(outcome, treatment)
    if not isinstance(rounds, int) or rounds < 100:
        raise ValueError("Use at least 100 bootstrap rounds")
    rng = np.random.default_rng(seed)
    treated_index = np.where(assignment == 1)[0]
    control_index = np.where(assignment == 0)[0]
    draws = np.empty(rounds, dtype=float)
    for index in range(rounds):
        treated_sample = rng.choice(treated_index, size=len(treated_index), replace=True)
        control_sample = rng.choice(control_index, size=len(control_index), replace=True)
        draws[index] = float(y[treated_sample].mean() - y[control_sample].mean())
    return draws


def randomization_inference_pvalue(
    outcome: np.ndarray,
    treatment: np.ndarray,
    permutations: int = 2000,
    seed: int = 42,
) -> float:
    y, assignment = _validate_experiment_arrays(outcome, treatment)
    if not isinstance(permutations, int) or permutations < 100:
        raise ValueError("Use at least 100 permutations")
    observed = abs(effect_estimate(y, assignment, "observed").estimate)
    rng = np.random.default_rng(seed)
    more_extreme = 0
    for _ in range(permutations):
        shuffled = rng.permutation(assignment)
        candidate = abs(effect_estimate(y, shuffled, "permuted").estimate)
        if candidate >= observed:
            more_extreme += 1
    return float((more_extreme + 1) / (permutations + 1))


def bootstrap_summary(draws: np.ndarray) -> dict[str, float]:
    values = np.asarray(draws, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("draws must contain at least two finite bootstrap estimates")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci_low_95": float(np.quantile(values, 0.025)),
        "ci_high_95": float(np.quantile(values, 0.975)),
        "probability_positive": float((values > 0.0).mean()),
    }


def power_curve(
    outcome_sd: float,
    total_sample_sizes: Iterable[int],
    alpha: float = 0.05,
    target_effect: float = 2.5,
) -> pd.DataFrame:
    if not np.isfinite(outcome_sd) or outcome_sd <= 0:
        raise ValueError("outcome_sd must be finite and positive")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if not np.isfinite(target_effect) or target_effect < 0:
        raise ValueError("target_effect must be finite and non-negative")
    sizes = list(total_sample_sizes)
    if not sizes or any(not isinstance(n, (int, np.integer)) or n < 4 for n in sizes):
        raise ValueError("total sample sizes must be integers of at least four")
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    rows: list[dict[str, float | int]] = []
    for total_rows in sizes:
        treatment_n = int(total_rows) // 2
        control_n = int(total_rows) - treatment_n
        standard_error = float(outcome_sd * np.sqrt(1.0 / treatment_n + 1.0 / control_n))
        noncentrality = float(target_effect / standard_error)
        achieved_power = float(1.0 - norm.cdf(critical - noncentrality) + norm.cdf(-critical - noncentrality))
        minimum_detectable_effect = float((critical + norm.ppf(0.80)) * standard_error)
        rows.append({
            "total_rows": int(total_rows),
            "target_effect": float(target_effect),
            "achieved_power": achieved_power,
            "mde_at_80pct_power": minimum_detectable_effect,
        })
    return pd.DataFrame(rows)


def guardrail_sensitivity_table(
    guardrail_estimate: float,
    guardrail_ci_low: float,
    effect_ci_low: float,
    margins: Iterable[float] = (-0.05, -0.10, -0.20, -0.30, -0.50),
) -> pd.DataFrame:
    values = np.asarray([guardrail_estimate, guardrail_ci_low, effect_ci_low], dtype=float)
    margin_values = np.asarray(list(margins), dtype=float)
    if not np.isfinite(values).all() or margin_values.size == 0 or not np.isfinite(margin_values).all():
        raise ValueError("guardrail estimates and margins must be finite and margins non-empty")
    rows: list[dict[str, float | str | bool]] = []
    for margin in margin_values:
        guardrail_pass = bool(guardrail_ci_low > margin)
        primary_pass = bool(effect_ci_low > 0.0)
        rows.append({
            "non_inferiority_margin": float(margin),
            "guardrail_estimate": float(guardrail_estimate),
            "guardrail_ci_low": float(guardrail_ci_low),
            "primary_ci_low": float(effect_ci_low),
            "guardrail_pass": guardrail_pass,
            "primary_pass": primary_pass,
            "decision": "ship" if guardrail_pass and primary_pass else "hold",
        })
    return pd.DataFrame(rows)


def build_diagnostics(
    frame: pd.DataFrame,
    adjusted_outcome: np.ndarray,
    seed: int = 42,
) -> dict[str, object]:
    numeric_balance = numeric_balance_table(frame, treatment_col="treatment", numeric_columns=["pre_metric"])
    categorical_balance = categorical_balance_table(frame, treatment_col="treatment", categorical_column="segment")
    segment_effects = segment_effect_table(frame)
    draws = bootstrap_effect_distribution(adjusted_outcome, frame["treatment"].to_numpy(), rounds=2000, seed=seed)
    permutation_pvalue = randomization_inference_pvalue(adjusted_outcome, frame["treatment"].to_numpy(), permutations=1500, seed=seed + 1)
    return {
        "numeric_balance": numeric_balance.to_dict(orient="records"),
        "categorical_balance": categorical_balance.to_dict(orient="records"),
        "segment_effects": segment_effects.to_dict(orient="records"),
        "bootstrap": bootstrap_summary(draws),
        "randomization_inference_pvalue": permutation_pvalue,
    }
