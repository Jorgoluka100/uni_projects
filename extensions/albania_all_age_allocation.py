"""All-age demographic extension for Albania's healthcare allocation scenario.

Population counts are transcribed from INSTAT Regional Statistical Yearbook 2024,
Table 1.6, sourced to the Population and Housing Census 2023. This module models
a transparent demographic allocation scenario; it does not estimate clinical
need, individual health expenditure, or causal health outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TOTAL_POPULATION = 2_402_113
EXPECTED_AGE_TOTALS = {
    "age_0_14": 373_929,
    "age_15_64": 1_555_080,
    "age_65_plus": 473_104,
    "age_80_plus": 97_417,
}

# INSTAT Table 1.6: total population by prefecture and broad age group.
AGE_COUNTS = [
    ("Berat", 18_346, 90_381, 32_229),
    ("Dibër", 18_690, 70_620, 17_868),
    ("Durrës", 38_002, 147_874, 40_987),
    ("Elbasan", 35_563, 151_278, 45_739),
    ("Fier", 34_068, 149_555, 56_754),
    ("Gjirokastër", 6_908, 36_350, 16_755),
    ("Korçë", 23_663, 108_534, 40_894),
    ("Kukës", 12_990, 40_859, 8_149),
    ("Lezhë", 16_302, 63_513, 19_569),
    ("Shkodër", 23_569, 98_471, 32_439),
    ("Tiranë", 126_699, 508_596, 123_218),
    ("Vlorë", 19_129, 89_049, 38_503),
]
# Keep the three broad cohorts exactly as published by INSTAT. In particular,
# do not invent separate child, youth or working-age sub-bands when the source
# table only supplies 0–14 and 15–64; finer analysis needs finer source data.
AGE_COLUMNS = ["age_0_14", "age_15_64", "age_65_plus"]
# 80+ is a nested subset of 65+, not a fourth mutually exclusive cohort.
# It is retained for a focused older-age comparison and must never be added
# again when reconciling the all-age population total.
# Derived from the original 09B notebook age_80_84 + age_85_plus prefecture counts.
AGE_80_PLUS = [6_197, 3_696, 7_968, 8_821, 11_729, 3_859, 8_739, 2_051, 4_395, 6_934, 24_651, 8_377]
DEFAULT_AGE_WEIGHTS = {
    "age_0_14": 0.20,
    "age_15_64": 0.20,
    "age_65_plus": 0.20,
    "population": 0.40,
}
DEFAULT_EQUITY_FLOOR = 0.15


def build_age_profile() -> pd.DataFrame:
    """Return a validated prefecture-level population profile for all ages."""
    df = pd.DataFrame(AGE_COUNTS, columns=["prefecture", *AGE_COLUMNS])
    df["age_80_plus"] = AGE_80_PLUS
    # Reconstruct each prefecture's total from mutually exclusive age bands.
    df["population"] = df[AGE_COLUMNS].sum(axis=1)
    for col in AGE_COLUMNS:
        # National share answers: what proportion of this age group lives here?
        # Local percentage answers a different question: what fraction of this
        # prefecture belongs to the cohort. Keeping both avoids denominator confusion.
        df[f"{col}_share_national"] = df[col] / df[col].sum()
        df[f"{col}_pct_local"] = df[col] / df["population"]
    df["population_share"] = df["population"] / df["population"].sum()
    df["young_dependency_ratio"] = 100 * df["age_0_14"] / df["age_15_64"]
    df["old_dependency_ratio"] = 100 * df["age_65_plus"] / df["age_15_64"]
    df["total_dependency_ratio"] = (
        100 * (df["age_0_14"] + df["age_65_plus"]) / df["age_15_64"]
    )
    if df["prefecture"].duplicated().any():
        raise ValueError("Prefecture names must be unique.")
    if (df["age_80_plus"] > df["age_65_plus"]).any():
        raise ValueError("The 80+ cohort cannot exceed the 65+ cohort.")
    if int(df["population"].sum()) != TOTAL_POPULATION:
        raise ValueError("Prefecture population totals do not reconcile to Census 2023.")
    for col, expected in EXPECTED_AGE_TOTALS.items():
        if int(df[col].sum()) != expected:
            raise ValueError(f"{col} does not reconcile to INSTAT's published total.")
    return df


def calculate_all_age_need_shares(
    profile: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
    equity_floor: float = DEFAULT_EQUITY_FLOOR,
) -> pd.Series:
    """Build a transparent all-age need share with an equal-prefecture floor.

    The weights are scenario assumptions, not empirically estimated health needs.
    """
    profile = build_age_profile() if profile is None else profile.copy()
    weights = DEFAULT_AGE_WEIGHTS if weights is None else weights
    required = {
        "prefecture", "population", "population_share",
        "age_0_14_share_national", "age_15_64_share_national",
        "age_65_plus_share_national", "age_80_plus", *AGE_COLUMNS,
    }
    if profile.empty or not required.issubset(profile.columns):
        raise ValueError(f"Profile must be non-empty and include: {sorted(required)}")
    if profile["prefecture"].isna().any() or profile["prefecture"].duplicated().any():
        raise ValueError("Profile prefectures must be present and unique.")
    numeric = profile[list(required - {"prefecture"})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or np.any(numeric < 0):
        raise ValueError("Profile demographic values and shares must be finite and non-negative.")
    if (profile["age_80_plus"] > profile["age_65_plus"]).any():
        raise ValueError("The 80+ cohort cannot exceed the 65+ cohort.")
    if not np.isclose(profile["population_share"].sum(), 1.0):
        raise ValueError("Population shares must sum to one.")
    for col in AGE_COLUMNS:
        share_col = f"{col}_share_national"
        if not np.isclose(profile[share_col].sum(), 1.0):
            raise ValueError(f"{share_col} must sum to one.")
        if not np.allclose(profile["population"], profile[AGE_COLUMNS].sum(axis=1)):
            raise ValueError("Prefecture population must equal the sum of its age cohorts.")
    if not 0 <= equity_floor < 1:
        raise ValueError("equity_floor must be in [0, 1).")
    if set(weights) != {"population", *AGE_COLUMNS}:
        raise ValueError("Weights must include population and all three age cohorts.")
    values = np.asarray(list(weights.values()), dtype=float)
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("All weights must be non-negative and sum to 1.")
    # Weighted demographic signal: each input is a share of a national cohort.
    # This gives each cohort representation in the allocation scenario without
    # treating its raw population count as a proxy for disease burden.
    # Weights are policy choices for scenario testing, not clinical estimates.
    raw = (
        weights["population"] * profile["population_share"]
        + weights["age_0_14"] * profile["age_0_14_share_national"]
        + weights["age_15_64"] * profile["age_15_64_share_national"]
        + weights["age_65_plus"] * profile["age_65_plus_share_national"]
    )
    # Reserve a minimum equal share for every prefecture, then distribute
    # the remaining budget in proportion to the weighted demographic signal.
    shares = equity_floor / len(profile) + (1 - equity_floor) * raw
    shares.index = profile["prefecture"]
    return shares


def allocate_budget(
    budget_eur: float,
    need_shares: pd.Series,
) -> pd.DataFrame:
    """Allocate a non-negative euro budget according to validated need shares."""
    if not np.isfinite(budget_eur) or budget_eur < 0:
        raise ValueError("budget_eur must be finite and non-negative.")
    shares = need_shares.astype(float)
    if shares.empty or not np.isfinite(shares.to_numpy()).all() or (shares < 0).any() or not np.isclose(shares.sum(), 1.0):
        raise ValueError("need_shares must be non-empty, non-negative and sum to 1.")
    # Multiplying shares by the budget converts relative priorities into euros.
    # The share checks above ensure the returned allocations conserve the budget.
    return pd.DataFrame({
        "prefecture": shares.index,
        "need_share": shares.to_numpy(),
        "allocation_eur": shares.to_numpy() * budget_eur,
    })


def compare_allocation_scenarios(
    regional_budget_eur: float,
    equity_floor: float = DEFAULT_EQUITY_FLOOR,
) -> pd.DataFrame:
    """Compare population-only, all-age and older-age-focused scenarios."""
    if not np.isfinite(regional_budget_eur) or regional_budget_eur < 0:
        raise ValueError("regional_budget_eur must be finite and non-negative.")
    if not 0 <= equity_floor < 1:
        raise ValueError("equity_floor must be in [0, 1).")
    profile = build_age_profile().set_index("prefecture")
    # Comparator 1: population alone, with the same equity floor as other cases.
    population_only = (
        equity_floor / len(profile)
        + (1 - equity_floor) * profile["population_share"]
    )
    all_age = calculate_all_age_need_shares(
        profile.reset_index(), equity_floor=equity_floor
    )
    # Comparator 3: deliberately older-age-weighted; this is a scenario,
    # not a claim that older residents are the only group with healthcare needs.
    # Showing it beside the all-age model makes the demographic trade-off visible.
    older_focused = (
        equity_floor / len(profile)
        + (1 - equity_floor) * (
            0.50 * profile["population_share"]
            + 0.30 * profile["age_65_plus_share_national"]
            + 0.20 * (profile["age_80_plus"] / profile["age_80_plus"].sum())
        )
    )
    out = pd.DataFrame({
        "population_only_share": population_only,
        "all_age_share": all_age,
        "older_age_focused_share": older_focused,
    })
    for col in list(out.columns):
        out[col.replace("_share", "_allocation_eur")] = out[col] * regional_budget_eur
    out.index.name = "prefecture"
    return out.reset_index()


def run_weight_sensitivity(
    regional_budget_eur: float,
    n_sims: int = 1_000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Quantify scenario sensitivity to all-age weights and equity floor."""
    if not np.isfinite(regional_budget_eur) or regional_budget_eur < 0:
        raise ValueError("regional_budget_eur must be finite and non-negative.")
    if not isinstance(n_sims, int) or n_sims < 1:
        raise ValueError("n_sims must be a positive integer.")
    profile = build_age_profile()
    base = np.array([0.40, 0.20, 0.20, 0.20])
    # Dirichlet draws vary the weights around the baseline; larger concentration
    # keeps draws nearer the baseline while still exploring plausible alternatives.
    alpha = base * 60
    rng = np.random.default_rng(seed)
    simulated = []
    for _ in range(n_sims):
        w = rng.dirichlet(alpha)
        # Vary the minimum regional equity allocation alongside the age weights.
        floor = rng.uniform(0.10, 0.20)
        weights = dict(zip(
            ["population", "age_0_14", "age_15_64", "age_65_plus"], w
        ))
        shares = calculate_all_age_need_shares(
            profile, weights=weights, equity_floor=floor
        )
        simulated.append(shares.to_numpy() * regional_budget_eur)
    # Summarise the distribution across simulations, not a confidence interval
    # for real-world spending or clinical outcomes.
    values = np.asarray(simulated)
    return pd.DataFrame({
        "prefecture": profile["prefecture"],
        "p10_eur": np.quantile(values, 0.10, axis=0),
        "median_eur": np.quantile(values, 0.50, axis=0),
        "p90_eur": np.quantile(values, 0.90, axis=0),
    })


def main() -> None:
    """Print the all-age profile, scenario comparison and sensitivity summary."""
    profile = build_age_profile()
    regional_budget_eur = 40_000_000
    comparison = compare_allocation_scenarios(regional_budget_eur)
    sensitivity = run_weight_sensitivity(regional_budget_eur)
    print("CENSUS 2023 — ALL-AGE PROFILE BY PREFECTURE")
    print(profile[[
        "prefecture", "population", "age_0_14", "age_15_64", "age_65_plus",
        "young_dependency_ratio", "old_dependency_ratio", "total_dependency_ratio",
    ]].to_string(index=False))
    print("\nREGIONAL ALLOCATION SCENARIOS — €40m illustrative regional pool")
    print(comparison.to_string(index=False))
    print("\nWEIGHT SENSITIVITY — 10th / 50th / 90th percentile")
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
