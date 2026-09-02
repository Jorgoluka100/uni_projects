from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42


@dataclass
class HoldoutMetrics:
    mae: float
    rmse: float
    r2: float


def generate_weekly_data(weeks: int = 208, seed: int = SEED) -> pd.DataFrame:
    if weeks < 104:
        raise ValueError("Use at least two years of weekly history")
    rng = np.random.default_rng(seed)
    date = pd.date_range("2022-01-03", periods=weeks, freq="W-MON")
    trend = np.arange(weeks)
    season_sin = np.sin(2 * np.pi * trend / 52)
    season_cos = np.cos(2 * np.pi * trend / 52)

    search_spend = rng.gamma(shape=4.0, scale=4500.0, size=weeks)
    social_spend = rng.gamma(shape=3.0, scale=3500.0, size=weeks)
    display_spend = rng.gamma(shape=2.5, scale=2800.0, size=weeks)
    promotion = rng.binomial(1, 0.22, size=weeks)
    price_index = np.clip(rng.normal(1.0, 0.045, size=weeks), 0.88, 1.13)
    competitor_index = np.clip(rng.normal(1.0, 0.08, size=weeks), 0.75, 1.30)
    distribution = np.clip(0.78 + trend * 0.00055 + rng.normal(0, 0.018, size=weeks), 0.72, 0.95)

    search_response = 8500.0 * np.log1p(search_spend / 10000.0)
    social_response = 5000.0 * np.log1p(social_spend / 10000.0)
    display_response = 2300.0 * np.log1p(display_spend / 10000.0)
    promotion_effect = 11500.0 * promotion
    promo_social_interaction = 1800.0 * promotion * np.log1p(social_spend / 10000.0)
    price_effect = -62000.0 * (price_index - 1.0)
    competitor_effect = -17000.0 * (competitor_index - 1.0)
    distribution_effect = 52000.0 * (distribution - distribution.mean())
    seasonal_effect = 6500.0 * season_sin + 2800.0 * season_cos
    trend_effect = 42.0 * trend
    noise_scale = 5000.0 + 1200.0 * promotion
    noise = rng.normal(0, noise_scale)

    sales = (
        82000.0
        + search_response
        + social_response
        + display_response
        + promotion_effect
        + promo_social_interaction
        + price_effect
        + competitor_effect
        + distribution_effect
        + seasonal_effect
        + trend_effect
        + noise
    )

    return pd.DataFrame(
        {
            "week": date,
            "sales": sales,
            "search_spend": search_spend,
            "social_spend": social_spend,
            "display_spend": display_spend,
            "promotion": promotion,
            "price_index": price_index,
            "competitor_index": competitor_index,
            "distribution": distribution,
            "trend": trend,
            "season_sin": season_sin,
            "season_cos": season_cos,
        }
    )


def audit_data(df: pd.DataFrame) -> dict[str, Any]:
    required = {
        "week", "sales", "search_spend", "social_spend", "display_spend", "promotion",
        "price_index", "competitor_index", "distribution", "trend", "season_sin", "season_cos",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df["week"].duplicated().any():
        raise ValueError("Week must be unique")
    if df["week"].isna().any() or df["sales"].isna().any():
        raise ValueError("Critical columns contain missing values")
    for col in ["search_spend", "social_spend", "display_spend", "sales"]:
        if (df[col] < 0).any():
            raise ValueError(f"{col} contains negative values")
    return {
        "rows": int(len(df)),
        "start": str(df["week"].min().date()),
        "end": str(df["week"].max().date()),
        "mean_sales": float(df["sales"].mean()),
        "mean_search_spend": float(df["search_spend"].mean()),
        "mean_social_spend": float(df["social_spend"].mean()),
        "mean_display_spend": float(df["display_spend"].mean()),
        "promotion_rate": float(df["promotion"].mean()),
    }


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_search"] = np.log1p(out["search_spend"] / 10000.0)
    out["log_social"] = np.log1p(out["social_spend"] / 10000.0)
    out["log_display"] = np.log1p(out["display_spend"] / 10000.0)
    out["promo_x_social"] = out["promotion"] * out["log_social"]
    out["price_gap"] = out["price_index"] - 1.0
    out["competitor_gap"] = out["competitor_index"] - 1.0
    out["distribution_centered"] = out["distribution"] - out["distribution"].mean()
    return out


def split_time(df: pd.DataFrame, holdout_weeks: int = 26) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_weeks <= 0 or holdout_weeks >= len(df):
        raise ValueError("Invalid holdout size")
    train = df.iloc[:-holdout_weeks].copy()
    test = df.iloc[-holdout_weeks:].copy()
    if train["week"].max() >= test["week"].min():
        raise ValueError("Time split is not strictly ordered")
    return train, test


def design_columns() -> list[str]:
    return [
        "log_search",
        "log_social",
        "log_display",
        "promotion",
        "promo_x_social",
        "price_gap",
        "competitor_gap",
        "distribution_centered",
        "trend",
        "season_sin",
        "season_cos",
    ]


def fit_ols(train: pd.DataFrame):
    x = sm.add_constant(train[design_columns()], has_constant="add")
    model = sm.OLS(train["sales"], x)
    return model.fit(cov_type="HC3")


def coefficient_table(model) -> pd.DataFrame:
    interval = model.conf_int(alpha=0.05)
    return pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error_hc3": model.bse.values,
            "p_value": model.pvalues.values,
            "ci_low_95": interval.iloc[:, 0].values,
            "ci_high_95": interval.iloc[:, 1].values,
        }
    )


def variance_inflation_table(train: pd.DataFrame) -> pd.DataFrame:
    x = train[design_columns()].copy()
    matrix = sm.add_constant(x, has_constant="add")
    rows = []
    for i, column in enumerate(matrix.columns):
        if column == "const":
            continue
        rows.append({"term": column, "vif": float(variance_inflation_factor(matrix.values, i))})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def residual_diagnostics(model) -> dict[str, Any]:
    residuals = np.asarray(model.resid)
    fitted = np.asarray(model.fittedvalues)
    bp_stat, bp_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, model.model.exog)
    ljung = acorr_ljungbox(residuals, lags=[1, 4, 8], return_df=True)
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    return {
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std(ddof=1)),
        "correlation_fitted_abs_residual": float(np.corrcoef(fitted, np.abs(residuals))[0, 1]),
        "breusch_pagan": {
            "lm_stat": float(bp_stat),
            "lm_pvalue": float(bp_pvalue),
            "f_stat": float(f_stat),
            "f_pvalue": float(f_pvalue),
        },
        "jarque_bera": {"stat": float(jb_stat), "pvalue": float(jb_pvalue)},
        "ljung_box": {
            str(index): {
                "stat": float(row["lb_stat"]),
                "pvalue": float(row["lb_pvalue"]),
            }
            for index, row in ljung.iterrows()
        },
    }


def evaluate_holdout(model, test: pd.DataFrame) -> HoldoutMetrics:
    x_test = sm.add_constant(test[design_columns()], has_constant="add")
    prediction = np.asarray(model.predict(x_test))
    truth = test["sales"].to_numpy()
    return HoldoutMetrics(
        mae=float(mean_absolute_error(truth, prediction)),
        rmse=float(np.sqrt(mean_squared_error(truth, prediction))),
        r2=float(r2_score(truth, prediction)),
    )


def bootstrap_coefficients(train: pd.DataFrame, iterations: int = 500, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    terms = ["log_search", "log_social", "log_display", "promotion", "price_gap"]
    rows: list[dict[str, float | int]] = []
    for iteration in range(iterations):
        indices = rng.integers(0, len(train), size=len(train))
        sample = train.iloc[indices]
        x = sm.add_constant(sample[design_columns()], has_constant="add")
        fitted = sm.OLS(sample["sales"], x).fit()
        row: dict[str, float | int] = {"iteration": iteration}
        for term in terms:
            row[term] = float(fitted.params[term])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_summary(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in bootstrap.columns:
        if column == "iteration":
            continue
        values = bootstrap[column].to_numpy()
        rows.append(
            {
                "term": column,
                "bootstrap_mean": float(values.mean()),
                "bootstrap_std": float(values.std(ddof=1)),
                "p025": float(np.percentile(values, 2.5)),
                "p975": float(np.percentile(values, 97.5)),
                "positive_probability": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def channel_response_curve(model, reference: pd.Series, channel: str, spends: np.ndarray) -> pd.DataFrame:
    if channel not in {"search", "social", "display"}:
        raise ValueError("Unknown channel")
    rows = []
    for spend in spends:
        row = reference.copy()
        row[f"{channel}_spend"] = float(spend)
        row["log_search"] = np.log1p(row["search_spend"] / 10000.0)
        row["log_social"] = np.log1p(row["social_spend"] / 10000.0)
        row["log_display"] = np.log1p(row["display_spend"] / 10000.0)
        row["promo_x_social"] = row["promotion"] * row["log_social"]
        x = sm.add_constant(pd.DataFrame([row[design_columns()]]), has_constant="add")
        prediction = float(model.predict(x).iloc[0])
        rows.append({"channel": channel, "spend": float(spend), "predicted_sales": prediction})
    return pd.DataFrame(rows)


def scenario_table(model, test: pd.DataFrame) -> pd.DataFrame:
    reference = test.iloc[-1].copy()
    base_search = float(reference["search_spend"])
    base_social = float(reference["social_spend"])
    base_display = float(reference["display_spend"])
    scenarios = [
        ("current_mix", base_search, base_social, base_display),
        ("search_plus_20", base_search * 1.20, base_social, base_display),
        ("social_plus_20", base_search, base_social * 1.20, base_display),
        ("display_plus_20", base_search, base_social, base_display * 1.20),
        ("shift_10_display_to_search", base_search + 0.10 * base_display, base_social, base_display * 0.90),
        ("shift_10_display_to_social", base_search, base_social + 0.10 * base_display, base_display * 0.90),
    ]
    rows = []
    for name, search_spend, social_spend, display_spend in scenarios:
        row = reference.copy()
        row["search_spend"] = search_spend
        row["social_spend"] = social_spend
        row["display_spend"] = display_spend
        row["log_search"] = np.log1p(search_spend / 10000.0)
        row["log_social"] = np.log1p(social_spend / 10000.0)
        row["log_display"] = np.log1p(display_spend / 10000.0)
        row["promo_x_social"] = row["promotion"] * row["log_social"]
        x = sm.add_constant(pd.DataFrame([row[design_columns()]]), has_constant="add")
        predicted_sales = float(model.predict(x).iloc[0])
        rows.append(
            {
                "scenario": name,
                "search_spend": float(search_spend),
                "social_spend": float(social_spend),
                "display_spend": float(display_spend),
                "total_media_spend": float(search_spend + social_spend + display_spend),
                "predicted_sales": predicted_sales,
            }
        )
    scenarios_df = pd.DataFrame(rows)
    baseline = float(scenarios_df.loc[scenarios_df["scenario"] == "current_mix", "predicted_sales"].iloc[0])
    scenarios_df["predicted_sales_uplift_vs_current"] = scenarios_df["predicted_sales"] - baseline
    return scenarios_df.sort_values("predicted_sales", ascending=False)


def decision_summary(coefficients: pd.DataFrame, scenarios: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, Any]:
    significant = coefficients[
        (coefficients["p_value"] < 0.05)
        & (coefficients["term"] != "const")
    ][["term", "coefficient", "p_value", "ci_low_95", "ci_high_95"]]
    best = scenarios.iloc[0]
    stable_terms = bootstrap[
        (bootstrap["positive_probability"] >= 0.95)
        | (bootstrap["positive_probability"] <= 0.05)
    ]
    return {
        "statistically_clear_terms": significant.to_dict(orient="records"),
        "bootstrap_stable_terms": stable_terms.to_dict(orient="records"),
        "best_modelled_scenario": best.to_dict(),
        "decision_caveat": "Modelled associations and synthetic ground truth do not replace causal incrementality evidence.",
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    raw = generate_weekly_data()
    audit = audit_data(raw)
    data = add_model_features(raw)
    train, test = split_time(data, holdout_weeks=26)
    model = fit_ols(train)

    coefficients = coefficient_table(model)
    vif = variance_inflation_table(train)
    diagnostics = residual_diagnostics(model)
    holdout = evaluate_holdout(model, test)
    bootstrap_draws = bootstrap_coefficients(train, iterations=500)
    bootstrap = bootstrap_summary(bootstrap_draws)
    scenarios = scenario_table(model, test)

    reference = test.iloc[-1].copy()
    curves = pd.concat(
        [
            channel_response_curve(model, reference, "search", np.linspace(0, 50000, 40)),
            channel_response_curve(model, reference, "social", np.linspace(0, 40000, 40)),
            channel_response_curve(model, reference, "display", np.linspace(0, 30000, 40)),
        ],
        ignore_index=True,
    )

    coefficients.to_csv(RESULTS / "coefficients.csv", index=False)
    vif.to_csv(RESULTS / "vif.csv", index=False)
    bootstrap.to_csv(RESULTS / "bootstrap_summary.csv", index=False)
    scenarios.to_csv(RESULTS / "budget_scenarios.csv", index=False)
    curves.to_csv(RESULTS / "response_curves.csv", index=False)

    payload = {
        "dataset_audit": audit,
        "model": {
            "r_squared_train": float(model.rsquared),
            "adjusted_r_squared_train": float(model.rsquared_adj),
            "aic": float(model.aic),
            "bic": float(model.bic),
            "covariance_type": str(model.cov_type),
        },
        "holdout": asdict(holdout),
        "diagnostics": diagnostics,
        "decision_summary": decision_summary(coefficients, scenarios, bootstrap),
        "limitations": [
            "Synthetic data with known relationships; useful for validation but not market evidence.",
            "Regression association is not automatically causal attribution.",
            "Real marketing mix requires carryover/adstock, endogeneity controls, incrementality evidence and longer history.",
        ],
    }
    save_json(RESULTS / "metrics.json", payload)
    save_json(RESULTS / "diagnostics.json", diagnostics)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
