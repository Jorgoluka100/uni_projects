"""Linear Regression — Building Energy Efficiency Decision Model.

This file intentionally reads like a notebook/script rather than a framework full of
helper functions. The goal is to make the actual junior/graduate data-science work
visible: data loading, validation, EDA, modelling, diagnostics, uncertainty and a
usable decision layer.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Building heating-load regression portfolio project")
parser.add_argument("--output-dir", default="results", help="Directory for metrics, tables and plots")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-size", type=float, default=0.20)
parser.add_argument("--bootstrap", type=int, default=250, help="Bootstrap refits for uncertainty summary")
parser.add_argument("--no-plots", action="store_true")
args = parser.parse_args()

RNG = np.random.default_rng(args.seed)
PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / args.output_dir
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 88)
print("LINEAR REGRESSION — BUILDING ENERGY EFFICIENCY DECISION MODEL")
print("=" * 88)
print("Project:", PROJECT)
print("Output:", OUTPUT)
print("Seed:", args.seed)

# -----------------------------------------------------------------------------
# 1. DATA ACQUISITION AND PROVENANCE
# -----------------------------------------------------------------------------
print("\n[1/12] Loading UCI Energy Efficiency dataset (ID 242)...")
energy = fetch_ucirepo(id=242)
X_raw = energy.data.features.copy()
y_raw = energy.data.targets.copy()

print("Dataset name:", energy.metadata.get("name", "Energy Efficiency"))
print("UCI id:", energy.metadata.get("uci_id", 242))
print("Rows:", len(X_raw))
print("Feature columns:", list(X_raw.columns))
print("Target columns:", list(y_raw.columns))

feature_names = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
]

target_names = ["heating_load", "cooling_load"]

if X_raw.shape[1] != 8:
    raise ValueError(f"Expected 8 features from UCI Energy Efficiency, received {X_raw.shape[1]}")
if y_raw.shape[1] < 2:
    raise ValueError(f"Expected two targets from UCI Energy Efficiency, received {y_raw.shape[1]}")

X_raw.columns = feature_names
y_raw = y_raw.iloc[:, :2].copy()
y_raw.columns = target_names

df = pd.concat([X_raw, y_raw], axis=1)

print("\nPreview:")
print(df.head().to_string(index=False))

# -----------------------------------------------------------------------------
# 2. DATA QUALITY AUDIT
# -----------------------------------------------------------------------------
print("\n[2/12] Running schema and quality audit...")

expected_columns = feature_names + target_names
missing_columns = sorted(set(expected_columns) - set(df.columns))
unexpected_columns = sorted(set(df.columns) - set(expected_columns))
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")
if unexpected_columns:
    print("Unexpected columns:", unexpected_columns)

numeric_check = df[expected_columns].apply(pd.to_numeric, errors="coerce")
coercion_failures = numeric_check.isna().sum() - df[expected_columns].isna().sum()
if (coercion_failures > 0).any():
    raise ValueError(f"Non-numeric values detected: {coercion_failures[coercion_failures > 0].to_dict()}")

df = numeric_check.copy()

quality = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (100.0 * df.isna().mean()).round(3),
        "unique": df.nunique(dropna=False),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
    }
)
quality.to_csv(OUTPUT / "data_audit.csv")

print(quality.to_string())
print("Duplicate rows:", int(df.duplicated().sum()))

if df.isna().any().any():
    print("Missing values are present and will be handled by the preprocessing pipeline.")
else:
    print("Published UCI table contains no missing values in this load.")

if len(df) != 768:
    print(f"WARNING: UCI documentation describes 768 rows; current load contains {len(df)} rows.")

# Plausibility checks derived from the published schema.
if not df["relative_compactness"].between(0, 1.5).all():
    raise ValueError("Relative compactness outside a broad plausible range")
if not df["glazing_area"].between(0, 1).all():
    raise ValueError("Glazing area outside [0, 1]")
if (df["heating_load"] <= 0).any() or (df["cooling_load"] <= 0).any():
    raise ValueError("Energy loads should be positive")

# -----------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------------------------
print("\n[3/12] Exploratory analysis...")

summary = df.describe(include="all").T
summary.to_csv(OUTPUT / "descriptive_statistics.csv")
print(summary.round(3).to_string())

corr = df.corr(numeric_only=True)
corr.to_csv(OUTPUT / "correlation_matrix.csv")

heating_quantiles = df["heating_load"].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
print("\nHeating-load quantiles:")
print(heating_quantiles.round(3).to_string())

orientation_summary = (
    df.groupby("orientation", as_index=False)
    .agg(
        rows=("heating_load", "size"),
        heating_mean=("heating_load", "mean"),
        heating_median=("heating_load", "median"),
        cooling_mean=("cooling_load", "mean"),
    )
    .sort_values("orientation")
)
orientation_summary.to_csv(OUTPUT / "orientation_summary.csv", index=False)
print("\nLoad by orientation:")
print(orientation_summary.round(3).to_string(index=False))

glazing_summary = (
    df.groupby("glazing_area", as_index=False)
    .agg(
        rows=("heating_load", "size"),
        heating_mean=("heating_load", "mean"),
        heating_std=("heating_load", "std"),
        cooling_mean=("cooling_load", "mean"),
    )
    .sort_values("glazing_area")
)
glazing_summary.to_csv(OUTPUT / "glazing_summary.csv", index=False)

height_summary = (
    df.groupby("overall_height", as_index=False)
    .agg(
        rows=("heating_load", "size"),
        heating_mean=("heating_load", "mean"),
        cooling_mean=("cooling_load", "mean"),
    )
    .sort_values("overall_height")
)
height_summary.to_csv(OUTPUT / "height_summary.csv", index=False)

if not args.no_plots:
    plt.figure(figsize=(8, 5))
    plt.hist(df["heating_load"], bins=28, alpha=0.80)
    plt.axvline(df["heating_load"].median(), linestyle="--", linewidth=2, label=f"median={df['heating_load'].median():.2f}")
    plt.axvline(df["heating_load"].quantile(0.75), linestyle=":", linewidth=2, label=f"75th pct={df['heating_load'].quantile(0.75):.2f}")
    plt.title("Heating-load distribution")
    plt.xlabel("Heating Load")
    plt.ylabel("Buildings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "01_heating_load_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["heating_load"], df["cooling_load"], alpha=0.45, s=25)
    plt.xlabel("Heating Load")
    plt.ylabel("Cooling Load")
    plt.title("Heating vs cooling load")
    plt.tight_layout()
    plt.savefig(OUTPUT / "02_heating_vs_cooling.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 8))
    image = plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(image, label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=70, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT / "03_correlation_matrix.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["relative_compactness"], df["heating_load"], alpha=0.45, s=25)
    plt.xlabel("Relative compactness")
    plt.ylabel("Heating Load")
    plt.title("Heating load vs relative compactness")
    plt.tight_layout()
    plt.savefig(OUTPUT / "04_compactness_vs_heating.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["surface_area"], df["heating_load"], alpha=0.45, s=25)
    plt.xlabel("Surface area")
    plt.ylabel("Heating Load")
    plt.title("Heating load vs surface area")
    plt.tight_layout()
    plt.savefig(OUTPUT / "05_surface_area_vs_heating.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["glazing_area"], df["heating_load"], alpha=0.45, s=25)
    plt.xlabel("Glazing area")
    plt.ylabel("Heating Load")
    plt.title("Heating load vs glazing area")
    plt.tight_layout()
    plt.savefig(OUTPUT / "06_glazing_vs_heating.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(orientation_summary["orientation"].astype(str), orientation_summary["heating_mean"])
    plt.xlabel("Orientation")
    plt.ylabel("Mean Heating Load")
    plt.title("Average heating load by orientation")
    plt.tight_layout()
    plt.savefig(OUTPUT / "07_orientation_heating.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(glazing_summary["glazing_area"].astype(str), glazing_summary["heating_mean"])
    plt.xlabel("Glazing area")
    plt.ylabel("Mean Heating Load")
    plt.title("Average heating load by glazing area")
    plt.tight_layout()
    plt.savefig(OUTPUT / "08_glazing_group_heating.png", dpi=160)
    plt.close()

# -----------------------------------------------------------------------------
# 4. TRAIN / TEST DESIGN
# -----------------------------------------------------------------------------
print("\n[4/12] Creating holdout split and preprocessing...")

feature_columns = feature_names
categorical_columns = ["orientation", "glazing_area_distribution"]
numeric_columns = [c for c in feature_columns if c not in categorical_columns]

target = "heating_load"
X = df[feature_columns].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=args.test_size,
    random_state=args.seed,
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Train target mean:", round(float(y_train.mean()), 3))
print("Test target mean:", round(float(y_test.mean()), 3))

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ],
    remainder="drop",
)

# -----------------------------------------------------------------------------
# 5. BASELINE + ORDINARY LINEAR REGRESSION
# -----------------------------------------------------------------------------
print("\n[5/12] Training baseline and ordinary LinearRegression...")

baseline = DummyRegressor(strategy="median")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

linear_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ]
)
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# -----------------------------------------------------------------------------
# 6. REGULARISED AND NON-LINEAR ALTERNATIVES
# -----------------------------------------------------------------------------
print("\n[6/12] Training Ridge, Lasso and polynomial alternatives...")

ridge_alphas = np.logspace(-4, 4, 80)
ridge_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RidgeCV(alphas=ridge_alphas)),
    ]
)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

lasso_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LassoCV(alphas=np.logspace(-4, 1, 80), cv=5, random_state=args.seed, max_iter=100_000)),
    ]
)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)

poly_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric_columns,
        ),
        ("categorical", categorical_pipeline, categorical_columns),
    ],
    remainder="drop",
)

poly_model = Pipeline(
    steps=[
        ("preprocess", poly_preprocessor),
        ("model", RidgeCV(alphas=ridge_alphas)),
    ]
)
poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

# -----------------------------------------------------------------------------
# 7. HOLDOUT METRICS
# -----------------------------------------------------------------------------
print("\n[7/12] Comparing holdout performance...")

prediction_map = {
    "dummy_median": baseline_pred,
    "linear_regression": linear_pred,
    "ridge": ridge_pred,
    "lasso": lasso_pred,
    "polynomial_ridge": poly_pred,
}

metric_rows = []
for model_name, pred in prediction_map.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred)
    metric_rows.append(
        {
            "model": model_name,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
        }
    )

metrics_df = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
print(metrics_df.round(4).to_string(index=False))
metrics_df.to_csv(OUTPUT / "model_comparison.csv", index=False)

best_model_name = str(metrics_df.iloc[0]["model"])
model_lookup = {
    "dummy_median": baseline,
    "linear_regression": linear_model,
    "ridge": ridge_model,
    "lasso": lasso_model,
    "polynomial_ridge": poly_model,
}
best_model = model_lookup[best_model_name]
best_pred = prediction_map[best_model_name]

linear_metrics = metrics_df.loc[metrics_df["model"] == "linear_regression"].iloc[0]
baseline_metrics = metrics_df.loc[metrics_df["model"] == "dummy_median"].iloc[0]

improvement_vs_baseline = 100.0 * (baseline_metrics["rmse"] - linear_metrics["rmse"]) / baseline_metrics["rmse"]
print(f"\nLinear Regression RMSE improvement vs median baseline: {improvement_vs_baseline:.2f}%")
print("Best holdout model:", best_model_name)

if not args.no_plots:
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["model"], metrics_df["rmse"])
    plt.ylabel("RMSE")
    plt.title("Model comparison — lower is better")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT / "09_model_rmse_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, linear_pred, alpha=0.65, s=28)
    lower = min(float(y_test.min()), float(linear_pred.min()))
    upper = max(float(y_test.max()), float(linear_pred.max()))
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual Heating Load")
    plt.ylabel("Predicted Heating Load")
    plt.title("Ordinary Linear Regression — actual vs predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT / "10_linear_actual_vs_predicted.png", dpi=160)
    plt.close()

# -----------------------------------------------------------------------------
# 8. CROSS-VALIDATION AND LEARNING CURVE
# -----------------------------------------------------------------------------
print("\n[8/12] Cross-validation and learning-curve diagnostics...")

cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)
cv_rows = []
for model_name, model in {
    "linear_regression": linear_model,
    "ridge": ridge_model,
    "polynomial_ridge": poly_model,
}.items():
    neg_rmse = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=None)
    cv_rows.append(
        {
            "model": model_name,
            "cv_rmse_mean": float(-neg_rmse.mean()),
            "cv_rmse_std": float(neg_rmse.std()),
            "fold_rmse": [float(-x) for x in neg_rmse],
        }
    )

cv_df = pd.DataFrame(
    [{k: v for k, v in row.items() if k != "fold_rmse"} for row in cv_rows]
).sort_values("cv_rmse_mean")
cv_df.to_csv(OUTPUT / "cross_validation.csv", index=False)
print(cv_df.round(4).to_string(index=False))

train_sizes, train_scores, valid_scores = learning_curve(
    linear_model,
    X_train,
    y_train,
    train_sizes=np.linspace(0.20, 1.0, 6),
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=None,
)

learning_df = pd.DataFrame(
    {
        "train_rows": train_sizes,
        "train_rmse": -train_scores.mean(axis=1),
        "validation_rmse": -valid_scores.mean(axis=1),
        "validation_rmse_std": valid_scores.std(axis=1),
    }
)
learning_df.to_csv(OUTPUT / "learning_curve.csv", index=False)
print("\nLinear Regression learning curve:")
print(learning_df.round(4).to_string(index=False))

if not args.no_plots:
    plt.figure(figsize=(8, 5))
    plt.plot(learning_df["train_rows"], learning_df["train_rmse"], marker="o", label="train RMSE")
    plt.plot(learning_df["train_rows"], learning_df["validation_rmse"], marker="o", label="validation RMSE")
    plt.xlabel("Training rows")
    plt.ylabel("RMSE")
    plt.title("Linear Regression learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "11_linear_learning_curve.png", dpi=160)
    plt.close()

# -----------------------------------------------------------------------------
# 9. RESIDUAL AND ERROR ANALYSIS
# -----------------------------------------------------------------------------
print("\n[9/12] Residual diagnostics and error slices...")

predictions = X_test.reset_index(drop=False).rename(columns={"index": "source_index"})
predictions["actual_heating_load"] = y_test.reset_index(drop=True)
predictions["linear_prediction"] = linear_pred
predictions["best_prediction"] = best_pred
predictions["linear_residual"] = predictions["actual_heating_load"] - predictions["linear_prediction"]
predictions["best_residual"] = predictions["actual_heating_load"] - predictions["best_prediction"]
predictions["linear_abs_error"] = predictions["linear_residual"].abs()
predictions["best_abs_error"] = predictions["best_residual"].abs()
predictions.to_csv(OUTPUT / "predictions.csv", index=False)

print("Linear residual mean:", round(float(predictions["linear_residual"].mean()), 4))
print("Linear residual std:", round(float(predictions["linear_residual"].std()), 4))
print("Linear worst absolute error:", round(float(predictions["linear_abs_error"].max()), 4))

worst = predictions.nlargest(12, "linear_abs_error")
worst.to_csv(OUTPUT / "worst_linear_errors.csv", index=False)
print("\nWorst Linear Regression cases:")
print(
    worst[
        [
            "source_index",
            "actual_heating_load",
            "linear_prediction",
            "linear_abs_error",
            "relative_compactness",
            "surface_area",
            "overall_height",
            "glazing_area",
            "orientation",
        ]
    ].round(3).to_string(index=False)
)

orientation_error = (
    predictions.groupby("orientation", as_index=False)
    .agg(
        rows=("linear_abs_error", "size"),
        linear_mae=("linear_abs_error", "mean"),
        best_mae=("best_abs_error", "mean"),
        residual_bias=("linear_residual", "mean"),
    )
    .sort_values("linear_mae", ascending=False)
)
orientation_error.to_csv(OUTPUT / "error_by_orientation.csv", index=False)

predictions["glazing_band"] = pd.cut(
    predictions["glazing_area"],
    bins=[-np.inf, 0.0, 0.15, 0.30, np.inf],
    labels=["none", "low", "medium", "high"],
)
glazing_error = (
    predictions.groupby("glazing_band", observed=False, as_index=False)
    .agg(
        rows=("linear_abs_error", "size"),
        linear_mae=("linear_abs_error", "mean"),
        best_mae=("best_abs_error", "mean"),
        residual_bias=("linear_residual", "mean"),
    )
)
glazing_error.to_csv(OUTPUT / "error_by_glazing_band.csv", index=False)

predictions["compactness_band"] = pd.qcut(
    predictions["relative_compactness"],
    q=4,
    duplicates="drop",
)
compactness_error = (
    predictions.groupby("compactness_band", observed=False, as_index=False)
    .agg(
        rows=("linear_abs_error", "size"),
        linear_mae=("linear_abs_error", "mean"),
        best_mae=("best_abs_error", "mean"),
        residual_bias=("linear_residual", "mean"),
    )
)
compactness_error.to_csv(OUTPUT / "error_by_compactness.csv", index=False)

print("\nLinear error by orientation:")
print(orientation_error.round(4).to_string(index=False))
print("\nLinear error by glazing band:")
print(glazing_error.round(4).to_string(index=False))

if not args.no_plots:
    plt.figure(figsize=(8, 5))
    plt.scatter(predictions["linear_prediction"], predictions["linear_residual"], alpha=0.60, s=26)
    plt.axhline(0, linestyle="--", linewidth=1.5)
    plt.xlabel("Linear Regression prediction")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residuals vs predicted values")
    plt.tight_layout()
    plt.savefig(OUTPUT / "12_linear_residuals_vs_prediction.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(predictions["linear_residual"], bins=25, alpha=0.82)
    plt.axvline(0, linestyle="--", linewidth=1.5)
    plt.xlabel("Residual")
    plt.ylabel("Rows")
    plt.title("Linear Regression residual distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT / "13_linear_residual_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(orientation_error["orientation"].astype(str), orientation_error["linear_mae"])
    plt.xlabel("Orientation")
    plt.ylabel("Linear Regression MAE")
    plt.title("Error slice by orientation")
    plt.tight_layout()
    plt.savefig(OUTPUT / "14_error_by_orientation.png", dpi=160)
    plt.close()

# -----------------------------------------------------------------------------
# 10. COEFFICIENT INTERPRETATION
# -----------------------------------------------------------------------------
print("\n[10/12] Inspecting Linear Regression coefficients...")

preprocess_fitted = linear_model.named_steps["preprocess"]
model_fitted = linear_model.named_steps["model"]

numeric_names = numeric_columns
categorical_encoder = preprocess_fitted.named_transformers_["categorical"].named_steps["onehot"]
categorical_names = categorical_encoder.get_feature_names_out(categorical_columns).tolist()
transformed_feature_names = numeric_names + categorical_names

if len(transformed_feature_names) != len(model_fitted.coef_):
    raise RuntimeError("Transformed feature names do not align with LinearRegression coefficients")

coefficients = pd.DataFrame(
    {
        "feature": transformed_feature_names,
        "coefficient": model_fitted.coef_.astype(float),
    }
)
coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
coefficients = coefficients.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
coefficients.to_csv(OUTPUT / "coefficients.csv", index=False)

print("Linear intercept:", round(float(model_fitted.intercept_), 4))
print("\nLargest absolute coefficients after preprocessing:")
print(coefficients.head(20).round(4).to_string(index=False))

if not args.no_plots:
    coef_plot = coefficients.head(15).sort_values("coefficient")
    plt.figure(figsize=(9, 6))
    plt.barh(coef_plot["feature"], coef_plot["coefficient"])
    plt.xlabel("Coefficient")
    plt.title("Linear Regression coefficient magnitude")
    plt.tight_layout()
    plt.savefig(OUTPUT / "15_linear_coefficients.png", dpi=160)
    plt.close()

print(
    "\nInterpretation warning: coefficients describe conditional associations in this simulated dataset. "
    "Correlated geometric variables and preprocessing mean they should not be presented as causal effects."
)

# -----------------------------------------------------------------------------
# 11. BOOTSTRAP UNCERTAINTY
# -----------------------------------------------------------------------------
print("\n[11/12] Bootstrap uncertainty for Linear Regression...")

bootstrap_rmses = []
bootstrap_maes = []
bootstrap_prediction_matrix = []

bootstrap_iterations = max(25, int(args.bootstrap))
for iteration in range(bootstrap_iterations):
    sample_positions = RNG.integers(0, len(X_train), size=len(X_train))
    X_boot = X_train.iloc[sample_positions]
    y_boot = y_train.iloc[sample_positions]

    boot_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    boot_model.fit(X_boot, y_boot)
    boot_pred = boot_model.predict(X_test)

    bootstrap_rmses.append(math.sqrt(mean_squared_error(y_test, boot_pred)))
    bootstrap_maes.append(mean_absolute_error(y_test, boot_pred))
    bootstrap_prediction_matrix.append(boot_pred)

bootstrap_prediction_matrix = np.vstack(bootstrap_prediction_matrix)
predictions["linear_bootstrap_p05"] = np.quantile(bootstrap_prediction_matrix, 0.05, axis=0)
predictions["linear_bootstrap_p50"] = np.quantile(bootstrap_prediction_matrix, 0.50, axis=0)
predictions["linear_bootstrap_p95"] = np.quantile(bootstrap_prediction_matrix, 0.95, axis=0)
predictions["linear_bootstrap_width"] = predictions["linear_bootstrap_p95"] - predictions["linear_bootstrap_p05"]
predictions.to_csv(OUTPUT / "predictions.csv", index=False)

bootstrap_summary = {
    "iterations": bootstrap_iterations,
    "rmse_mean": float(np.mean(bootstrap_rmses)),
    "rmse_p05": float(np.quantile(bootstrap_rmses, 0.05)),
    "rmse_p95": float(np.quantile(bootstrap_rmses, 0.95)),
    "mae_mean": float(np.mean(bootstrap_maes)),
    "mae_p05": float(np.quantile(bootstrap_maes, 0.05)),
    "mae_p95": float(np.quantile(bootstrap_maes, 0.95)),
    "mean_prediction_interval_width": float(predictions["linear_bootstrap_width"].mean()),
}
(OUTPUT / "bootstrap_summary.json").write_text(json.dumps(bootstrap_summary, indent=2), encoding="utf-8")
print(json.dumps(bootstrap_summary, indent=2))

if not args.no_plots:
    plt.figure(figsize=(8, 5))
    plt.hist(bootstrap_rmses, bins=25, alpha=0.82)
    plt.axvline(np.mean(bootstrap_rmses), linestyle="--", linewidth=2, label=f"mean={np.mean(bootstrap_rmses):.3f}")
    plt.xlabel("Bootstrap RMSE")
    plt.ylabel("Refits")
    plt.title("Linear Regression bootstrap RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "16_bootstrap_rmse.png", dpi=160)
    plt.close()

# -----------------------------------------------------------------------------
# 12. SCENARIO ANALYSIS + DECISION POLICY
# -----------------------------------------------------------------------------
print("\n[12/12] Building scenario analysis and decision policy...")

load_review_threshold = float(y_train.quantile(0.75))

scenario_rows = []
compactness_values = sorted(set(np.quantile(df["relative_compactness"], [0.15, 0.50, 0.85]).round(3)))
surface_values = sorted(set(np.quantile(df["surface_area"], [0.20, 0.50, 0.80]).round(3)))
glazing_values = sorted(df["glazing_area"].unique())
orientation_values = sorted(df["orientation"].unique())

base = X_train.median(numeric_only=True).to_dict()
base["orientation"] = float(X_train["orientation"].mode().iloc[0])
base["glazing_area_distribution"] = float(X_train["glazing_area_distribution"].mode().iloc[0])

for compactness in compactness_values:
    for surface in surface_values:
        for glazing in glazing_values:
            for orientation in orientation_values:
                row = dict(base)
                row["relative_compactness"] = float(compactness)
                row["surface_area"] = float(surface)
                row["glazing_area"] = float(glazing)
                row["orientation"] = float(orientation)
                scenario_rows.append(row)

scenarios = pd.DataFrame(scenario_rows)[feature_columns]
scenarios["linear_prediction"] = linear_model.predict(scenarios)
scenarios["ridge_prediction"] = ridge_model.predict(scenarios)
scenarios["best_prediction"] = best_model.predict(scenarios)
scenarios["model_spread"] = scenarios[["linear_prediction", "ridge_prediction", "best_prediction"]].max(axis=1) - scenarios[["linear_prediction", "ridge_prediction", "best_prediction"]].min(axis=1)
scenarios["decision"] = np.where(
    scenarios["linear_prediction"] >= load_review_threshold,
    "HIGH LOAD - REVIEW",
    "LOW / NORMAL LOAD",
)
scenarios = scenarios.sort_values(["linear_prediction", "model_spread"], ascending=[False, False]).reset_index(drop=True)
scenarios.to_csv(OUTPUT / "scenario_analysis.csv", index=False)

print("Heating-load review threshold (training 75th percentile):", round(load_review_threshold, 3))
print("\nHighest predicted heating-load scenarios:")
print(scenarios.head(15).round(3).to_string(index=False))

high_load_share = float((scenarios["decision"] == "HIGH LOAD - REVIEW").mean())
print(f"Scenario grid flagged for review: {100 * high_load_share:.1f}%")

# A concrete single-design inference example.
example_design = X_train.median(numeric_only=True).to_dict()
example_design["orientation"] = float(X_train["orientation"].mode().iloc[0])
example_design["glazing_area_distribution"] = float(X_train["glazing_area_distribution"].mode().iloc[0])
example = pd.DataFrame([example_design])[feature_columns]
example_linear = float(linear_model.predict(example)[0])
example_best = float(best_model.predict(example)[0])
example_decision = "HIGH LOAD - REVIEW" if example_linear >= load_review_threshold else "LOW / NORMAL LOAD"

print("\nExample design:")
print(example.round(3).to_string(index=False))
print("Linear Regression heating-load estimate:", round(example_linear, 3))
print("Best-model heating-load estimate:", round(example_best, 3))
print("Decision:", example_decision)

if not args.no_plots:
    scenario_plot = scenarios.groupby("glazing_area", as_index=False)["linear_prediction"].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(scenario_plot["glazing_area"], scenario_plot["linear_prediction"], marker="o")
    plt.axhline(load_review_threshold, linestyle="--", label="review threshold")
    plt.xlabel("Glazing area")
    plt.ylabel("Mean predicted Heating Load")
    plt.title("Scenario sensitivity — glazing area")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "17_scenario_glazing_sensitivity.png", dpi=160)
    plt.close()

# -----------------------------------------------------------------------------
# RETAIN MODEL AND MACHINE-READABLE EVIDENCE
# -----------------------------------------------------------------------------
joblib.dump(linear_model, OUTPUT / "linear_regression.joblib")
joblib.dump(best_model, OUTPUT / "best_model.joblib")

metrics_payload = {
    "dataset": {
        "name": "UCI Energy Efficiency",
        "uci_id": 242,
        "rows": int(len(df)),
        "features": int(len(feature_columns)),
        "primary_target": target,
        "licence": "CC BY 4.0",
        "doi": "10.24432/C51307",
    },
    "split": {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": float(args.test_size),
        "random_seed": int(args.seed),
    },
    "models": {
        row["model"]: {
            "mae": row["mae"],
            "rmse": row["rmse"],
            "r2": row["r2"],
            "mape": row["mape"],
        }
        for row in metric_rows
    },
    "linear_regression": {
        "rmse_improvement_vs_dummy_pct": float(improvement_vs_baseline),
        "intercept": float(model_fitted.intercept_),
        "residual_mean": float(predictions["linear_residual"].mean()),
        "residual_std": float(predictions["linear_residual"].std()),
    },
    "regularisation": {
        "ridge_alpha": float(ridge_model.named_steps["model"].alpha_),
        "lasso_alpha": float(lasso_model.named_steps["model"].alpha_),
        "polynomial_ridge_alpha": float(poly_model.named_steps["model"].alpha_),
    },
    "cross_validation": {
        row["model"]: {
            "rmse_mean": row["cv_rmse_mean"],
            "rmse_std": row["cv_rmse_std"],
            "fold_rmse": row["fold_rmse"],
        }
        for row in cv_rows
    },
    "bootstrap": bootstrap_summary,
    "decision": {
        "heating_load_review_threshold": load_review_threshold,
        "scenario_high_load_share": high_load_share,
        "example_linear_prediction": example_linear,
        "example_best_prediction": example_best,
        "example_decision": example_decision,
    },
    "best_holdout_model": best_model_name,
    "limitations": [
        "The UCI data is simulated building-energy data, not a representative sample of all real buildings.",
        "Regression coefficients are associations conditional on correlated design variables, not causal effects.",
        "A production engineering decision would require external measured data, climate/location variables and domain validation.",
    ],
}

(OUTPUT / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

print("\n" + "=" * 88)
print("PROJECT COMPLETE")
print("=" * 88)
print("Linear Regression holdout RMSE:", round(float(linear_metrics["rmse"]), 4))
print("Linear Regression holdout R²:", round(float(linear_metrics["r2"]), 4))
print("Best holdout model:", best_model_name)
print("Decision threshold:", round(load_review_threshold, 4))
print("Evidence written to:", OUTPUT)
print(
    "Final interpretation: ordinary Linear Regression is retained as the transparent baseline. "
    "Use the comparison models to test whether extra complexity materially improves the measured decision problem."
)
