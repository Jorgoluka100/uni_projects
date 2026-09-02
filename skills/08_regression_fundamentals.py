"""Focused linear regression fundamentals with baseline, preprocessing and regularisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "price"
NUMERIC_FEATURES = ["size_m2", "bedrooms", "property_age"]
CATEGORICAL_FEATURES = ["area"]


def build_data(n: int = 320) -> pd.DataFrame:
    """Create a deterministic housing dataset with mixed feature types and missing values."""
    rng = np.random.default_rng(RANDOM_STATE)
    size = rng.normal(82, 19, n).clip(35, 165)
    bedrooms = rng.integers(1, 6, n)
    property_age = rng.integers(0, 70, n)
    area = rng.choice(["north", "south", "east", "west"], n, p=[0.25] * 4)

    area_effect = np.select(
        [area == "south", area == "west", area == "north"],
        [28000, 16000, 8000],
        default=0,
    )
    price = (
        65000
        + size * 2850
        + bedrooms * 16500
        - property_age * 850
        + area_effect
        + rng.normal(0, 17500, n)
    )

    frame = pd.DataFrame(
        {
            "size_m2": size,
            "bedrooms": bedrooms,
            "property_age": property_age,
            "area": area,
            TARGET: price,
        }
    )
    frame.loc[[5, 19, 77, 201], "size_m2"] = np.nan
    frame.loc[[11, 99], "property_age"] = np.nan
    return frame


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, object]:
    """Compare a naive baseline, ordinary least squares and regularised regression."""
    return {
        "median_baseline": DummyRegressor(strategy="median"),
        "linear_regression": Pipeline(
            [("prep", build_preprocessor()), ("model", LinearRegression())]
        ),
        "ridge_regression": Pipeline(
            [("prep", build_preprocessor()), ("model", Ridge(alpha=1.0))]
        ),
    }


def evaluate(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "r2": float(r2_score(y_true, predictions)),
    }


def residual_summary(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    residuals = np.asarray(y_true) - np.asarray(predictions)
    return {
        "mean_residual": float(residuals.mean()),
        "median_abs_residual": float(np.median(np.abs(residuals))),
        "p90_abs_residual": float(np.quantile(np.abs(residuals), 0.90)),
    }


def fit_and_compare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    models = build_models()
    rows: list[dict[str, float | str]] = []
    fitted: dict[str, object] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate(y_test, predictions)
        rows.append({"model": name, **metrics})
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best_name = str(results.loc[0, "model"])
    best_predictions = fitted[best_name].predict(X_test)
    diagnostics = residual_summary(y_test, best_predictions)

    print("Model comparison")
    print(results.round({"mae": 2, "rmse": 2, "r2": 3}).to_string(index=False))
    print("\nBest model:", best_name)
    print("Residual diagnostics:", {k: round(v, 2) for k, v in diagnostics.items()})

    return results, fitted


def predict_example(model: object) -> float:
    example = pd.DataFrame(
        [
            {
                "size_m2": 92.0,
                "bedrooms": 3,
                "property_age": 12,
                "area": "south",
            }
        ]
    )
    return float(model.predict(example)[0])


def main() -> None:
    df = build_data()
    print("Rows:", len(df))
    print("Missing values:", df.isna().sum().to_dict())

    results, fitted = fit_and_compare(df)
    preferred = "linear_regression" if "linear_regression" in fitted else str(results.loc[0, "model"])
    example_prediction = predict_example(fitted[preferred])
    print(f"\nExample {preferred} prediction: £{example_prediction:,.0f}")
    print(
        "Interpretation: ordinary Linear Regression is shown explicitly, then compared "
        "with a naive baseline and Ridge regularisation using the same leakage-safe pipeline."
    )


if __name__ == "__main__":
    main()
