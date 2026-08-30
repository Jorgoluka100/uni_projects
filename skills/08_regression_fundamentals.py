"""Focused regression fundamentals: baseline, preprocessing, regularisation and evaluation."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 240
    size = rng.normal(80, 18, n).clip(35, 160)
    bedrooms = rng.integers(1, 5, n)
    area = rng.choice(["north", "south", "east"], n, p=[0.35, 0.4, 0.25])
    price = (
        60000
        + size * 2800
        + bedrooms * 18000
        + np.where(area == "south", 25000, 0)
        + rng.normal(0, 18000, n)
    )
    frame = pd.DataFrame(
        {"size_m2": size, "bedrooms": bedrooms, "area": area, "price": price}
    )
    frame.loc[[5, 19, 77], "size_m2"] = np.nan
    return frame


def main() -> None:
    df = build_data()
    X = df.drop(columns="price")
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["size_m2", "bedrooms"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["area"]),
        ]
    )

    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    model = Pipeline([("prep", prep), ("ridge", Ridge(alpha=1.0))]).fit(
        X_train, y_train
    )

    baseline_pred = baseline.predict(X_test)
    pred = model.predict(X_test)
    print("Baseline MAE:", round(mean_absolute_error(y_test, baseline_pred), 2))
    print("Ridge MAE:", round(mean_absolute_error(y_test, pred), 2))
    print("Ridge R2:", round(r2_score(y_test, pred), 3))


if __name__ == "__main__":
    main()
