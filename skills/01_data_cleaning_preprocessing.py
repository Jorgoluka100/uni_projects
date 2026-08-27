"""Focused data-cleaning and leakage-safe preprocessing example."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_example() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 31, 31, 44, np.nan, 52, 29, 38, 41, 35],
            "income": ["32000", "41000", "41000", "62000", "50000", "not_known", "37000", "48000", "250000", "45000"],
            "city": ["London", "London", "London", "Leeds", "Leeds", None, "London", "Bristol", "Bristol", "Leeds"],
            "signup_date": ["2026-01-05", "2026-01-08", "2026-01-08", "2026-02-01", "bad_date", "2026-02-10", "2026-03-02", "2026-03-11", "2026-03-20", "2026-04-01"],
            "converted": [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
        }
    )


def main() -> None:
    raw = build_example()
    print("Exact duplicate rows:", int(raw.duplicated().sum()))

    df = raw.drop_duplicates().copy()
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["signup_month"] = df["signup_date"].dt.month
    df = df.drop(columns="signup_date")

    X = df.drop(columns="converted")
    y = df["converted"]
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric = ["age", "income", "signup_month"]
    categorical = ["city"]
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    train_ready = preprocessor.fit_transform(X_train)
    test_ready = preprocessor.transform(X_test)
    print("Train shape:", train_ready.shape)
    print("Test shape:", test_ready.shape)


if __name__ == "__main__":
    main()
