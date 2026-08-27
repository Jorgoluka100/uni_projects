"""Small end-to-end scikit-learn classification workflow."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def evaluate(name: str, fitted: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float | str]:
    pred = fitted.predict(X_test)
    proba = fitted.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 3),
    }


def main() -> None:
    X_num, y = make_classification(
        n_samples=1200,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        weights=[0.72, 0.28],
        class_sep=1.0,
        random_state=42,
    )
    X = pd.DataFrame(X_num, columns=["x1", "x2", "x3", "x4", "x5"])
    X["segment"] = pd.cut(X["x1"], bins=[-np.inf, -0.5, 0.5, np.inf], labels=["low", "mid", "high"])
    X.loc[X.sample(frac=0.04, random_state=7).index, "x3"] = np.nan

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["x1", "x2", "x3", "x4", "x5"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["segment"]),
        ]
    )

    baseline = Pipeline([("prep", preprocess), ("model", DummyClassifier(strategy="prior"))])
    model = Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=1000))])
    baseline.fit(X_train, y_train)
    model.fit(X_train, y_train)

    print(pd.DataFrame([evaluate("dummy_prior", baseline, X_test, y_test), evaluate("logistic_regression", model, X_test, y_test)]))


if __name__ == "__main__":
    main()
