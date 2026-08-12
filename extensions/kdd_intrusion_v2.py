"""KDD Cup 1999 v2 — historical intrusion-detection benchmark with honest evaluation.

This extension uses scikit-learn's maintained ``fetch_kddcup99`` loader and is
explicitly a historical methodology benchmark, not evidence about modern network
traffic. It enforces the documented KDD schema instead of trusting pandas object
dtypes, removes exact duplicate records before splitting, compares a trivial prior
baseline with a nonlinear model, reports imbalance-aware held-out metrics, exports
attack-type diagnostics and verifies model reload parity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAL_FEATURES = {"protocol_type", "service", "flag"}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.20
    percent10: bool = True
    artifact_dir: Path = Path("kdd_artifacts")


def _decode(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray)) else value


def load_frame(config: Config) -> pd.DataFrame:
    bunch = fetch_kddcup99(
        percent10=config.percent10,
        shuffle=False,
        as_frame=True,
    )
    frame = bunch.frame.copy()
    target_name = bunch.target.name if getattr(bunch.target, "name", None) else frame.columns[-1]
    if target_name not in frame:
        frame[target_name] = bunch.target
    frame = frame.rename(columns={target_name: "attack_type"})

    if "attack_type" not in frame:
        raise KeyError("KDD target column could not be resolved")

    feature_columns = [column for column in frame.columns if column != "attack_type"]
    missing_categorical = sorted(CATEGORICAL_FEATURES - set(feature_columns))
    if missing_categorical:
        raise KeyError(f"missing documented KDD categorical fields: {missing_categorical}")

    # sklearn's dataframe representation can surface numeric KDD columns as object
    # dtype. Decode only the documented categorical fields and target; coerce every
    # other predictor to numeric so the model contract is explicit and reproducible.
    for column in sorted(CATEGORICAL_FEATURES):
        frame[column] = frame[column].map(_decode).astype(str)
    frame["attack_type"] = frame["attack_type"].map(_decode).astype(str)

    numeric_columns = [column for column in feature_columns if column not in CATEGORICAL_FEATURES]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[numeric_columns].isna().any().any():
        bad = frame[numeric_columns].columns[frame[numeric_columns].isna().any()].tolist()
        raise ValueError(f"numeric coercion produced missing values in: {bad[:8]}")

    frame["is_attack"] = (~frame["attack_type"].str.startswith("normal")).astype(int)
    before = len(frame)
    frame = frame.drop_duplicates(subset=feature_columns + ["attack_type"]).reset_index(drop=True)
    frame.attrs["raw_rows"] = before
    frame.attrs["duplicates_removed"] = before - len(frame)

    if frame.empty or frame["is_attack"].nunique() != 2:
        raise ValueError("cleaned benchmark must contain both normal and attack rows")
    return frame


def make_pipeline(frame: pd.DataFrame, seed: int) -> tuple[Pipeline, list[str]]:
    features = [column for column in frame.columns if column not in {"attack_type", "is_attack"}]
    categorical = [column for column in features if column in CATEGORICAL_FEATURES]
    numeric = [column for column in features if column not in CATEGORICAL_FEATURES]
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ]
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )
    return Pipeline([("preprocess", pre), ("model", model)]), features


def binary_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "prevalence": float(y.mean()),
    }


def run(config: Config = Config()) -> dict[str, object]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(config)
    raw_rows = int(frame.attrs.get("raw_rows", len(frame)))
    duplicates_removed = int(frame.attrs.get("duplicates_removed", 0))
    pipeline, features = make_pipeline(frame, config.seed)

    train, test = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=frame["is_attack"],
    )
    pipeline.fit(train[features], train["is_attack"])
    score = pipeline.predict_proba(test[features])[:, 1]
    pred = (score >= 0.5).astype(int)
    model_metrics = binary_metrics(test["is_attack"].to_numpy(), score, pred)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(train), 1)), train["is_attack"])
    dummy_score = dummy.predict_proba(np.zeros((len(test), 1)))[:, 1]
    dummy_pred = dummy.predict(np.zeros((len(test), 1)))
    baseline_metrics = binary_metrics(test["is_attack"].to_numpy(), dummy_score, dummy_pred)

    attack_report = pd.crosstab(
        test["attack_type"].astype(str),
        pd.Series(pred, index=test.index, name="predicted_attack"),
        margins=True,
    )
    attack_report.to_csv(config.artifact_dir / "attack_type_vs_binary_prediction.csv")

    model_path = config.artifact_dir / "kdd_binary_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)
    check = reloaded.predict_proba(test[features].head(100))[:, 1]
    if not np.allclose(check, score[:100]):
        raise AssertionError("reload smoke check failed")

    payload = {
        "dataset": "KDD Cup 1999 via sklearn.datasets.fetch_kddcup99",
        "historical_warning": "This dataset is obsolete as a proxy for modern cyber threats.",
        "config": {**asdict(config), "artifact_dir": str(config.artifact_dir)},
        "rows": {
            "raw": raw_rows,
            "deduplicated": len(frame),
            "duplicates_removed": duplicates_removed,
            "train": len(train),
            "test": len(test),
        },
        "schema": {
            "categorical": sorted(CATEGORICAL_FEATURES),
            "numeric_count": len(features) - len(CATEGORICAL_FEATURES),
        },
        "baseline_test": baseline_metrics,
        "model_test": model_metrics,
        "reload_prediction_match": True,
        "limitations": [
            "Random stratified splitting is a benchmark convenience, not a deployment simulation.",
            "KDD Cup 1999 contains dated attack and network characteristics even after exact deduplication.",
            "Modern evaluation should use contemporary traffic, temporal splits and operational false-positive costs.",
        ],
    }
    (config.artifact_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use full KDD dataset instead of 10% subset")
    parser.add_argument("--artifact-dir", type=Path, default=Path("kdd_artifacts"))
    args = parser.parse_args()
    result = run(Config(percent10=not args.full, artifact_dir=args.artifact_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
