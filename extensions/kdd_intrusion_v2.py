"""KDD Cup 1999 v2 — historical intrusion-detection benchmark with honest evaluation.

This extension uses scikit-learn's maintained `fetch_kddcup99` loader and is
explicitly a historical methodology benchmark, not evidence about modern network
traffic. It adds a deterministic data contract, held-out split, trivial baseline,
imbalance-aware metrics, class-level diagnostics and model reload verification.
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


@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.20
    percent10: bool = True
    artifact_dir: Path = Path("kdd_artifacts")


def load_frame(config: Config) -> pd.DataFrame:
    bunch = fetch_kddcup99(
        percent10=config.percent10,
        shuffle=True,
        random_state=config.seed,
        as_frame=True,
    )
    frame = bunch.frame.copy()
    target_name = bunch.target.name if getattr(bunch.target, "name", None) else "target"
    if target_name not in frame:
        frame[target_name] = bunch.target
    frame = frame.rename(columns={target_name: "attack_type"})

    def decode(value):
        return value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray)) else value

    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].map(decode)
    frame["is_attack"] = (~frame["attack_type"].astype(str).str.startswith("normal")).astype(int)
    if frame.isna().all(axis=1).any():
        raise ValueError("found completely empty rows")
    return frame


def make_pipeline(frame: pd.DataFrame, seed: int) -> tuple[Pipeline, list[str]]:
    drop = {"attack_type", "is_attack"}
    features = [c for c in frame.columns if c not in drop]
    categorical = [c for c in features if frame[c].dtype == "object"]
    numeric = [c for c in features if c not in categorical]
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
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "prevalence": float(y.mean()),
    }


def run(config: Config = Config()) -> dict[str, object]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(config)
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
    check = reloaded.predict_proba(test[features].head(50))[:, 1]
    if not np.allclose(check, score[:50]):
        raise AssertionError("reload smoke check failed")

    payload = {
        "dataset": "KDD Cup 1999 via sklearn.datasets.fetch_kddcup99",
        "historical_warning": "This dataset is obsolete as a proxy for modern cyber threats.",
        "config": {**asdict(config), "artifact_dir": str(config.artifact_dir)},
        "rows": {"total": len(frame), "train": len(train), "test": len(test)},
        "baseline_test": baseline_metrics,
        "model_test": model_metrics,
        "limitations": [
            "Random stratified splitting is a benchmark convenience, not a deployment simulation.",
            "KDD Cup 1999 contains redundancy and dated attack/network characteristics.",
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
    print(json.dumps(result["model_test"], indent=2))


if __name__ == "__main__":
    main()
