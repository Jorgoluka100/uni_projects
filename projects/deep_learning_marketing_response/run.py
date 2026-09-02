from __future__ import annotations

import io
import json
import random
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
for folder in (DATA, RESULTS, ARTIFACTS):
    folder.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
SEED = 42
TARGET = "y"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ScoreCard:
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    brier: float
    log_loss: float
    threshold: float
    positive_rate: float


def download_data(force: bool = False) -> Path:
    target = DATA / "bank-additional-full.csv"
    if target.exists() and not force:
        return target
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(name for name in archive.namelist() if name.endswith("bank-additional-full.csv"))
        target.write_bytes(archive.read(member))
    return target


def load_data(path: Path | None = None) -> pd.DataFrame:
    path = path or download_data()
    frame = pd.read_csv(path, sep=";")
    frame.columns = [str(c).strip().lower().replace(".", "_") for c in frame.columns]
    frame[TARGET] = frame[TARGET].map({"yes": 1, "no": 0}).astype(int)
    return frame


def audit_data(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "age", "job", "marital", "education", "default", "housing", "loan", "contact",
        "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
        "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed", TARGET,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Both target classes are required")
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "positive_rate": float(frame[TARGET].mean()),
        "age_min": int(frame["age"].min()),
        "age_max": int(frame["age"].max()),
    }


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop_duplicates().copy()
    # `duration` is only known after the call completes and is therefore removed
    # for a pre-contact targeting application.
    if "duration" in out.columns:
        out = out.drop(columns=["duration"])
    out["was_previously_contacted"] = (out["pdays"] != 999).astype(int)
    out["previous_success"] = (out["poutcome"] == "success").astype(int)
    out["campaign_intensity"] = np.log1p(out["campaign"].clip(lower=0))
    out["macro_pressure"] = out["euribor3m"] * out["emp_var_rate"]
    return out


def temporal_like_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # UCI release does not expose a full event timestamp. We preserve order and
    # use contiguous blocks to avoid random mixing across the source sequence.
    n = len(frame)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:val_end].copy()
    test = frame.iloc[val_end:].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Split produced an empty partition")
    return train, validation, test


def feature_groups(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = [c for c in frame.columns if c != TARGET]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_splits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    numeric, categorical = feature_groups(train)
    preprocessor = build_preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(train.drop(columns=[TARGET])).astype(np.float32)
    x_validation = preprocessor.transform(validation.drop(columns=[TARGET])).astype(np.float32)
    x_test = preprocessor.transform(test.drop(columns=[TARGET])).astype(np.float32)
    y_train = train[TARGET].to_numpy(dtype=np.float32)
    y_validation = validation[TARGET].to_numpy(dtype=np.float32)
    y_test = test[TARGET].to_numpy(dtype=np.float32)
    return preprocessor, x_train, y_train, x_validation, y_validation, x_test, y_test


class MarketingMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict_probabilities(model: nn.Module, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (features,) in loader:
            logits = model(features.to(DEVICE))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            outputs.append(probabilities)
    return np.concatenate(outputs)


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, max_contact_rate: float = 0.25) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        contact_rate = float(pred.mean())
        if contact_rate > max_contact_rate or pred.sum() == 0:
            continue
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold


def score(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> ScoreCard:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    prediction = (clipped >= threshold).astype(int)
    return ScoreCard(
        roc_auc=float(roc_auc_score(y_true, clipped)),
        pr_auc=float(average_precision_score(y_true, clipped)),
        f1=float(f1_score(y_true, prediction, zero_division=0)),
        precision=float(precision_score(y_true, prediction, zero_division=0)),
        recall=float(recall_score(y_true, prediction, zero_division=0)),
        brier=float(brier_score_loss(y_true, clipped)),
        log_loss=float(log_loss(y_true, clipped)),
        threshold=float(threshold),
        positive_rate=float(prediction.mean()),
    )


def train_neural_network(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    max_epochs: int = 60,
) -> tuple[MarketingMLP, pd.DataFrame]:
    model = MarketingMLP(x_train.shape[1]).to(DEVICE)
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    train_loader = make_loader(x_train, y_train, batch_size=512, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -np.inf
    patience = 8
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for features, target in train_loader:
            features = features.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_probability = predict_probabilities(model, x_validation)
        validation_auc = float(roc_auc_score(y_validation, validation_probability))
        scheduler.step(validation_auc)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_roc_auc": validation_auc,
            "learning_rate": current_lr,
        })

        if validation_auc > best_auc + 1e-4:
            best_auc = validation_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(DEVICE)
    return model, pd.DataFrame(history)


def fit_logistic_baseline(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    model.fit(x_train, y_train.astype(int))
    return model


def probability_bins(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"target": y_true, "probability": probabilities})
    frame["bin"] = pd.qcut(frame["probability"], q=bins, duplicates="drop")
    return (
        frame.groupby("bin", observed=True)
        .agg(rows=("target", "size"), predicted_rate=("probability", "mean"), observed_rate=("target", "mean"))
        .reset_index()
        .assign(bin=lambda x: x["bin"].astype(str))
    )


def segment_errors(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    work = frame[["age", "job", "contact", TARGET]].copy().reset_index(drop=True)
    work["probability"] = probabilities
    work["prediction"] = (work["probability"] >= threshold).astype(int)
    work["correct"] = (work["prediction"] == work[TARGET]).astype(int)
    work["age_band"] = pd.cut(work["age"], [0, 29, 39, 49, 59, 200], labels=["<30", "30s", "40s", "50s", "60+"])
    outputs = []
    for column in ["age_band", "contact", "job"]:
        grouped = work.groupby(column, observed=True).agg(
            rows=(TARGET, "size"),
            response_rate=(TARGET, "mean"),
            predicted_probability=("probability", "mean"),
            accuracy=("correct", "mean"),
        ).reset_index()
        grouped.insert(0, "slice", column)
        grouped.rename(columns={column: "value"}, inplace=True)
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True)


def targeting_policy(probability: float, threshold: float) -> dict[str, Any]:
    if probability >= max(threshold, 0.70):
        tier = "priority"
        action = "include in high-priority outreach queue"
    elif probability >= threshold:
        tier = "eligible"
        action = "include if campaign capacity remains"
    else:
        tier = "hold"
        action = "do not target in this campaign"
    return {"probability": float(probability), "threshold": float(threshold), "tier": tier, "action": action}


def save_checkpoint(model: MarketingMLP, input_dim: int) -> None:
    torch.save({"state_dict": model.state_dict(), "input_dim": int(input_dim)}, ARTIFACTS / "marketing_mlp.pt")


def main() -> None:
    set_seed()
    raw = load_data()
    audit = audit_data(raw)
    data = prepare_frame(raw)
    train, validation, test = temporal_like_split(data)

    preprocessor, x_train, y_train, x_validation, y_validation, x_test, y_test = transform_splits(train, validation, test)
    baseline = fit_logistic_baseline(x_train, y_train)
    baseline_validation_probability = baseline.predict_proba(x_validation)[:, 1]
    baseline_threshold = choose_threshold(y_validation, baseline_validation_probability)
    baseline_score = score(y_validation, baseline_validation_probability, baseline_threshold)

    model, history = train_neural_network(x_train, y_train, x_validation, y_validation)
    validation_probability = predict_probabilities(model, x_validation)
    threshold = choose_threshold(y_validation, validation_probability)
    validation_score = score(y_validation, validation_probability, threshold)
    test_probability = predict_probabilities(model, x_test)
    test_score = score(y_test, test_probability, threshold)

    calibration = probability_bins(y_test, test_probability)
    slices = segment_errors(test, test_probability, threshold)
    history.to_csv(RESULTS / "training_history.csv", index=False)
    calibration.to_csv(RESULTS / "calibration_bins.csv", index=False)
    slices.to_csv(RESULTS / "error_slices.csv", index=False)
    pd.DataFrame({
        "target": y_test.astype(int),
        "probability": test_probability,
        "prediction": (test_probability >= threshold).astype(int),
    }).to_csv(RESULTS / "test_predictions.csv", index=False)

    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
    joblib.dump(baseline, ARTIFACTS / "logistic_baseline.joblib")
    save_checkpoint(model, x_train.shape[1])

    example_probability = float(test_probability[0])
    payload = {
        "dataset_audit": audit,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "preprocessed_input_dimension": int(x_train.shape[1]),
        "device": str(DEVICE),
        "logistic_validation": asdict(baseline_score),
        "neural_network_validation": asdict(validation_score),
        "neural_network_test": asdict(test_score),
        "roc_auc_gain_vs_logistic": float(validation_score.roc_auc - baseline_score.roc_auc),
        "epochs_trained": int(len(history)),
        "example_targeting_decision": targeting_policy(example_probability, threshold),
        "limitations": [
            "Historical bank campaign data; not representative of every market or current behaviour.",
            "This model is for marketing-response prioritisation, not lending or eligibility decisions.",
            "A real deployment requires consent/privacy review, calibrated campaign costs and ongoing drift checks.",
        ],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
