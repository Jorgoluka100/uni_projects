from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "income", "tenure_months", "usage"]


def generate(n: int, seed: int, shift: float = 0.0, concept: float = 0.0) -> pd.DataFrame:
    if not isinstance(n, int) or n < 100:
        raise ValueError("n must be an integer of at least 100")
    if not np.isfinite([shift, concept]).all() or shift < -9 or shift > 8:
        raise ValueError("shift and concept must be finite; shift must be between -9 and 8")
    rng = np.random.default_rng(seed)
    segment_probability = [0.7 - max(0.0, shift) * 0.08, 0.3 + max(0.0, shift) * 0.08]
    segment = rng.choice(["A", "B"], n, p=segment_probability)
    age = rng.normal(39 + 3 * shift, 11, n).clip(18, 80)
    income = rng.lognormal(np.log(42000 * (1 + 0.10 * shift)), 0.45, n)
    tenure = rng.gamma(2.4 + 0.15 * shift, 12, n).clip(1, 120)
    usage = rng.normal(52 - 5 * shift, 14, n).clip(0, 100)
    logit = (
        -2.2
        + 0.018 * (age - 40)
        + 0.000012 * (income - 42000)
        - 0.018 * (tenure - 24)
        + 0.025 * (usage - 50)
        + 0.55 * (segment == "B")
        + concept * (0.04 * (age - 40) - 0.03 * (usage - 50))
    )
    probability = 1 / (1 + np.exp(-logit))
    target = (rng.random(n) < probability).astype(int)
    if np.unique(target).size != 2:
        raise ValueError("generated sample must contain both target classes; increase n")
    return pd.DataFrame({
        "age": age,
        "income": income,
        "tenure_months": tenure,
        "usage": usage,
        "segment": segment,
        "target": target,
    })


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with a defined fallback for constant references."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.ndim != 1 or cur.ndim != 1 or ref.size == 0 or cur.size == 0:
        raise ValueError("reference and current must be non-empty 1D arrays")
    if not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("reference and current values must be finite")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least two")

    if np.unique(ref).size == 1:
        centre = float(ref[0])
        epsilon = max(abs(centre) * 1e-9, 1e-9)
        edges = np.array([-np.inf, centre - epsilon, centre + epsilon, np.inf])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if edges.size < 3:
            edges = np.linspace(float(ref.min()), float(ref.max()), 3)
        edges[0], edges[-1] = -np.inf, np.inf

    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_share = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    truth = np.asarray(y)
    probability = np.asarray(p, dtype=float)
    if truth.ndim != 1 or probability.ndim != 1 or truth.size == 0 or truth.shape != probability.shape:
        raise ValueError("target and probability must be non-empty aligned 1D arrays")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("target must contain binary 0/1 labels")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probability must be finite and within [0, 1]")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least two")
    truth = truth.astype(int)
    edges = np.linspace(0, 1, bins + 1)
    score = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probability >= low) & (
            probability <= high if index == bins - 1 else probability < high
        )
        if mask.any():
            score += float(mask.mean()) * abs(float(truth[mask].mean() - probability[mask].mean()))
    return float(score)


def subgroup_metrics(df: pd.DataFrame, p: np.ndarray) -> dict:
    required = {"segment", "target"}
    missing = sorted(required - set(df.columns))
    probabilities = np.asarray(p, dtype=float)
    if df.empty or missing or probabilities.ndim != 1 or len(probabilities) != len(df):
        raise ValueError(f"subgroup data must be non-empty, aligned and include {sorted(required)}")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("subgroup probabilities must be finite and within [0, 1]")
    target = df["target"].to_numpy()
    if not np.isin(target, [0, 1]).all():
        raise ValueError("subgroup target must contain binary 0/1 labels")
    out = {}
    for segment in sorted(df.segment.astype(str).unique()):
        mask = df.segment.astype(str).to_numpy() == segment
        labels = target[mask].astype(int)
        scores = probabilities[mask]
        out[segment] = {
            "rows": int(mask.sum()),
            "prevalence": float(labels.mean()),
            "brier": float(brier_score_loss(labels, scores)),
            "mean_score": float(scores.mean()),
        }
    return out


def evaluate_batch(model, ref: pd.DataFrame, batch: pd.DataFrame, ref_auc: float, name: str) -> dict:
    required = set(FEATURES) | {"target"}
    missing = sorted(required - set(batch.columns))
    missing_reference = sorted(set(FEATURES) - set(ref.columns))
    if batch.empty or ref.empty or missing or missing_reference:
        raise ValueError(f"reference and batch must be non-empty with required columns; batch missing={missing}, reference missing={missing_reference}")
    if not np.isfinite(ref_auc) or not 0 <= ref_auc <= 1:
        raise ValueError("ref_auc must be finite and within [0, 1]")
    y = batch["target"].to_numpy()
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("batch target must contain both binary classes")
    probabilities = np.asarray(model.predict_proba(batch[FEATURES])[:, 1], dtype=float)
    if probabilities.shape != y.shape or not np.isfinite(probabilities).all():
        raise ValueError("model must return one finite probability per batch row")

    drift = {}
    for column in FEATURES:
        reference_values = ref[column].to_numpy(dtype=float)
        current_values = batch[column].to_numpy(dtype=float)
        if not np.isfinite(reference_values).all() or not np.isfinite(current_values).all():
            raise ValueError(f"feature {column} contains non-finite values")
        statistic, p_value = ks_2samp(reference_values, current_values)
        drift[column] = {
            "psi": psi(reference_values, current_values),
            "ks_stat": float(statistic),
            "ks_pvalue": float(p_value),
        }

    auc = float(roc_auc_score(y, probabilities))
    pr_auc = float(average_precision_score(y, probabilities))
    brier = float(brier_score_loss(y, probabilities))
    calibration_error = ece(y, probabilities)
    max_psi = max(value["psi"] for value in drift.values())
    auc_drop = ref_auc - auc
    level = "green"
    reasons = []
    if max_psi >= 0.25 or auc_drop >= 0.08 or calibration_error >= 0.08:
        level = "red"
    elif max_psi >= 0.10 or auc_drop >= 0.04 or calibration_error >= 0.05:
        level = "amber"
    if max_psi >= 0.10:
        reasons.append(f"feature drift PSI={max_psi:.3f}")
    if auc_drop >= 0.04:
        reasons.append(f"ROC-AUC drop={auc_drop:.3f}")
    if calibration_error >= 0.05:
        reasons.append(f"ECE={calibration_error:.3f}")
    return {
        "batch": name,
        "rows": len(batch),
        "drift": drift,
        "max_psi": max_psi,
        "performance": {
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "brier": brier,
            "ece": calibration_error,
            "auc_drop_vs_reference": auc_drop,
        },
        "subgroups": subgroup_metrics(batch, probabilities),
        "alert": level,
        "reasons": reasons,
    }


def run(output_dir: Path, seed: int = 42) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = generate(18000, seed)
    reference = generate(7000, seed + 1)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    model.fit(train[FEATURES], train.target)
    reference_probability = model.predict_proba(reference[FEATURES])[:, 1]
    reference_auc = float(roc_auc_score(reference.target, reference_probability))

    batches = [
        ("stable", generate(7000, seed + 2, 0.0, 0.0)),
        ("mild_shift", generate(7000, seed + 3, 0.7, 0.0)),
        ("feature_shift", generate(7000, seed + 4, 1.5, 0.2)),
        ("concept_shift", generate(7000, seed + 5, 2.0, 0.8)),
    ]
    reports = [evaluate_batch(model, reference, batch, reference_auc, name) for name, batch in batches]
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    reloaded = joblib.load(model_path)
    parity = bool(np.allclose(
        model.predict_proba(reference[FEATURES].head(100)),
        reloaded.predict_proba(reference[FEATURES].head(100)),
    ))
    payload = {
        "project": "ModelWatch",
        "verification_pass": bool(parity and reports[0]["alert"] == "green" and reports[-1]["alert"] == "red"),
        "scope": "deterministic production-monitoring simulation; not a live production system",
        "reference": {
            "rows": len(reference),
            "roc_auc": reference_auc,
            "pr_auc": float(average_precision_score(reference.target, reference_probability)),
            "brier": float(brier_score_loss(reference.target, reference_probability)),
            "ece": ece(reference.target.to_numpy(), reference_probability),
        },
        "batches": reports,
        "model_reload_parity": parity,
        "policy": {
            "amber": "PSI>=0.10 or AUC drop>=0.04 or ECE>=0.05",
            "red": "PSI>=0.25 or AUC drop>=0.08 or ECE>=0.08",
            "action": "red recommends investigation/retraining; this demo never auto-deploys a replacement model",
        },
    }
    (output_dir / "verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = [
        {"batch": report["batch"], "alert": report["alert"], "max_psi": report["max_psi"], **report["performance"]}
        for report in reports
    ]
    pd.DataFrame(summary).to_csv(output_dir / "batch_summary.csv", index=False)
    print(json.dumps(payload, indent=2))
    return payload


def self_test() -> None:
    result = run(Path("/tmp/modelwatch_selftest"), 123)
    assert result["verification_pass"], result
    assert result["batches"][0]["alert"] == "green"
    assert result["batches"][-1]["alert"] == "red"
    print("ModelWatch self-test passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ModelWatch drift and model-risk monitoring simulation.")
    parser.add_argument("--output-dir", type=Path, default=Path("modelwatch_artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result = run(args.output_dir, args.seed)
    return 0 if result["verification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
