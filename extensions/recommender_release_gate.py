"""Evidence gate for the temporal MovieLens recommender extension.

The gate validates artifacts emitted by ``recommender_v2.py``. It does not require
one model to beat another; ranking results are evidence, not a target to tune against.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def finite_01(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def validate(artifact_dir: Path) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    metrics_path = artifact_dir / "metrics.json"
    detail_path = artifact_dir / "ranking_evaluation.csv"
    for path in (metrics_path, detail_path):
        if not path.is_file():
            errors.append(f"missing artifact: {path.name}")
    if errors:
        return errors, {"verification_pass": False, "errors": errors}

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    detail = pd.read_csv(detail_path)
    config = payload.get("config", {})
    rows = payload.get("rows", {})
    evaluation = payload.get("evaluation", {})
    limitations = payload.get("limitations", [])

    k = int(config.get("k", 0))
    held_out = int(rows.get("held_out_users", 0))
    if k <= 0:
        errors.append("invalid ranking cutoff k")
    if held_out <= 0:
        errors.append("no held-out users")
    if set(evaluation) != {"popularity", "latent_svd"}:
        errors.append("evaluation must contain popularity and latent_svd")
    if len(limitations) < 3:
        errors.append("limitations are incomplete")
    if "files.grouplens.org" not in str(payload.get("source", "")):
        errors.append("MovieLens source URL missing")

    expected_columns = {
        "userId", "target_movieId", "model", "rank", f"recall@{k}",
        f"ndcg@{k}", "candidate_count", "title",
    }
    missing_columns = sorted(expected_columns - set(detail.columns))
    if missing_columns:
        errors.append("ranking detail missing columns: " + ", ".join(missing_columns))
    else:
        if detail.duplicated(["userId", "model"]).any():
            errors.append("duplicate user/model ranking rows")
        if set(detail["model"]) != {"popularity", "latent_svd"}:
            errors.append("ranking detail model set is incomplete")
        if (detail["rank"] < 1).any():
            errors.append("rank must be >= 1")
        if (detail["candidate_count"] < 2).any():
            errors.append("candidate set must contain at least two items")
        for column in (f"recall@{k}", f"ndcg@{k}"):
            if not detail[column].map(finite_01).all():
                errors.append(f"invalid {column} values")

    for model in ("popularity", "latent_svd"):
        values = evaluation.get(model, {}) if isinstance(evaluation, dict) else {}
        if int(values.get("users", 0)) != held_out:
            errors.append(f"{model} user count does not match held-out users")
        for metric in (f"recall@{k}", f"ndcg@{k}"):
            if not finite_01(values.get(metric)):
                errors.append(f"invalid {model} {metric}")
        try:
            median_rank = float(values.get("median_rank"))
            if not math.isfinite(median_rank) or median_rank < 1:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"invalid {model} median_rank")

    report = {
        "project": "Movie Recommender v2",
        "verification_pass": not errors,
        "errors": errors,
        "held_out_users": held_out,
        "k": k,
        "evaluation": evaluation,
        "evidence_policy": "Offline MovieLens sampled-negative ranking benchmark; not current product traffic.",
    }
    return errors, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    errors, report = validate(args.artifact_dir)
    output = args.artifact_dir / "verification.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
