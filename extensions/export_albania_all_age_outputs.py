"""Export reproducible, machine-readable outputs for the Albania all-age scenario.

Run from the repository root:
    python extensions/export_albania_all_age_outputs.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from albania_all_age_allocation import (
    TOTAL_POPULATION,
    build_age_profile,
    compare_allocation_scenarios,
    run_weight_sensitivity,
)


def export_outputs(
    output_dir: Path,
    regional_budget_eur: float = 40_000_000,
    n_sims: int = 1_000,
    seed: int = 42,
) -> dict[str, str]:
    """Write source-derived profile, scenario and sensitivity CSVs plus metadata."""
    if regional_budget_eur < 0:
        raise ValueError("regional_budget_eur must be non-negative.")
    if n_sims < 1:
        raise ValueError("n_sims must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    profile = build_age_profile()
    comparison = compare_allocation_scenarios(regional_budget_eur)
    sensitivity = run_weight_sensitivity(
        regional_budget_eur=regional_budget_eur,
        n_sims=n_sims,
        seed=seed,
    )

    files = {
        "prefecture_age_profile.csv": profile,
        "allocation_scenario_comparison.csv": comparison,
        "allocation_weight_sensitivity.csv": sensitivity,
    }
    written: dict[str, str] = {}
    for filename, frame in files.items():
        path = output_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8")
        written[filename] = str(path)

    metadata = {
        "source": "INSTAT Regional Statistical Yearbook 2024, Table 1.6; Census 2023",
        "population_total": TOTAL_POPULATION,
        "prefecture_count": int(len(profile)),
        "scenario_scope": "balanced_all_age",
        "age_cohorts": ["0-14", "15-64", "65+"],
        "regional_budget_eur": float(regional_budget_eur),
        "sensitivity_simulations": int(n_sims),
        "random_seed": int(seed),
        "equity_floor": 0.15,
        "default_weights": {
            "population": 0.40,
            "age_0_14": 0.20,
            "age_15_64": 0.20,
            "age_65_plus": 0.20,
        },
        "interpretation": (
            "Demographic allocation scenarios only. Weights and equity floor are "
            "assumptions, not measured clinical need, utilisation, or expenditure."
        ),
    }
    metadata_path = output_dir / "scenario_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    written["scenario_metadata.json"] = str(metadata_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/albania_all_age"),
        help="Directory for generated CSV and JSON outputs.",
    )
    parser.add_argument(
        "--regional-budget-eur",
        type=float,
        default=40_000_000,
        help="Illustrative regional budget pool in euros (default: 40 million).",
    )
    parser.add_argument("--simulations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for filename, path in export_outputs(
        args.output_dir,
        regional_budget_eur=args.regional_budget_eur,
        n_sims=args.simulations,
        seed=args.seed,
    ).items():
        print(f"Wrote {filename}: {path}")


if __name__ == "__main__":
    main()
