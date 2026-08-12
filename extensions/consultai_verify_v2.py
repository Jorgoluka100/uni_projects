"""Independent verifier for ConsultAI.

ConsultAI is intentionally a deterministic synthetic decision-science project. This
script reconstructs the use-case register, seeded Monte Carlo outcomes, exhaustive
budget-constrained portfolio search and stress tests in a fresh process, then checks
the notebook's exported governance artefacts against those independently reproduced
results.

No commercial-performance claim is created here. The purpose is reproducibility and
decision integrity.

Usage:
    python extensions/consultai_verify_v2.py --artifact-dir consultai_application_artifacts
    python extensions/consultai_verify_v2.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SEED = 42
BUDGET = 600_000.0


@dataclass(frozen=True)
class AIUseCase:
    name: str
    department: str
    annual_value_gbp: float
    delivery_cost_gbp: float
    months_to_value: int
    data_readiness: int
    technical_feasibility: int
    adoption_readiness: int
    risk: int

    def validate(self) -> None:
        if self.annual_value_gbp <= 0 or self.delivery_cost_gbp <= 0:
            raise ValueError("value and cost must be positive")
        if self.months_to_value not in range(1, 37):
            raise ValueError("months_to_value must be 1-36")
        for field in ("data_readiness", "technical_feasibility", "adoption_readiness", "risk"):
            if not 1 <= getattr(self, field) <= 5:
                raise ValueError(f"{field} must be 1-5")


def use_cases() -> list[AIUseCase]:
    return [
        AIUseCase("Support ticket copilot", "Customer Service", 520_000, 160_000, 5, 4, 4, 4, 2),
        AIUseCase("Demand forecasting", "Operations", 740_000, 260_000, 8, 4, 4, 3, 2),
        AIUseCase("Automated CV screening", "HR", 210_000, 140_000, 6, 3, 4, 2, 5),
        AIUseCase("Invoice anomaly detection", "Finance", 430_000, 190_000, 7, 4, 4, 4, 3),
        AIUseCase("Marketing content generator", "Marketing", 180_000, 70_000, 3, 3, 5, 4, 3),
        AIUseCase("Predictive maintenance", "Facilities", 610_000, 340_000, 12, 2, 3, 2, 3),
    ]


def priority_score(case: AIUseCase) -> float:
    case.validate()
    value_score = min(case.annual_value_gbp / case.delivery_cost_gbp, 8) / 8 * 100
    speed_score = (37 - case.months_to_value) / 36 * 100
    readiness = statistics.mean([
        case.data_readiness,
        case.technical_feasibility,
        case.adoption_readiness,
    ]) / 5 * 100
    risk_score = (6 - case.risk) / 5 * 100
    return round(0.35 * value_score + 0.15 * speed_score + 0.30 * readiness + 0.20 * risk_score, 1)


def simulate_npv(case: AIUseCase, trials: int = 10_000, seed: int = SEED) -> dict[str, float]:
    rng = random.Random(seed + sum(map(ord, case.name)))
    outcomes: list[float] = []
    for _ in range(trials):
        adoption = min(1.0, max(0.0, rng.gauss(case.adoption_readiness / 5, 0.12)))
        delivery_multiplier = max(0.75, rng.lognormvariate(0, 0.18))
        value_multiplier = max(0.25, rng.gauss(1, 0.22))
        realised_value = case.annual_value_gbp * adoption * value_multiplier
        realised_cost = case.delivery_cost_gbp * delivery_multiplier
        outcomes.append(realised_value - realised_cost)
    outcomes.sort()
    return {
        "mean_npv": round(statistics.mean(outcomes), 2),
        "p10_npv": round(outcomes[int(0.10 * trials)], 2),
        "probability_positive": round(sum(value > 0 for value in outcomes) / trials, 4),
    }


def build_register() -> list[dict[str, Any]]:
    ranked = sorted(((priority_score(case), case) for case in use_cases()), reverse=True, key=lambda pair: pair[0])
    rows: list[dict[str, Any]] = []
    for score, case in ranked:
        row = {**asdict(case), "priority_score": score, **simulate_npv(case)}
        row["risk_adjusted_npv"] = row["mean_npv"] * row["probability_positive"]
        row["downside_gap"] = row["mean_npv"] - row["p10_npv"]
        row["readiness_mean"] = statistics.mean([
            row["data_readiness"],
            row["technical_feasibility"],
            row["adoption_readiness"],
        ])
        rows.append(row)
    return rows


def evaluate_portfolio(register: list[dict[str, Any]], candidate_names: Iterable[str]) -> dict[str, Any]:
    names = set(candidate_names)
    chosen = [row for row in register if row["name"] in names]
    return {
        "selected": [row["name"] for row in chosen],
        "number_selected": len(chosen),
        "spend_gbp": sum(float(row["delivery_cost_gbp"]) for row in chosen),
        "expected_npv_gbp": sum(float(row["mean_npv"]) for row in chosen),
        "p10_npv_gbp": sum(float(row["p10_npv"]) for row in chosen),
        "risk_adjusted_npv_gbp": sum(float(row["risk_adjusted_npv"]) for row in chosen),
        "average_priority": statistics.mean([float(row["priority_score"]) for row in chosen]) if chosen else 0.0,
    }


def exact_portfolio_search(
    register: list[dict[str, Any]],
    budget_gbp: float,
    minimum_probability: float = 0.70,
    maximum_projects: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = [row for row in register if float(row["probability_positive"]) >= minimum_probability]
    names = [row["name"] for row in eligible]
    candidates: list[dict[str, Any]] = []
    for subset_size in range(1, len(names) + 1):
        if maximum_projects is not None and subset_size > maximum_projects:
            continue
        for subset in itertools.combinations(names, subset_size):
            result = evaluate_portfolio(register, subset)
            if result["spend_gbp"] <= budget_gbp:
                result["objective"] = (
                    result["risk_adjusted_npv_gbp"]
                    + 0.10 * result["p10_npv_gbp"]
                    + 500.0 * result["average_priority"]
                )
                candidates.append(result)
    if not candidates:
        empty = evaluate_portfolio(register, [])
        empty["objective"] = 0.0
        return empty, [empty]
    candidates.sort(key=lambda row: (row["objective"], row["expected_npv_gbp"]), reverse=True)
    return candidates[0], candidates


STRESS_SCENARIOS = {
    "base": {"value_multiplier": 1.00, "cost_multiplier": 1.00, "probability_shift": 0.00},
    "delivery_delay": {"value_multiplier": 0.92, "cost_multiplier": 1.20, "probability_shift": -0.05},
    "weak_adoption": {"value_multiplier": 0.72, "cost_multiplier": 1.05, "probability_shift": -0.15},
    "strong_adoption": {"value_multiplier": 1.18, "cost_multiplier": 1.02, "probability_shift": 0.08},
    "combined_downside": {"value_multiplier": 0.65, "cost_multiplier": 1.30, "probability_shift": -0.20},
}


def stress_portfolio(register: list[dict[str, Any]], selected_names: list[str]) -> list[dict[str, Any]]:
    selected = [row for row in register if row["name"] in set(selected_names)]
    output: list[dict[str, Any]] = []
    for scenario, assumptions in STRESS_SCENARIOS.items():
        expected = 0.0
        risk_adjusted = 0.0
        probabilities: list[float] = []
        for row in selected:
            stressed_npv = (
                float(row["annual_value_gbp"]) * assumptions["value_multiplier"]
                - float(row["delivery_cost_gbp"]) * assumptions["cost_multiplier"]
            )
            probability = min(1.0, max(0.0, float(row["probability_positive"]) + assumptions["probability_shift"]))
            expected += stressed_npv
            risk_adjusted += stressed_npv * probability
            probabilities.append(probability)
        output.append({
            "scenario": scenario,
            "expected_npv_gbp": expected,
            "risk_adjusted_npv_gbp": risk_adjusted,
            "minimum_success_probability": min(probabilities) if probabilities else 0.0,
        })
    output.sort(key=lambda row: row["risk_adjusted_npv_gbp"], reverse=True)
    return output


def budget_frontier(register: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in range(200_000, 1_000_001, 50_000):
        recommendation, _ = exact_portfolio_search(register, float(budget))
        row = dict(recommendation)
        row["budget_gbp"] = float(budget)
        row["budget_utilisation"] = row["spend_gbp"] / float(budget)
        rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def close_number(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def verify_artifacts(artifact_dir: Path) -> tuple[list[str], dict[str, Any]]:
    required = {
        "opportunity_register.csv",
        "budget_frontier.csv",
        "portfolio_stress_tests.csv",
        "data_card.json",
        "decision_record.json",
        "project_manifest.json",
    }
    errors: list[str] = []
    missing = sorted(name for name in required if not (artifact_dir / name).is_file())
    if missing:
        return ["missing required artifacts: " + ", ".join(missing)], {}

    register = build_register()
    recommendation, _ = exact_portfolio_search(register, BUDGET)
    frontier = budget_frontier(register)
    stress = stress_portfolio(register, recommendation["selected"])

    saved_register = read_csv(artifact_dir / "opportunity_register.csv")
    by_name = {row["name"]: row for row in saved_register}
    if set(by_name) != {row["name"] for row in register}:
        errors.append("opportunity register names do not match deterministic source register")
    numeric_fields = {
        "annual_value_gbp", "delivery_cost_gbp", "months_to_value", "data_readiness",
        "technical_feasibility", "adoption_readiness", "risk", "priority_score",
        "mean_npv", "p10_npv", "probability_positive", "risk_adjusted_npv",
        "downside_gap", "readiness_mean",
    }
    for expected in register:
        saved = by_name.get(expected["name"])
        if saved is None:
            continue
        for field in numeric_fields:
            if field in saved and not close_number(saved[field], expected[field], tolerance=1e-7):
                errors.append(f"opportunity_register mismatch: {expected['name']}::{field}")

    decision = read_json(artifact_dir / "decision_record.json")
    evidence = decision.get("evidence", {}) if isinstance(decision.get("evidence", {}), dict) else {}
    if not close_number(evidence.get("budget_gbp"), BUDGET):
        errors.append("decision record budget does not match configured budget")
    for field in ("spend_gbp", "expected_npv_gbp", "risk_adjusted_npv_gbp"):
        if not close_number(evidence.get(field), recommendation[field]):
            errors.append(f"decision record does not reproduce exhaustive optimum: {field}")
    if recommendation["spend_gbp"] > BUDGET:
        errors.append("reproduced portfolio exceeds budget")
    if ", ".join(recommendation["selected"]) not in str(decision.get("recommendation", "")):
        errors.append("decision recommendation does not list independently reproduced selection")

    data_card = read_json(artifact_dir / "data_card.json")
    if data_card.get("synthetic") is not True:
        errors.append("data card must explicitly label ConsultAI inputs as synthetic")
    if len(data_card.get("known_limitations", [])) < 3:
        errors.append("data card must preserve at least three limitations")
    if len(data_card.get("prohibited_uses", [])) < 2:
        errors.append("data card must preserve prohibited-use guardrails")

    manifest = read_json(artifact_dir / "project_manifest.json")
    if "Synthetic" not in str(manifest.get("truth_first_notice", "")):
        errors.append("project manifest truth-first synthetic-data notice is missing")
    quality = manifest.get("quality", {})
    if isinstance(quality, dict) and quality.get("failed_errors", 0) not in (0, "0", None):
        errors.append("project manifest reports failed quality-gate errors")

    saved_frontier = read_csv(artifact_dir / "budget_frontier.csv")
    frontier_by_budget = {int(float(row["budget_gbp"])): row for row in saved_frontier}
    for expected in frontier:
        saved = frontier_by_budget.get(int(expected["budget_gbp"]))
        if saved is None:
            errors.append(f"budget frontier missing {int(expected['budget_gbp'])}")
            continue
        for field in ("spend_gbp", "expected_npv_gbp", "risk_adjusted_npv_gbp", "budget_utilisation"):
            if not close_number(saved.get(field), expected[field]):
                errors.append(f"budget frontier mismatch at {int(expected['budget_gbp'])}: {field}")

    saved_stress = read_csv(artifact_dir / "portfolio_stress_tests.csv")
    stress_by_name = {row["scenario"]: row for row in saved_stress}
    for expected in stress:
        saved = stress_by_name.get(expected["scenario"])
        if saved is None:
            errors.append(f"stress test missing scenario: {expected['scenario']}")
            continue
        for field in ("expected_npv_gbp", "risk_adjusted_npv_gbp", "minimum_success_probability"):
            if not close_number(saved.get(field), expected[field]):
                errors.append(f"stress mismatch {expected['scenario']}::{field}")

    report = {
        "project": "ConsultAI",
        "verification_pass": not errors,
        "errors": errors,
        "reproduced_recommendation": recommendation,
        "register_rows": len(register),
        "frontier_points": len(frontier),
        "stress_scenarios": [row["scenario"] for row in stress],
        "evidence_policy": "Synthetic educational inputs; no commercial-performance claim.",
    }
    return errors, report


def write_fixture(root: Path) -> None:
    register = build_register()
    recommendation, _ = exact_portfolio_search(register, BUDGET)
    frontier = budget_frontier(register)
    stress = stress_portfolio(register, recommendation["selected"])

    register_fields = list(register[0])
    with (root / "opportunity_register.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=register_fields)
        writer.writeheader(); writer.writerows(register)
    frontier_fields = list(frontier[0])
    with (root / "budget_frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=frontier_fields)
        writer.writeheader(); writer.writerows(frontier)
    stress_fields = list(stress[0])
    with (root / "portfolio_stress_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=stress_fields)
        writer.writeheader(); writer.writerows(stress)
    (root / "data_card.json").write_text(json.dumps({
        "synthetic": True,
        "known_limitations": ["illustrative estimates", "dependencies omitted", "assumed distributions"],
        "prohibited_uses": ["real capital allocation", "automatic approval"],
    }), encoding="utf-8")
    (root / "decision_record.json").write_text(json.dumps({
        "recommendation": "Fund: " + ", ".join(recommendation["selected"]),
        "evidence": {
            "budget_gbp": BUDGET,
            "spend_gbp": recommendation["spend_gbp"],
            "expected_npv_gbp": recommendation["expected_npv_gbp"],
            "risk_adjusted_npv_gbp": recommendation["risk_adjusted_npv_gbp"],
        },
    }), encoding="utf-8")
    (root / "project_manifest.json").write_text(json.dumps({
        "truth_first_notice": "Synthetic or educational results are not production or commercial claims.",
        "quality": {"failed_errors": 0},
    }), encoding="utf-8")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_fixture(root)
        errors, report = verify_artifacts(root)
        if errors or report.get("verification_pass") is not True:
            raise AssertionError(errors)
    print("ConsultAI verifier self-test passed.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.artifact_dir is None:
        parser.error("--artifact-dir is required unless --self-test is used")
    errors, report = verify_artifacts(args.artifact_dir)
    output = args.artifact_dir / "consultai_verification.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"verification report: {output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
