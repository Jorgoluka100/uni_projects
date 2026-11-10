from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def simulate(n: int, seed: int, effect: float) -> pd.DataFrame:
    """Generate a deterministic balanced randomized experiment."""
    if not isinstance(n, int) or n < 4:
        raise ValueError("n must be an integer of at least four")
    if not np.isfinite(effect):
        raise ValueError("effect must be finite")
    rng = np.random.default_rng(seed)
    pre = rng.normal(100, 20, n)
    segment = rng.choice(["new", "returning"], n, p=[0.35, 0.65])
    treatment = np.zeros(n, dtype=int)
    treatment[rng.permutation(n)[: n // 2]] = 1
    noise = rng.normal(0, 12, n)
    outcome = 25 + 0.62 * pre + 4.0 * (segment == "returning") + effect * treatment + noise
    guardrail = rng.normal(5.0, 1.1, n) + 0.02 * treatment
    return pd.DataFrame({
        "pre_metric": pre,
        "segment": segment,
        "treatment": treatment,
        "outcome": outcome,
        "guardrail": guardrail,
    })


def _validate_arms(y, treatment) -> tuple[np.ndarray, np.ndarray]:
    outcome = np.asarray(y, dtype=float)
    assignment = np.asarray(treatment)
    if outcome.ndim != 1 or assignment.ndim != 1 or outcome.size == 0 or outcome.shape != assignment.shape:
        raise ValueError("outcome and treatment must be non-empty aligned 1D arrays")
    if not np.isfinite(outcome).all():
        raise ValueError("outcome must contain only finite values")
    if not np.isin(assignment, [0, 1]).all():
        raise ValueError("treatment must contain only binary 0/1 assignments")
    assignment = assignment.astype(int)
    if np.count_nonzero(assignment == 1) < 2 or np.count_nonzero(assignment == 0) < 2:
        raise ValueError("both experiment arms need at least two observations")
    return outcome, assignment


def mean_effect(y: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    outcome, assignment = _validate_arms(y, t)
    treated = outcome[assignment == 1]
    control = outcome[assignment == 0]
    estimate = float(treated.mean() - control.mean())
    standard_error = math.sqrt(float(
        treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)
    ))
    return estimate, estimate - 1.96 * standard_error, estimate + 1.96 * standard_error


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    outcome = np.asarray(y, dtype=float)
    covariate = np.asarray(x, dtype=float)
    if outcome.ndim != 1 or covariate.ndim != 1 or outcome.size < 2 or outcome.shape != covariate.shape:
        raise ValueError("outcome and pre-treatment covariate must be aligned 1D arrays with at least two rows")
    if not np.isfinite(outcome).all() or not np.isfinite(covariate).all():
        raise ValueError("outcome and pre-treatment covariate must be finite")
    variance = float(np.var(covariate, ddof=1))
    if variance <= 0:
        raise ValueError("pre-treatment covariate must have positive sample variance")
    theta = float(np.cov(outcome, covariate, ddof=1)[0, 1] / variance)
    return outcome - theta * (covariate - covariate.mean()), theta


def stratified_bootstrap(
    y: np.ndarray,
    t: np.ndarray,
    rounds: int,
    seed: int,
) -> tuple[float, float]:
    outcome, assignment = _validate_arms(y, t)
    if not isinstance(rounds, int) or rounds < 100:
        raise ValueError("rounds must be an integer of at least 100")
    rng = np.random.default_rng(seed)
    treated_index = np.where(assignment == 1)[0]
    control_index = np.where(assignment == 0)[0]
    values = np.empty(rounds, dtype=float)
    for index in range(rounds):
        treated_sample = rng.choice(treated_index, len(treated_index), replace=True)
        control_sample = rng.choice(control_index, len(control_index), replace=True)
        values[index] = outcome[treated_sample].mean() - outcome[control_sample].mean()
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def power_and_mde(
    sd: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float]:
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("sd must be finite and positive")
    if not isinstance(n, int) or n < 4:
        raise ValueError("n must be an integer of at least four")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if not np.isfinite(power) or not 0.5 < power < 1:
        raise ValueError("power must be between 0.5 and one")
    treatment_n = n // 2
    control_n = n - treatment_n
    standard_error_multiplier = math.sqrt(1 / treatment_n + 1 / control_n)
    return {
        "alpha": float(alpha),
        "target_power": float(power),
        "mde_outcome_units": float(
            (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd * standard_error_multiplier
        ),
    }


def run(n: int, seed: int, effect: float, output_dir: Path) -> dict:
    df = simulate(n, seed, effect)
    y = df.outcome.to_numpy()
    treatment = df.treatment.to_numpy()
    pre = df.pre_metric.to_numpy()

    raw = mean_effect(y, treatment)
    adjusted, theta = cuped_adjust(y, pre)
    cuped = mean_effect(adjusted, treatment)
    bootstrap_ci = stratified_bootstrap(adjusted, treatment, 1500, seed + 99)
    variance_reduction = 1 - float(np.var(adjusted, ddof=1)) / float(np.var(y, ddof=1))
    guardrail = mean_effect(df.guardrail.to_numpy(), treatment)
    decision = "ship" if cuped[1] > 0 and guardrail[1] > -0.20 else "hold"

    payload = {
        "project": "ExperimentLab",
        "verification_pass": bool(abs(cuped[0] - effect) < 1.0 and variance_reduction > 0.30),
        "scope": "deterministic synthetic randomized-experiment methodology demo",
        "rows": n,
        "known_simulated_effect": effect,
        "treatment_rows": int((treatment == 1).sum()),
        "control_rows": int((treatment == 0).sum()),
        "raw_effect": {"estimate": raw[0], "ci95": [raw[1], raw[2]]},
        "cuped_effect": {
            "estimate": cuped[0],
            "ci95": [cuped[1], cuped[2]],
            "bootstrap_ci95": list(bootstrap_ci),
            "theta": theta,
        },
        "variance_reduction": variance_reduction,
        "guardrail_effect": {
            "estimate": guardrail[0],
            "ci95": [guardrail[1], guardrail[2]],
            "non_inferiority_margin": -0.20,
        },
        "power": power_and_mde(float(np.std(adjusted, ddof=1)), n),
        "decision": decision,
        "rules": [
            "balanced random assignment",
            "fixed-horizon inference",
            "CUPED uses a pre-treatment covariate",
            "bootstrap resamples within treatment arms",
            "guardrail must not cross the pre-declared harm margin",
        ],
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination / "experiment_data.csv", index=False)
    (destination / "verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def self_test() -> None:
    result = run(12000, 42, 2.5, Path("/tmp/experimentlab_selftest"))
    assert result["verification_pass"], result
    assert result["variance_reduction"] > 0.30
    assert result["cuped_effect"]["ci95"][0] < 2.5 < result["cuped_effect"]["ci95"][1]
    print("ExperimentLab self-test passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ExperimentLab randomized-experiment demo.")
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--effect", type=float, default=2.5)
    parser.add_argument("--output-dir", type=Path, default=Path("experimentlab_artifacts"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result = run(args.rows, args.seed, args.effect, args.output_dir)
    return 0 if result["verification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
