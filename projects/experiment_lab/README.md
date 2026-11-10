# ExperimentLab — Experiment Decision Engine

**Decision question:** should a randomized product change ship when the primary metric, uncertainty, variance reduction, guardrails and statistical power all matter?

## What the experiment does

A deterministic synthetic experiment is generated with a known treatment effect. The pipeline then:

1. creates randomized treatment/control arms and a pre-treatment covariate;
2. checks numeric and categorical randomisation balance using standardized differences;
3. estimates the raw difference in means with a 95% confidence interval;
4. applies CUPED using only the pre-treatment covariate;
5. estimates uncertainty with a stratified bootstrap that resamples within treatment arms;
6. evaluates a pre-declared guardrail non-inferiority margin;
7. calculates a minimum detectable effect and target power;
8. emits an explicit `ship` or `hold` decision plus machine-readable verification output.

## Why the design matters

- **CUPED:** reduces outcome variance using pre-experiment information; it does not use post-treatment variables.
- **Fixed-horizon inference:** keeps the example aligned with a pre-specified analysis rather than repeatedly checking significance.
- **Bootstrap:** preserves treatment/control arm sizes while estimating an empirical interval.
- **Guardrail:** a positive primary effect is not enough if a secondary metric crosses its harm margin.
- **Known simulated effect:** lets the self-test check whether the method recovers a controlled effect. It does not establish a real-world product uplift.

## Run

From the repository root:

```bash
python -m pip install numpy pandas scipy
python projects/experiment_lab/run.py --self-test
python projects/experiment_lab/run.py --rows 20000 --seed 42 --effect 2.5 --output-dir experimentlab_artifacts
```

Outputs include `experiment_data.csv` and `verification.json`. The implementation is in `run.py` and `src/diagnostics.py`; the notebook provides the interactive analysis.

## Limitations

The data are synthetic by design. The example demonstrates statistical workflow and decision discipline, not an observed product experiment, business uplift or a universal ship rule. Real experiments also require a valid randomisation unit, sample-ratio-mismatch checks, pre-registration, multiple-testing policy, exposure logging and domain-specific guardrails.
