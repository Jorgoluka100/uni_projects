# ExperimentLab — Experiment Decision Engine

**Decision:** should a randomized product change ship when the primary metric, uncertainty, variance reduction, guardrails and statistical power all matter?

The project simulates a randomized experiment with a known effect, compares raw difference-in-means with **CUPED**, adds a stratified bootstrap interval, checks a pre-declared guardrail non-inferiority margin, calculates minimum detectable effect and emits a machine-readable ship/hold decision.

```bash
pip install numpy pandas scipy
python projects/experiment_lab/run.py --self-test
python projects/experiment_lab/run.py --output-dir experimentlab_artifacts
```

Outputs: `experiment_data.csv` and `verification.json`.

**Evidence boundary:** data are synthetic by design so the true treatment effect is known. The project demonstrates experimentation methodology and decision discipline, not a real product uplift claim.
