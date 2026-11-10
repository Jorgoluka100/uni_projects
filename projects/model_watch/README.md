# ModelWatch — Drift, Calibration & Model-Risk Monitoring

**Decision question:** when should a deployed classifier trigger investigation or retraining rather than continue silently under distribution shift?

## Monitoring workflow

The deterministic simulation trains a reference classifier and evaluates sequential production-like batches:

- stable reference-like data;
- mild covariate shift;
- stronger feature shift with a changed target relationship;
- concept shift.

For each batch, the pipeline reports:

- **Feature drift:** Population Stability Index (PSI) and two-sample Kolmogorov–Smirnov statistics.
- **Discrimination:** ROC-AUC and PR-AUC, compared with the reference performance.
- **Probability quality:** Brier score and expected calibration error (ECE).
- **Subgroups:** row counts, prevalence, Brier score and mean predicted score by segment.
- **Operational status:** green / amber / red severity with explicit trigger reasons.
- **Artifact reliability:** save/reload parity check for the trained model.

## Alert policy

The current simulation uses transparent example thresholds:

| Signal | Amber | Red |
|---|---:|---:|
| Maximum feature PSI | ≥ 0.10 | ≥ 0.25 |
| ROC-AUC decline vs reference | ≥ 0.04 | ≥ 0.08 |
| ECE | ≥ 0.05 | ≥ 0.08 |

The highest severity triggered by any signal determines the batch alert. These are demonstration thresholds, not universal industry standards; a real deployment should calibrate them to the model, data volume, risk tolerance and operating context.

## Run

From the repository root:

```bash
python -m pip install numpy pandas scipy scikit-learn joblib
python projects/model_watch/run.py --self-test
python projects/model_watch/run.py --output-dir modelwatch_artifacts
```

Outputs include `verification.json`, `batch_summary.csv` and `model.joblib`. The reusable monitoring analysis helpers live in `src/analysis.py`.

## Limitations and safe operation

This is a deterministic monitoring simulation, not a live production service. PSI and KS identify distribution changes; neither proves model failure or establishes causality. A red state recommends investigation and a retraining candidate review—it deliberately does **not** auto-deploy a replacement model. Human approval, data-quality checks, fresh holdout evaluation, rollback and monitoring of the replacement are required for a production lifecycle.
