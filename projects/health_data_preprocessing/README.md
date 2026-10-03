# Health Data Cleaning & Preprocessing

A focused Python portfolio project: turn deliberately messy health observations into auditable, machine-learning-ready features. All bundled observations are **synthetic**, generated with seed 42. They describe no real people and support no medical conclusions.

## Run locally

From this folder, with Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
python -m unittest -v
```

On Windows, activate with `.venv\Scripts\activate` instead.

For your own compatible CSV:

```bash
python run.py --input observations.csv --output-dir outputs
```

## What you learn

1. Validate a schema before modifying data.
2. Remove exact duplicates; reject conflicting observation IDs rather than silently choosing one.
3. Parse numeric strings and distinguish missing, unparseable and out-of-range values.
4. Normalise activity categories; retain unknowns as missing values.
5. Split observations **before fitting** median imputation, scaling and category encoding.
6. Fit preprocessing only on training data and apply those same statistics to test data.
7. Export clean data, aligned features and a JSON quality report.

`record_id` identifies an observation and is excluded from features. This demo assumes one independent observation per person. Repeated patient records would require a patient-level split; forecasting would require a temporal split.

## Input schema and transparent rules

| Column | Rule |
| --- | --- |
| record_id | Nonempty, unique observation identifier |
| age | Numeric, 18–100 years |
| sleep_hours | Numeric, 0–24 hours |
| steps | Numeric, 0–100,000 steps per day |
| resting_hr | Numeric, 30–220 beats per minute |
| activity | low, moderate, high; case and whitespace normalised |

These ranges are illustrative quality rules, not clinical standards. Invalid numeric values become missing; valid extremes are retained. Median imputation and standardisation learn from training rows. Missing numeric indicators are added for columns with training missingness; unknown activity becomes an explicit category. Unseen test categories are accepted by the encoder.

## Retained evidence

`results/` contains the actual demo run: raw and cleaned CSVs, train/test feature tables and `quality_report.json`. Rerunning writes to `outputs/`. There is no predictive model or medical assessment in this project: this is the preprocessing stage on which a later neural network could build.

Tests check known corruptions, idempotency, conflicting IDs, training-only statistics, unseen categories and end-to-end execution.

## Limitations and next steps

Synthetic patterns do not establish performance on real health data. The pipeline intentionally uses a fixed schema rather than guessing units or repairing ambiguous dates. Next steps: add unit-aware rules, visualise before/after distributions, then use a properly sourced dataset and define a target before training a model. Never include a future target in the feature columns.
