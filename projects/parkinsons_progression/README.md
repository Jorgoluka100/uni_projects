# Parkinson's Telemonitoring Progression — from MSc notebook to leakage-aware ML

This project is the strengthened version of my original university notebook, [`Parkinsons_Progression_ML.ipynb`](../../Parkinsons_Progression_ML.ipynb).

I have deliberately **kept the original executed notebook unchanged**. It shows the work I originally completed using the UCI Parkinson's Telemonitoring dataset. This folder shows how I would approach the same problem now with clearer data checks, leakage controls, model baselines, grouped validation and reproducible output.

> **Portfolio / educational use only.** This is a regression exercise on a public research dataset. It is not a diagnostic model, medical device or clinical decision-support system.

## What this project proves

A recruiter can inspect one small project and see the core workflow rather than just a technology name on a CV:

1. **Data loading** — local CSV or the official UCI source.
2. **Schema validation** — fail clearly if required Parkinson's telemonitoring fields are missing.
3. **Data cleaning** — remove exact duplicates; remove rows with no subject or target; retain feature-level missing values for pipeline-safe imputation.
4. **Feature policy** — `subject#` is never a model feature. `motor_UPDRS` is excluded by default so the main experiment focuses on demographic, test-time and voice measurements.
5. **Leakage-aware splitting** — hold out entire subjects with `GroupShuffleSplit`.
6. **Training-only cross-validation** — compare models with `GroupKFold`, again keeping subjects separated.
7. **Meaningful baseline** — compare against a median `DummyRegressor` before judging ML models.
8. **Model comparison** — Ridge regression gives an interpretable linear baseline; Gradient Boosting captures nonlinear relationships.
9. **Evaluation** — MAE, RMSE and R² on grouped CV and the untouched subject holdout.
10. **Reproducibility** — save the full experiment definition and metrics as JSON.

## Why the validation changed

The telemonitoring dataset contains repeated measurements from the same people. A normal random row split can place observations from one subject in both training and test data. That makes the test set less representative of the real question: **does the model generalise to a patient it has not already seen?**

The original notebook is useful evidence of my university work, but this version uses the patient identifier only as a grouping key. The final holdout therefore has **zero subject overlap** with training data. Model selection happens only inside the training subjects before the chosen model is evaluated on the untouched holdout.

This is the main improvement I would discuss in an interview: model quality is not just the algorithm or a high R²; the validation design has to match the intended generalisation problem.

## Dataset

**UCI Parkinson's Telemonitoring** contains biomedical voice measurements collected remotely from people with early-stage Parkinson's disease. The regression target used here is `total_UPDRS`.

The original notebook loads the dataset from the UCI Machine Learning Repository. The runner below accepts either that URL or a local copy of the CSV.

## Run it

From the repository root:

```bash
python -m pip install -r projects/parkinsons_progression/requirements.txt
python projects/parkinsons_progression/run.py
```

To use a downloaded copy instead:

```bash
python projects/parkinsons_progression/run.py \
  --data data/parkinsons_updrs.data
```

The default run writes:

```text
projects/parkinsons_progression/results/latest_metrics.json
```

I do **not** hard-code a new headline score into this README before the grouped-validation run has produced it. The JSON report is intended to be the evidence source for any future metric claim.

## Experiment design

```text
raw UCI data
   │
   ├── schema checks
   ├── exact-duplicate removal
   └── rows without subject/target removed
            │
            ▼
     subject-level 80/20 holdout
        │                │
        │                └── untouched subjects
        ▼
   training subjects
        │
        ├── GroupKFold
        │      ├── Dummy median baseline
        │      ├── Ridge
        │      └── Gradient Boosting
        │
        └── select candidate by mean CV RMSE
                    │
                    ▼
          fit on all training subjects
                    │
                    ▼
          evaluate untouched subjects
                    │
                    ▼
          MAE / RMSE / R² → JSON
```

## Important modelling choices

### Why `subject#` is excluded

It identifies the person rather than a transferable biological signal. It is used to enforce grouped validation, never as a predictor.

### Why `motor_UPDRS` is excluded by default

`motor_UPDRS` is another UPDRS assessment and is closely related to the `total_UPDRS` target. Including it can answer a less interesting question than predicting from the demographic/test-time/voice measurements. The runner provides `--include-motor-updrs` only so that the effect can be investigated explicitly rather than hidden in the feature set.

### Why imputation happens inside the pipeline

The original notebook removed missing rows. Here, feature values are median-imputed **after splitting**, within each fitted scikit-learn pipeline. This avoids using information from validation or holdout data to determine preprocessing values.

## Files

```text
projects/parkinsons_progression/
├── README.md
├── requirements.txt
├── run.py
└── results/
    └── README.md
```

## Original work vs strengthened version

| Original MSc notebook | Strengthened portfolio version |
| --- | --- |
| executed exploratory notebook retained exactly as evidence | compact reproducible Python runner |
| duplicate / missing-row cleaning | explicit schema + cleaning report + pipeline imputation |
| random row train/test split | subject-level train/holdout separation |
| regression modelling and tuning | baseline + linear + nonlinear model comparison |
| metric output in notebook cells | machine-readable JSON evidence |
| useful university learning artifact | clearer explanation of leakage, generalisation and limitations |

The point is not to pretend the original project was something it was not. The point is to show **progression in how I reason about data science**.
