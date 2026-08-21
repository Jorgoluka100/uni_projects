# Customer Churn Prediction & Retention Screening

A leakage-aware churn project that treats prediction as a **decision problem**, not an accuracy leaderboard.

The model ranks customers for human retention review, calibrates its probabilities from grouped out-of-fold predictions and selects an intervention threshold using explicit scenario costs. The final holdout stays untouched until those choices are fixed.

## Holdout result

The protected holdout contains **628 customers** and 98 churners.

| Metric | Result | Bootstrap 95% interval |
| --- | ---: | ---: |
| PR-AUC / average precision | **0.955** | 0.927–0.980 |
| ROC-AUC | **0.990** | 0.982–0.996 |
| Recall at selected threshold | **94.9%** | 89.8%–99.0% |
| Precision at selected threshold | **73.2%** | 66.9%–80.2% |
| Brier score | **0.0259** | 0.0173–0.0351 |

At the training-selected **0.15** threshold, **20.2%** of holdout customers are flagged for review.

## What makes the evaluation safer

### 1. Grouped split instead of pretending there is a timestamp

The UCI file contains 3,150 records but no customer ID or event timestamp. Predictors describe months 1–9 and churn is measured at the end of month 12.

Identical operational predictor profiles are therefore grouped with `StratifiedGroupKFold` so duplicate profiles cannot appear on both sides of the holdout boundary:

- training: 2,522 rows / 2,229 predictor profiles;
- holdout: 628 rows / 565 predictor profiles;
- churn prevalence: 15.74% train / 15.61% holdout.

### 2. Proxy-risk fields are not used to win a metric

`Status` improved validation average precision by **0.0219** in a sensitivity test, but it stays out of the deployed feature set because its meaning makes it a potential target/proxy feature.

`Age` and `Age Group` are also excluded from prediction and retained only for subgroup checks.

### 3. Calibration is training-only

Grouped out-of-fold predictions from the selected histogram-gradient-boosting model are passed through a logistic calibration layer. The final base model is then fitted to all training rows. The holdout does not fit the calibrator.

### 4. The threshold has an explicit decision meaning

The notebook uses an illustrative scenario:

- 25 units for contacting/reviewing a flagged customer;
- 200 units for missing a later churner.

The training-only optimisation selected threshold **0.15**, reducing the scenario cost from 79,400 units under no intervention to 17,800 units on the training decision exercise. These are **not currency or claimed savings**; a company would replace them with measured economics from a retention experiment.

## Model inputs

Operational inputs:

- call failures and complaints;
- subscription length and charge amount;
- seconds/frequency of use and SMS frequency;
- distinct called numbers;
- tariff plan;
- customer value.

Derived ratios include call-failure rate, seconds/SMS per call, contact diversity, customer value per month and usage per month.

## Reproducibility

The UCI source is pinned with SHA-256 hashes for both the archive and extracted CSV. The package includes:

- source download/fingerprint checks;
- grouped holdout logic;
- feature engineering;
- the selected histogram-gradient-boosting configuration;
- grouped OOF calibration;
- cost-aware threshold/evaluation functions;
- unit tests and retained evidence checks.

```bash
python run.py --self-test
python run.py --check-evidence
pytest -q
```

## Structure

```text
projects/customer_churn_prediction/
├── README.md
├── MODEL_CARD.md
├── run.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   └── evaluation.py
├── tests/
│   └── test_evaluation.py
├── results/
│   └── verified_metrics.json
└── requirements.txt
```

The original executed notebook remains the full training and analysis record:
[`03_Customer_Churn_Prediction.ipynb`](../../03_Customer_Churn_Prediction.ipynb)

## Scope

This is a screening prototype for retention review. It does not prove why someone churned and should not automatically trigger adverse treatment. See [`MODEL_CARD.md`](MODEL_CARD.md).
