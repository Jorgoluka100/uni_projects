# Customer Churn Prediction & Retention Screening

I built this around a practical retention question: **which customers should a team review first?** The point is not to chase accuracy. The model has to rank customers well, produce usable probabilities and make the review threshold clear.

The final holdout is only scored after the model choice, calibration and threshold are fixed.

## Holdout result

The holdout contains **628 customers**, including 98 churners.

| Metric | Result | Bootstrap 95% interval |
| --- | ---: | ---: |
| PR-AUC / average precision | **0.955** | 0.927–0.980 |
| ROC-AUC | **0.990** | 0.982–0.996 |
| Recall at selected threshold | **94.9%** | 89.8%–99.0% |
| Precision at selected threshold | **73.2%** | 66.9%–80.2% |
| Brier score | **0.0259** | 0.0173–0.0351 |

At the training-selected threshold of **0.15**, **20.2%** of holdout customers are sent for review.

## How I handled the split

The UCI dataset has 3,150 records, but it does not provide a customer ID or event timestamp. The predictors describe months 1–9 and churn is measured at the end of month 12.

Because of that, I do not pretend this is a true calendar-time split. I group identical predictor profiles with `StratifiedGroupKFold` so the same profile cannot appear on both sides of the holdout boundary.

- training: 2,522 rows / 2,229 predictor profiles
- holdout: 628 rows / 565 predictor profiles
- churn prevalence: 15.74% train / 15.61% holdout

## Features I chose not to use

`Status` improved validation average precision by **0.0219** in a sensitivity check, but I left it out of the operational feature set because its meaning creates a possible target/proxy risk.

`Age` and `Age Group` are also excluded from prediction. I keep them only for subgroup checks.

## Probability calibration

The selected histogram-gradient-boosting model produces grouped out-of-fold probabilities on the training data. I then fit a logistic calibration layer to those predictions before fitting the final base model on all training rows.

The holdout is not used to fit the calibrator.

## Choosing the review threshold

To make the threshold concrete, the notebook uses an illustrative cost setup:

- 25 units for contacting or reviewing a flagged customer
- 200 units for missing a customer who later churns

This selects a threshold of **0.15** on training data. In that training exercise, scenario cost falls from 79,400 units under no intervention to 17,800 units.

Those numbers are not currency and they are not claimed business savings. A real company would replace them with measured contact cost and retention value.

## Model inputs

The operational model uses:

- call failures and complaints
- subscription length and charge amount
- seconds and frequency of use
- SMS frequency
- distinct called numbers
- tariff plan
- customer value

I also derive call-failure rate, seconds/SMS per call, contact diversity, customer value per month and usage per month.

## Reproducibility

The UCI archive and extracted CSV are both pinned with SHA-256 hashes. The project includes the source checks, grouped split, feature engineering, model configuration, out-of-fold calibration, threshold logic, unit tests and saved result file.

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

The original executed notebook is still available as the full analysis record:
[`03_Customer_Churn_Prediction.ipynb`](../../03_Customer_Churn_Prediction.ipynb)

## Scope

This is a screening prototype for human retention review. It does not explain why a customer churned and should not automatically trigger adverse treatment. See [`MODEL_CARD.md`](MODEL_CARD.md).