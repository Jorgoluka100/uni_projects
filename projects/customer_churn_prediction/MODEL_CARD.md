# Model card — customer churn screening

## Purpose

Prioritise customers for **human retention review** using behaviour observed during the first nine months of the UCI Iranian Churn dataset. The outcome is churn measured at the end of month 12, leaving a three-month planning gap.

## Data

- UCI Iranian Churn, DOI `10.24432/C5JW3Z`, CC BY 4.0.
- 3,150 customer records.
- 15.71% observed churn rate.
- Source archive and extracted CSV are SHA-256 pinned.

The file has no customer ID or calendar event timestamp. A true time split would therefore be fabricated. Instead, identical operational predictor profiles are grouped and kept on one side of the train/holdout boundary.

## Protected evaluation

- Training: 2,522 rows / 2,229 unique predictor profiles.
- Holdout: 628 rows / 565 unique predictor profiles.
- Holdout is not used for algorithm choice, calibration fitting or decision-threshold selection.

## Model

Grouped cross-validation selected histogram gradient boosting. Probability calibration is fitted to grouped out-of-fold training predictions rather than in-sample predictions.

`Status` is deliberately excluded from the operational model. A sensitivity check increased average precision by 0.0219 when it was included, but its semantics make it a risky outcome/proxy feature. `Age` and `Age Group` are retained only for subgroup monitoring, not prediction.

## Decision policy

The notebook uses transparent scenario weights:

- 25 cost units for contacting/reviewing a flagged customer;
- 200 cost units for missing a later churner.

The training-only cost calculation selected a probability threshold of **0.15**. These units are illustrative and are not claimed currency, revenue or realised savings.

## Holdout evidence

- Average precision: **0.955** (bootstrap 95% interval 0.927–0.980).
- ROC-AUC: **0.990** (0.982–0.996).
- Recall: **94.9%** (89.8%–99.0%).
- Precision: **73.2%** (66.9%–80.2%).
- Brier score: **0.0259** (0.0173–0.0351).
- Review/alert rate: **20.2%**.

## Limitations

The dataset is small and comes from one telecom context. It is not evidence of live commercial performance. No causal claim is made about why customers churn, including the finding that `complains` has the largest holdout permutation importance.

A real deployment would need fresh local data, monitored calibration and drift, a measured retention-treatment effect, capacity constraints, subgroup review and an experimentally validated intervention policy.

## Not for

- automatic adverse treatment;
- pricing or eligibility decisions;
- claiming individual customers will certainly churn;
- claiming the illustrative cost weights are financial savings.
