# Model card — flight delay risk classifier

## Intended use

Rank scheduled U.S. passenger flights by the probability of arriving at least 15 minutes late, using information known before departure. The model is designed as a decision-support example for prioritising limited operational review capacity.

## Model

CatBoost binary classifier with native handling of categorical features. The alert threshold is selected on the April 2026 validation period to match a configurable review capacity; the May 2026 test period remains untouched until final evaluation.

## Primary metrics

The project prioritises **PR-AUC** because delayed flights are the minority class, then checks ROC-AUC, log loss, Brier score, precision, recall, calibration and risk concentration at several operational capacity levels.

## Retained portfolio result

On the retained May 2026 test sample of 180,000 flights:

- PR-AUC: **0.291** vs **0.215** delay prevalence baseline
- ROC-AUC: **0.618**
- Highest-risk 10% of flights: **34.1%** delayed vs **21.5%** overall, a **1.58x lift**
- Validation-selected alert policy: approximately 20% target review capacity

These figures describe the retained run in `results/verified_test_metrics.json`; they should be regenerated before making new claims after changing modelling code or data.

## Known limitations

- Completed, non-diverted flights only.
- No weather or live operational features.
- Performance may drift as schedules, carriers and airport conditions change.
- A risk score is not a guarantee that an individual flight will be delayed.
- Carrier-level slices are retained for inspection, but subgroup behaviour should be re-evaluated before deployment.
