# Flight Delay Prediction & Risk Analysis

[![Flight delay project](https://github.com/Jorgoluka100/uni_projects/actions/workflows/flight-delay-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/flight-delay-ci.yml)

This project uses official 2026 U.S. Bureau of Transportation Statistics data to identify scheduled flights that are more likely to arrive **15+ minutes late**.

## Why I changed the original approach

I first tried to predict the exact number of delay minutes from schedule information. That regression version did not beat a simple baseline consistently, so I did not keep pushing the same idea just to produce a better-looking result.

I changed the question to something more useful for operations: **which flights should be looked at first if review capacity is limited?** That makes the project a classification and ranking problem rather than an exact-delay forecast.

## Test result

| Metric | May 2026 test result |
|---|---:|
| Test flights | 180,000 |
| Delay prevalence | 21.5% |
| PR-AUC | 0.291 |
| Constant-baseline PR-AUC | 0.215 |
| ROC-AUC | 0.618 |
| Precision at validation-selected alert threshold | 32.0% |
| Recall at validation-selected alert threshold | 27.9% |
| Top-10% risk lift | 1.58x |

The highest-scored 10% of flights had a **34.1%** delay rate, compared with **21.5%** across the full May test set. The saved numbers are in [`results/verified_test_metrics.json`](results/verified_test_metrics.json).

## Train, validation and test setup

```text
Jan–Mar 2026             Apr 2026                    May 2026
TRAIN -----------------> VALIDATE -----------------> TEST
fit model                 early stopping              final metrics
                          choose alert threshold       no tuning
```

I used a chronological split because a random split would make the evaluation less realistic. May stays out of model fitting and threshold selection.

## Leakage checks

The model only uses information that exists on the schedule, including carrier, route, airports, scheduled departure and arrival times, scheduled duration, distance and calendar features.

I exclude actual departure and arrival times, taxi information and delay-cause fields. Cancelled and diverted flights are also left out because they are separate prediction problems.

## What is in the code

- BTS monthly data download and local caching
- schema and data-quality checks
- deterministic temporal sampling
- cyclical time features and route/time-block interactions
- CatBoost categorical modelling
- validation-only alert-capacity threshold selection
- PR-AUC, ROC-AUC, Brier score, log loss, precision and recall
- calibration checks
- top-5/10/20/30/50% risk-lift analysis
- carrier-level slices
- saved-model reload check
- offline self-test and unit tests

## Project structure

```text
flight_delay_risk/
├── run.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── evaluate.py
│   └── pipeline.py
├── tests/
│   └── test_core.py
├── results/
│   └── verified_test_metrics.json
├── DATA_CARD.md
├── MODEL_CARD.md
└── requirements.txt
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python run.py --self-test
python -m unittest discover -s tests
python run.py --output-dir artifacts/flight_delay_risk
```

## Limitations

Schedule data on its own only explains part of flight-delay risk. Weather, aircraft rotations, crew, airport congestion and live operational information are not included here. The May 2026 holdout is useful evidence for this version of the model, but future performance can change as routes and operating conditions change.

This is decision support, not a guarantee that an individual flight will be delayed. See [`MODEL_CARD.md`](MODEL_CARD.md).