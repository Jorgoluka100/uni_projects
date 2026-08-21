# Flight Delay Prediction & Risk Analysis

[![Flight delay project](https://github.com/Jorgoluka100/uni_projects/actions/workflows/flight-delay-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/flight-delay-ci.yml)

A leakage-safe machine-learning project that predicts which scheduled U.S. flights are at elevated risk of arriving **15+ minutes late** using official 2026 Bureau of Transportation Statistics data.

## Why this project exists

The useful business question is not “can I guess the exact number of delay minutes from the timetable?” An earlier regression experiment showed that schedule-only features were not strong enough to beat a simple MAE baseline consistently. Rather than hide that negative result, I reframed the task around a decision an operations team could actually make: **which flights should receive limited review attention before departure?**

That change turned the project into a ranking and risk-classification problem with a measurable operational capacity constraint.

## Retained result

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

The highest-scored 10% of flights had a **34.1%** observed delay rate versus **21.5%** across the full test population. Metrics come from the retained evidence file in `results/verified_test_metrics.json`.

## Evaluation design

```text
Jan–Mar 2026             Apr 2026                    May 2026
TRAIN -----------------> VALIDATE -----------------> TEST
fit model                 early stopping              final metrics
                          choose alert threshold       never used for tuning
```

This is deliberately temporal rather than a random split. The final test month is not used for model selection or threshold selection.

## Leakage controls

The model only sees information available from the schedule: carrier, route, origin, destination, scheduled departure/arrival time, scheduled duration, distance and calendar-derived features.

Actual departure/arrival times, taxi information and delay-cause fields are excluded. Cancelled and diverted flights are also excluded and treated as separate modelling problems.

## What the code adds

- official BTS monthly download with local caching
- schema and data-quality checks
- deterministic temporal sampling
- cyclic time features and route/time-block interactions
- CatBoost categorical modelling
- validation-only capacity threshold selection
- PR-AUC, ROC-AUC, Brier score, log loss, precision and recall
- calibration diagnostics
- top-5/10/20/30/50% capacity-lift curve
- carrier-level risk slices
- saved-model reload parity check
- fast offline self-test and unit tests

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

The full run downloads the requested BTS monthly archives on first use and caches them under `data/bts_cache/`.

## Outputs

A full run writes:

- `verification.json` — metrics, split policy, configuration and release checks
- `flight_delay_catboost.cbm` — saved CatBoost model
- `capacity_curve.csv` — risk concentration at different review capacities
- `calibration_table.csv` — predicted vs observed risk by probability bin
- `carrier_risk_slices.csv` — carrier-level prevalence, score and alert-rate checks

## What I would improve next

The clearest next gain would come from adding information that is genuinely available before or close to departure: weather forecasts, airport congestion, inbound-aircraft rotation and live network conditions. I would also monitor calibration and ranking lift over time instead of assuming the May 2026 relationship remains stable.

## Evidence policy

The retained metrics are evidence from a specific 2026 run, not generic claims about production airline performance. If modelling code, features, data or the evaluation window changes, the evidence should be regenerated before the README or CV is updated.
