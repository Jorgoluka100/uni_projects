# Flight Delay Risk Platform — Advanced Flagship

[![Flight delay project](https://github.com/Jorgoluka100/uni_projects/actions/workflows/flight-delay-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/flight-delay-ci.yml)

This is the **advanced flagship** in my portfolio: an end-to-end machine-learning system built around official **2026 U.S. Bureau of Transportation Statistics** data.

It starts as a realistic Data Science problem — rank scheduled flights by their risk of arriving **15+ minutes late** — and then carries the verified model into an engineering layer with release metadata, request validation, FastAPI, batch inference, Docker and CI.

## System flow

```text
BTS monthly flight data
        ↓
schema + data-quality checks
        ↓
leakage-safe schedule-time features
        ↓
CatBoost classifier
        ↓
chronological validation + threshold selection
        ↓
untouched May 2026 test set
        ↓
verified model + release metadata
        ↓
FastAPI single / batch inference
        ↓
Docker image
        ↓
unit tests + container-build CI
```

## Why I changed the original approach

I first tried to predict the exact number of delay minutes from schedule information. That regression version did not beat a simple baseline consistently, so I did not keep pushing the same idea just to produce a better-looking result.

I changed the question to something more useful for operations: **which flights should be looked at first if review capacity is limited?** That makes the project a classification and ranking problem rather than an exact-delay forecast.

## Untouched test result

| Metric | May 2026 test result |
|---|---:|
| Test flights | **180,000** |
| Delay prevalence | **21.5%** |
| PR-AUC | **0.291** |
| Constant-baseline PR-AUC | **0.215** |
| ROC-AUC | **0.618** |
| Precision at validation-selected alert threshold | **32.0%** |
| Recall at validation-selected alert threshold | **27.9%** |
| Top-10% risk lift | **1.58x** |

The highest-scored 10% of flights had a **34.1%** delay rate, compared with **21.5%** across the full May test set. The saved numbers are in [`results/verified_test_metrics.json`](results/verified_test_metrics.json).

The model is useful as a ranking signal but is not extremely predictive from schedule information alone. I keep that limitation visible rather than presenting moderate performance as a breakthrough.

## Train, validation and test design

```text
Jan–Mar 2026             Apr 2026                    May 2026
TRAIN -----------------> VALIDATE -----------------> TEST
fit model                 early stopping              final metrics
                          choose alert threshold       no tuning
```

A chronological split is used because a random split would make the evaluation less realistic. May stays out of model fitting and threshold selection.

## Leakage controls

The model only uses information that exists on the schedule, including carrier, route, airports, scheduled departure/arrival times, scheduled duration, distance and calendar features.

Actual departure/arrival times, taxi information and delay-cause fields are excluded. Cancelled and diverted flights are also excluded because they are separate prediction problems.

## Advanced serving layer

The API accepts only schedule-time inputs:

- flight date
- carrier
- origin / destination
- scheduled departure / arrival minutes
- scheduled elapsed time
- distance

[`src/inference.py`](src/inference.py) reconstructs the exact feature order expected by the training pipeline, including route, calendar, cyclical time and route/time interaction features.

The API will only load a release when both the model artifact and accompanying `verification.json` are present, and the metadata reports `verification_pass=true`. If that contract is not met, the service reports a degraded health state and returns HTTP 503 for prediction requests rather than silently using an unverified model.

See [`SERVICE_CONTRACT.md`](SERVICE_CONTRACT.md).

## API

After producing or mounting a verified model release:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict-batch` — up to 1,000 scheduled flights

Example request:

```json
{
  "flight_date": "2026-05-17",
  "carrier": "AA",
  "origin": "JFK",
  "dest": "LAX",
  "crs_dep_minutes": 480,
  "crs_arr_minutes": 690,
  "crs_elapsed_minutes": 390,
  "distance_miles": 2475
}
```

The response returns the risk score and the validation-selected review threshold saved with the release.

## Docker

```bash
docker build -t flight-delay-risk projects/flight_delay_risk

docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts:/app/artifacts" \
  flight-delay-risk
```

The image deliberately does not bundle a trained model by default. Model artifacts are produced or mounted as a separate release step so lineage is explicit.

## What is in the code

### Data / modelling

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
- saved-model reload verification

### Engineering

- release-gated artifact loading
- exact inference feature contract
- Pydantic request validation
- FastAPI single and batch inference
- health/readiness behaviour
- Docker service image
- offline self-test and unit tests
- GitHub Actions including container build

## Project structure

```text
flight_delay_risk/
├── api.py
├── Dockerfile
├── SERVICE_CONTRACT.md
├── run.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── inference.py
│   ├── model.py
│   ├── evaluate.py
│   └── pipeline.py
├── tests/
│   ├── test_core.py
│   └── test_inference.py
├── results/
│   └── verified_test_metrics.json
├── DATA_CARD.md
├── MODEL_CARD.md
└── requirements.txt
```

## Run the checks

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python run.py --self-test
python -m unittest discover -s tests -v
python -m compileall -q run.py api.py src tests
```

To retrain and produce a release:

```bash
python run.py --output-dir artifacts/flight_delay_risk
```

## Limitations

Schedule data on its own only explains part of flight-delay risk. Weather, aircraft rotations, crew, airport congestion and live operational information are not included here. The May 2026 holdout is useful evidence for this version of the model, but future performance can change as routes and operating conditions change.

This is operational decision support, not a guarantee that an individual flight will be delayed. A production deployment would also need authentication, rate limiting, observability, managed artifact storage, live drift checks and a documented retraining/release process.

See [`MODEL_CARD.md`](MODEL_CARD.md) and [`SERVICE_CONTRACT.md`](SERVICE_CONTRACT.md).