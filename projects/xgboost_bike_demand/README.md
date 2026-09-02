# XGBoost Bike Demand — Capacity Planning Application

A standalone gradient-boosting portfolio project that forecasts hourly bike-rental demand and turns model output into an operational capacity-planning signal.

## Decision problem

A mobility operator needs to estimate near-term hourly demand so it can rebalance bikes, allocate staff and identify periods where capacity is likely to be constrained.

## Dataset

UCI Bike Sharing Dataset (`hour.csv`), originally covering Capital Bikeshare demand with weather, calendar and seasonal variables. The loader downloads the public UCI archive reproducibly and caches it locally.

## What this project demonstrates

- reproducible public-data acquisition
- schema and range validation
- time-aware feature engineering
- leakage controls around the target and component counts
- chronological train/validation/test splitting
- naive seasonal baseline
- XGBoost regression
- hyperparameter search
- MAE, RMSE, RMSLE and R²
- segment-level error analysis
- permutation / model feature importance
- demand-band operational decisions
- uncertainty proxy from validation residuals
- model persistence and inference
- tests and machine-readable results

## Run

```bash
python run.py
```

## Portfolio files

- `project_notebook.ipynb` — full recruiter-facing notebook
- `run.py` — executable training/application pipeline
- `tests/test_xgboost.py` — feature and split tests
- `results/` — metrics, error slices and predictions
- `artifacts/` — saved XGBoost model

## Limitations

The data is historical and location-specific. It does not include live station inventory, road closures or current mobility patterns. A production service should combine current station telemetry, weather forecasts and recent demand, validate on rolling time windows and monitor drift before using forecasts for operations.
