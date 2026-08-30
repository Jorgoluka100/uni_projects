# Flight Delay Risk Service Contract

The training pipeline and serving layer are intentionally separated.

## Release contract

The API will only load a model when both files are present:

- `flight_delay_catboost.cbm` — CatBoost model saved by the training pipeline
- `verification.json` — release metadata with `verification_pass=true`

Default locations are:

```text
artifacts/flight_delay_risk/flight_delay_catboost.cbm
artifacts/flight_delay_risk/verification.json
```

They can be overridden with:

- `FLIGHT_DELAY_MODEL_PATH`
- `FLIGHT_DELAY_METADATA_PATH`

If the release is unavailable, `/health` reports a degraded state and prediction endpoints return HTTP 503 rather than silently falling back to an unverified model.

## Request fields

The service accepts only schedule-time information:

- `flight_date`
- `carrier`
- `origin`
- `dest`
- `crs_dep_minutes`
- `crs_arr_minutes`
- `crs_elapsed_minutes`
- `distance_miles`

The service derives month, day-of-week, route, cyclical time encodings and route/time interactions using the same feature names and order expected by the trained classifier.

No actual departure, arrival, taxi or delay-cause information is accepted because those fields would leak post-schedule outcomes into the prediction.

## Endpoints

- `GET /health` — release readiness and model metadata
- `POST /predict` — one scheduled flight
- `POST /predict-batch` — up to 1,000 scheduled flights

The response includes the risk score and the validation-selected review threshold retained in the release metadata. `flag_for_review` is therefore tied to the same capacity-based threshold selected before the untouched May 2026 test set was evaluated.

## Container

```bash
docker build -t flight-delay-risk projects/flight_delay_risk
docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts:/app/artifacts" \
  flight-delay-risk
```

The image does not contain a trained model by default. Training artifacts are mounted or copied as a separate release step so model lineage stays explicit.
