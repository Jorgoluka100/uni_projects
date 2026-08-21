# Model card — German electricity demand forecasting

## Purpose

Forecast the next **14 days** of German daily electricity consumption for portfolio demonstration and capacity-planning decision support.

## Data

- Open Power System Data Germany daily time series.
- 4,383 consecutive daily observations from 2006-01-01 through 2017-12-31.
- Target: daily electricity consumption in GWh.

The scaler is fitted only on the training period. Each sample uses the previous 60 days to predict the following 14 days.

## Evaluation design

The source is kept in chronological order:

- first 70%: training;
- next 15%: validation and early stopping;
- final 15%: untouched future test period.

This produces 2,995 training windows, 644 validation windows and 645 test windows. The test forecast start is 2016-03-14 and the final source date is 2017-12-31.

## Model

A compact TensorFlow sequence model:

1. causal Conv1D layer;
2. 48-unit LSTM;
3. dense hidden layer;
4. 14-output forecast head.

The model is trained with MAE loss, no shuffled time windows, early stopping and validation-based learning-rate reduction.

## Baselines

The neural model must beat both:

- last observed value repeated across the horizon;
- 7-day seasonal persistence.

The weekly seasonal baseline is the stronger comparator.

## Verified test evidence

- TensorFlow MAE: **43.51 GWh**.
- Weekly seasonal MAE: **53.18 GWh**.
- Improvement: **18.17%**.
- TensorFlow RMSE: **71.31 GWh**.
- TensorFlow MAPE: **3.26%**.

A 90% per-horizon interval is calibrated from validation absolute residuals. It achieved **88.37%** empirical test coverage with average width **169.79 GWh**. The under-coverage is reported rather than hidden.

The saved Keras artifact reproduced sample predictions with maximum absolute reload delta **0.0**.

## Limitations

The source ends in 2017 and does not represent today's electricity system. The model uses consumption history only and omits weather forecasts, prices, holidays, policy changes, structural demand shifts and supply constraints.

A production forecast would require rolling-origin backtests, newer licensed data, external covariates, residual/coverage monitoring and scheduled retraining.

## Not for

- automatic grid control;
- safety-critical dispatch;
- current German demand planning without retraining on recent data;
- interpreting the calibrated interval as a guaranteed bound.
