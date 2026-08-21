# Energy Demand Forecasting with TensorFlow

A 14-day German electricity-consumption forecasting project built around **chronological validation, strong seasonal baselines and uncertainty**, rather than a randomly shuffled time-series split.

## Result

On **645 untouched future test windows** beginning 2016-03-14:

| Model | MAE (GWh) | RMSE (GWh) | MAPE |
| --- | ---: | ---: | ---: |
| Last value | 146.62 | 197.13 | 11.23% |
| 7-day seasonal baseline | 53.18 | 93.39 | 3.96% |
| Conv1D + LSTM | **43.51** | **71.31** | **3.26%** |

The TensorFlow model reduces MAE by **18.17%** relative to the strongest baseline.

## Forecasting design

Source: Open Power System Data Germany daily time series, **4,383 consecutive days** from 2006 through 2017.

Each example uses:

- previous **60 days** of consumption;
- next **14 days** as the multi-output target.

The data remains chronological:

```text
2006 -------------------------------- 2014 | 2014 -------- 2016 | 2016 ---------------- 2017
                  TRAIN                   |    VALIDATION     |           TEST
               2,995 windows              |   644 windows     |        645 windows
```

The scaler is fitted on training data only. Training windows are never shuffled.

## Model

```text
60-day history
    ↓
causal Conv1D (32 filters)
    ↓
LSTM (48 units)
    ↓
Dense (48)
    ↓
14 daily forecasts
```

Training uses Adam, MAE loss, early stopping and validation-driven learning-rate reduction.

## Baselines matter

Electricity consumption is strongly weekly-seasonal. Comparing only with a naive global mean would make the neural model look better than it is.

The project therefore requires the LSTM to beat a **7-day seasonal persistence forecast**, which already reaches 53.18 GWh test MAE.

## Uncertainty

Absolute residuals from the validation period calibrate a separate error radius for each forecast horizon.

- nominal coverage: **90%**;
- empirical test coverage: **88.37%**;
- average interval width: **169.79 GWh**.

The slight under-coverage is visible in the evidence rather than being rounded into a false pass.

## Artifact reliability

The trained model was saved in Keras format and reloaded in a fresh object. Sample predictions reproduced with maximum absolute difference **0.0**.

## Project structure

```text
projects/energy_demand_forecasting/
├── README.md
├── MODEL_CARD.md
├── run.py
├── src/
│   ├── data.py
│   ├── evaluation.py
│   └── model.py
├── tests/
│   └── test_evaluation.py
├── results/
│   └── verified_metrics.json
├── requirements.txt
└── requirements-tensorflow.txt
```

Lightweight checks do not require TensorFlow:

```bash
pip install -r requirements.txt
python run.py --self-test
python run.py --check-evidence
pytest -q
```

To retrain the neural network, also install `requirements-tensorflow.txt`.

The original executed notebook remains the full training record:
[`05_Energy_Demand_Forecasting_with_TensorFlow.ipynb`](../../05_Energy_Demand_Forecasting_with_TensorFlow.ipynb)

## Limitations

The dataset ends in 2017, so the retained model is not presented as a current German grid forecast. A real system should use current licensed data, weather/calendar/price covariates, rolling-origin backtests, drift checks and interval-coverage monitoring.

See [`MODEL_CARD.md`](MODEL_CARD.md).
