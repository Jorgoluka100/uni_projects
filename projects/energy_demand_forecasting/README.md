# Energy Demand Forecasting with TensorFlow

This project forecasts the next **14 days** of German electricity consumption from the previous **60 days** of demand history.

I kept the time order intact, fitted preprocessing on training data only and compared the neural model with a proper 7-day seasonal baseline before accepting the result.

## Result

The final test contains **645 future windows** beginning on 14 March 2016.

| Model | MAE (GWh) | RMSE (GWh) | MAPE |
| --- | ---: | ---: | ---: |
| Last value | 146.62 | 197.13 | 11.23% |
| 7-day seasonal baseline | 53.18 | 93.39 | 3.96% |
| Conv1D + LSTM | **43.51** | **71.31** | **3.26%** |

The TensorFlow model improves MAE by **18.17%** compared with the stronger seasonal baseline.

## Data and split

The source is the Open Power System Data German daily time series with **4,383 consecutive days** from 2006 to 2017.

```text
2006 -------------------------------- 2014 | 2014 -------- 2016 | 2016 ---------------- 2017
                  TRAIN                   |    VALIDATION     |           TEST
               2,995 windows              |   644 windows     |        645 windows
```

The scaler is fitted only on the training period. Training windows are not shuffled.

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

Training uses Adam, MAE loss, early stopping and validation-based learning-rate reduction.

## Why the baseline matters

Electricity demand has a strong weekly pattern, so comparing a neural network only with a global mean would not tell me much.

The 7-day seasonal forecast already gets **53.18 GWh** MAE. The LSTM only counts as an improvement because it beats that stronger reference on the untouched future period.

## Forecast intervals

I use absolute validation residuals to set a separate error radius for each forecast horizon.

- nominal coverage: **90%**
- empirical test coverage: **88.37%**
- average interval width: **169.79 GWh**

The test coverage is slightly below the nominal target, so I leave that visible rather than describing the interval as better calibrated than it is.

## Saved-model check

The Keras model was saved and loaded again into a fresh object. Predictions on the check batch matched exactly, with maximum absolute difference **0.0**.

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

The lightweight checks do not need TensorFlow:

```bash
pip install -r requirements.txt
python run.py --self-test
python run.py --check-evidence
pytest -q
```

To retrain the neural network, also install `requirements-tensorflow.txt`.

The original executed notebook remains in the repository as the full training record:
[`05_Energy_Demand_Forecasting_with_TensorFlow.ipynb`](../../05_Energy_Demand_Forecasting_with_TensorFlow.ipynb)

## Limitations

The data ends in 2017, so I am not presenting this model as a current German grid forecast. A real forecasting system would need current licensed data, weather and calendar inputs, rolling backtests, drift monitoring and ongoing interval-coverage checks.

See [`MODEL_CARD.md`](MODEL_CARD.md).