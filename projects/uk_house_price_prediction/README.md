# UK House Price Analysis & Prediction

A forward-looking residential price model built from official HM Land Registry Price Paid Data for England and Wales.

The project is designed around a simple question: **does a machine-learning model add value beyond a strong local-property median when predicting future registered sale prices?**

## Result

On an untouched January–June 2026 test set of **216,564 sales**:

| Model | MAE | R² | Within 20% |
| --- | ---: | ---: | ---: |
| Global training median | £152,553 | -0.035 | 28.2% |
| Postcode-district + property-type median | £82,804 | 0.604 | 56.0% |
| CatBoost | **£81,805** | **0.604** | **56.1%** |

CatBoost improves MAE by **1.21%** over the strong baseline. That is a real but modest gain, and the project reports it as such.

## Why this project is useful

A weak portfolio project would compare a complex model only with a global average and declare success. This project instead:

- uses a genuine time-based holdout;
- builds a postcode-district/property-type baseline using training data only;
- avoids exact-address exposure;
- keeps 2026 registrations explicitly labelled provisional;
- evaluates error by property and geography;
- adds a validation-derived residual interval rather than presenting a point estimate as certainty;
- verifies saved-model reload parity.

## Data

Source: **HM Land Registry Price Paid Data**.

Raw snapshot:

- 1,184,740 registered transactions;
- 995,059 accepted into the modelling population;
- 595,617 train rows;
- 182,878 validation rows;
- 216,564 untouched 2026 test rows.

The modelling population keeps ordinary residential types, Category A transactions, complete postcodes and prices from £20,000 to £5,000,000.

## Temporal design

```text
Jan 2025 ---------------- Sep 2025 | Oct 2025 ----- Dec 2025 | Jan 2026 -------- Jun 2026
              TRAIN               |       VALIDATION         |        TEST
```

The test period is not used for model fitting, early stopping, baseline construction or interval selection.

## Features

Categorical features:

- postcode district and postcode area;
- property type;
- old/new indicator;
- tenure/duration;
- town/city;
- district;
- county.

Calendar features:

- year and month;
- cyclical month sine/cosine.

## Model

The model is CatBoost regression trained on `log1p(price)` so very high-value properties have less influence on the optimisation. CatBoost is useful here because the data contains high-cardinality categorical geography.

## Uncertainty

A 90th-percentile absolute validation residual is used as a simple distribution-free error radius. On the untouched test set the nominal 90% interval achieved **91.6% coverage**, with an average width of roughly **£381,679**.

The width is itself an important finding: the available registry features are not sufficient for precise individual-property valuation.

## Structure

```text
projects/uk_house_price_prediction/
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
└── requirements.txt
```

## Checks

```bash
python run.py --self-test
python run.py --check-evidence
pytest -q
```

The original executed notebook remains in the repository as the full exploration and training record:

[`01_UK_House_Price_Analysis_and_Prediction.ipynb`](../../01_UK_House_Price_Analysis_and_Prediction.ipynb)

## Limitations

The dataset does not contain bedrooms, floor area, condition, renovation quality or many property-specific characteristics. This is therefore broad valuation support, not a surveyor valuation, mortgage decision or investment recommendation.

See [`MODEL_CARD.md`](MODEL_CARD.md) for the full scope and limitations.
