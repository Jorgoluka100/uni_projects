# UK House Price Analysis & Prediction

This project uses HM Land Registry Price Paid Data to test a straightforward question: **does a machine-learning model improve on a strong local property-price baseline when predicting later sales?**

## Result

The final test set contains **216,564 sales** from January to June 2026.

| Model | MAE | R² | Within 20% |
| --- | ---: | ---: | ---: |
| Global training median | £152,553 | -0.035 | 28.2% |
| Postcode-district + property-type median | £82,804 | 0.604 | 56.0% |
| CatBoost | **£81,805** | **0.604** | **56.1%** |

CatBoost improves MAE by **1.21%** over the postcode/property baseline. That is a small improvement, so I report it as a small improvement rather than overselling it.

## Why I used a stronger baseline

Comparing CatBoost only with a global average would make the model look much better than it really is. Property prices are heavily tied to location and property type, so the more useful comparison is a median built from postcode district and property type using training data only.

The project also uses:

- a genuine time-based holdout
- no exact-address feature
- explicit labelling of provisional 2026 registrations
- error breakdowns by property and geography
- a validation-based residual interval
- saved-model reload checks

## Data

Source: **HM Land Registry Price Paid Data**.

- 1,184,740 registered transactions in the raw snapshot
- 995,059 rows in the modelling population
- 595,617 training rows
- 182,878 validation rows
- 216,564 untouched 2026 test rows

The modelling population keeps ordinary residential property types, Category A transactions, complete postcodes and prices between £20,000 and £5,000,000.

## Time split

```text
Jan 2025 ---------------- Sep 2025 | Oct 2025 ----- Dec 2025 | Jan 2026 -------- Jun 2026
              TRAIN               |       VALIDATION         |        TEST
```

The test period is not used for model fitting, early stopping, baseline construction or interval selection.

## Features

Categorical features:

- postcode district and postcode area
- property type
- old/new indicator
- tenure/duration
- town/city
- district
- county

Calendar features:

- year and month
- cyclical month sine/cosine

## Model

The model is CatBoost regression trained on `log1p(price)`. I use the log target to reduce the influence of very high-value properties during optimisation. CatBoost is a good fit for the high-cardinality geography in this dataset.

## Uncertainty

I use the 90th percentile of absolute validation residuals as a simple error radius. On the 2026 test set, the nominal 90% interval achieved **91.6% coverage**, with an average width of roughly **£381,679**.

That width is useful information in itself: the registry fields alone are not enough for precise individual-property valuation.

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

The original executed notebook is still in the repository as the full exploration and training record:
[`01_UK_House_Price_Analysis_and_Prediction.ipynb`](../../01_UK_House_Price_Analysis_and_Prediction.ipynb)

## Limitations

The dataset does not include bedrooms, floor area, property condition, renovation quality or many other details that matter to valuation. This is broad modelling support, not a surveyor valuation, mortgage decision or investment recommendation.

See [`MODEL_CARD.md`](MODEL_CARD.md).