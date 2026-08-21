# Model card — UK house price prediction

## Purpose

Broad residential sale-price estimation for portfolio demonstration and human-reviewed market analysis using HM Land Registry Price Paid Data.

## Data

- England and Wales registered sales.
- 2025 plus January–June 2026 snapshot.
- Ordinary residential property types only: detached, semi-detached, terraced and flats.
- Category A standard transactions.
- £20,000–£5,000,000 price range.
- Complete postcode required.

## Temporal evaluation

- Train: January–September 2025.
- Validation: October–December 2025.
- Untouched test: January–June 2026.

The test split is later in time than all model fitting and threshold/interval choices.

## Features

Only information available from the registry record is used: postcode district/area, property type, old/new indicator, tenure, town/city, district, county and calendar features. Exact addresses are not exposed in the project.

## Model

CatBoost regression on `log1p(price)`, with categorical handling built into CatBoost. Predictions are transformed back to pounds and clipped to the modelling range.

## Baselines

Two baselines are retained:

1. global training median;
2. postcode-district + property-type training median, with property-type and global fallback.

The second is deliberately strong because a model should beat a sensible geographical/property baseline rather than only a trivial constant.

## Verified test result

- Test rows: 216,564.
- CatBoost MAE: £81,805.
- Strong baseline MAE: £82,804.
- CatBoost R²: 0.604.
- MAE improvement over strong baseline: 1.21%.
- Nominal 90% residual interval achieved 91.6% coverage on the untouched test set.

## Important limitation

The improvement over the strong baseline is modest. That is reported directly rather than hidden. The Price Paid Data does not contain bedrooms, floor area, condition, renovation state, exact comparable-property attributes or many local micro-market variables. The model is therefore a broad market estimator, not a surveyor valuation.

2026 registrations are also provisional and can be revised as later registrations arrive.

## Not for

- mortgage underwriting;
- automated lending or investment decisions;
- surveying;
- individual financial advice;
- claiming a precise valuation for a specific property.
