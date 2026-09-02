# Linear Regression — Building Energy Efficiency Decision Model

A standalone data-science portfolio application built to make **ordinary linear regression** visible as a real modelling skill rather than a one-line classroom example.

## Business / decision problem

Building-design teams need an early way to estimate heating load before committing to a design. The project uses physical building characteristics to estimate **Heating Load** and asks a practical question:

> Which candidate designs are likely to require unusually high heating demand, and which design variables are associated with that demand?

The model is used as a **screening and decision-support tool**, not as an engineering certification calculation.

## Dataset

**UCI Energy Efficiency dataset (ID 242)**

- 768 simulated building configurations
- 8 input variables
- 2 continuous targets: Heating Load and Cooling Load
- no missing values in the published dataset
- source: UCI Machine Learning Repository
- DOI: `10.24432/C51307`
- licence: CC BY 4.0

The project models **Heating Load** as the primary target and retains Cooling Load for exploratory comparison.

### Input features

| Feature | Meaning |
| --- | --- |
| Relative Compactness | building compactness |
| Surface Area | total surface area |
| Wall Area | wall area |
| Roof Area | roof area |
| Overall Height | building height |
| Orientation | orientation category |
| Glazing Area | glazing proportion |
| Glazing Area Distribution | glazing distribution category |

## Why this project exists

Linear regression is one of the first techniques expected in junior Data Scientist, Data Analyst and graduate ML interviews, but simply importing `LinearRegression` proves very little. This project shows the surrounding work that makes regression useful:

- data provenance and schema checks
- exploratory analysis
- visualisation
- train/test separation
- a dummy baseline
- ordinary least squares Linear Regression
- Ridge and Lasso comparison
- polynomial/interaction alternative
- cross-validation
- residual analysis
- error slices
- coefficient interpretation
- bootstrap uncertainty
- scenario analysis
- a simple decision policy
- retained machine-readable evidence

## Notebook-first structure

Open [`project_notebook.ipynb`](project_notebook.ipynb).

The notebook is designed to show the actual analysis directly. It leads with data inspection, plots and decisions; modular/source code is retained later for engineering inspection rather than replacing the notebook analysis.

## Models

1. **Median DummyRegressor** — sanity baseline.
2. **LinearRegression** — transparent ordinary least-squares baseline and the main learning focus.
3. **RidgeCV** — tests whether coefficient shrinkage improves generalisation.
4. **LassoCV** — tests sparse regularisation.
5. **PolynomialFeatures + RidgeCV** — tests whether simple interactions/non-linearity materially improve performance.

The final recommendation is based on holdout and cross-validation evidence. The project does not assume that the most complicated model must win.

## Evaluation

Primary metrics:

- MAE
- RMSE
- R²
- MAPE
- cross-validated RMSE

Diagnostics:

- actual vs predicted
- residual distribution
- residuals vs predicted values
- error by orientation
- error by glazing-area band
- worst prediction cases
- coefficient magnitude
- learning curve
- bootstrap uncertainty

## Decision layer

The project converts predictions into a simple design-screening policy:

- **LOW / NORMAL LOAD** — predicted heating load below the training-set 75th percentile
- **HIGH LOAD — REVIEW** — predicted heating load at or above that threshold
- a prediction interval is attached so uncertainty is not hidden

This is intentionally a screening rule, not a building-code or safety decision.

## Run

```bash
cd projects/linear_regression_energy_efficiency
python -m pip install -r requirements.txt
python run.py
```

Outputs are written to `results/`:

- `metrics.json`
- `predictions.csv`
- `coefficients.csv`
- `data_audit.csv`
- `scenario_analysis.csv`
- `bootstrap_summary.json`
- visualisations as PNG files

## Interview discussion

A reviewer should be able to ask:

- Why is a dummy baseline necessary?
- Why encode Orientation and Glazing Area Distribution as categorical?
- Why compare OLS with regularised models?
- What does R² fail to tell us?
- What pattern would make a residual plot concerning?
- How do correlated physical features affect coefficient interpretation?
- Why does a better polynomial model not automatically make the linear model useless?
- What is the difference between prediction and causal inference?
- How would this change with real measured building-energy data?

## Limitations

The dataset contains simulated building configurations rather than a representative sample of all real buildings. The model therefore demonstrates regression methodology and transparent decision support; it does not establish performance on real-world construction stock. A production version would need external measured data, geography/climate features, stronger uncertainty modelling, temporal/site validation and domain review.
