# Statistical Marketing Mix — Decision Science Application

A standalone statistical-modelling project for extracting interpretable commercial insights from weekly marketing, pricing and demand data and translating uncertainty into budget decisions.

## Decision problem

A commercial team needs to understand which channels are associated with incremental sales, quantify uncertainty around those effects, detect unstable conclusions, and compare budget-allocation scenarios rather than relying on a black-box forecast alone.

## Dataset

The project generates an explicitly synthetic weekly marketing dataset with known data-generating relationships, seasonality, trend, price, promotions, competitor pressure and three media channels. Synthetic data is deliberate here: it allows coefficient recovery, statistical assumptions and decision logic to be tested against known ground truth.

## What this project demonstrates

- reproducible data generation with known ground truth
- descriptive statistics and correlations
- OLS regression
- robust HC3 standard errors
- log transformations for diminishing returns
- interaction terms
- hypothesis tests and confidence intervals
- multicollinearity checks with VIF
- residual diagnostics
- heteroskedasticity testing
- autocorrelation diagnostics
- bootstrap coefficient uncertainty
- out-of-time holdout evaluation
- scenario simulation
- budget-response curves
- decision recommendations with uncertainty caveats
- tests and retained evidence

## Run

```bash
python run.py
```

## Portfolio files

- `project_notebook.ipynb` — recruiter-facing statistical analysis
- `run.py` — reproducible statistical decision application
- `tests/test_statistics.py` — data-generation and modelling tests
- `results/` — coefficients, diagnostics and scenarios

## Limitations

Regression associations are not automatically causal effects. The synthetic generator gives known truth for portfolio validation, but a real marketing-mix model would require careful causal identification, adstock/carryover design, external controls, incrementality experiments where possible, longer history and explicit treatment of endogeneity.
