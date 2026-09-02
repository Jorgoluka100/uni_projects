# K-Nearest Neighbours — Product Quality Decision System

A standalone machine-learning portfolio project showing how distance-based models can support product-quality triage rather than serving as a toy KNN demonstration.

## Business problem

A quality team needs an interpretable way to classify product quality from measurable characteristics, understand the effect of feature scaling and neighbourhood size, and route uncertain cases for manual review.

## Dataset

This project uses scikit-learn's built-in Wine dataset so the project is reproducible without committing a large third-party raw file. The data contains 178 observations, 13 continuous chemical measurements and 3 target classes. The project treats the classes as product-quality categories for an educational decision-support use case.

## What this project demonstrates

- data loading and schema validation
- duplicate/missing-value checks
- exploratory analysis
- StandardScaler and leakage-safe Pipelines
- stratified train/test split
- KNN baseline and hyperparameter search
- distance vs uniform weighting
- cross-validation
- confusion matrix and classification metrics
- probability confidence and manual-review policy
- nearest-neighbour inspection for explainability
- sensitivity to scaling
- decision-oriented inference
- reusable Python application code
- tests and reproducible outputs

## Run

```bash
python run.py
```

## Portfolio files

- `project_notebook.ipynb` — recruiter-facing end-to-end notebook
- `run.py` — executable application pipeline
- `tests/test_knn.py` — core behavioural tests
- `results/metrics.json` — generated evaluation evidence after running

## Limitations

This is a compact benchmark dataset, so the project is intended to demonstrate rigorous KNN workflow design rather than claim production performance on a live manufacturing process. In a real deployment I would validate on time-separated factory batches, add cost-sensitive review thresholds, monitor feature drift and retrain only after an explicit quality gate.
