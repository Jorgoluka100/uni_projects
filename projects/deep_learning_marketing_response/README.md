# Deep Learning Marketing Response — Neural Network Decision Application

A standalone PyTorch project that trains, validates and persists a neural network for marketing-response prediction using the UCI Bank Marketing dataset.

## Decision problem

A marketing team needs to prioritise customers for outreach while controlling contact volume. The model estimates response probability, compares a neural network with a classical baseline, and converts scores into a configurable targeting policy.

## Dataset

UCI Bank Marketing (`bank-additional-full.csv`), a public real-world dataset of direct-marketing campaign contacts. The project downloads the archive reproducibly and keeps the original target definition. It is used strictly as a portfolio/educational decision-support benchmark.

## What this project demonstrates

- real public-data ingestion
- schema/data-quality checks
- categorical + numeric preprocessing
- leakage review
- train/validation/test separation
- logistic-regression baseline
- PyTorch Dataset/DataLoader pipeline
- multi-layer neural network
- batch normalisation and dropout
- class imbalance handling
- AdamW optimisation
- learning-rate scheduling
- early stopping
- ROC-AUC, PR-AUC, F1, Brier score and log loss
- threshold selection under outreach constraints
- calibration/error slices
- saved preprocessing + model state
- inference for one customer
- tests and retained metrics

## Run

```bash
python run.py
```

## Portfolio files

- `project_notebook.ipynb` — full end-to-end notebook
- `run.py` — training and inference application
- `tests/test_neural_network.py` — data/model tests
- `results/` — metrics and training history
- `artifacts/` — saved preprocessor and neural-network checkpoint

## Limitations

The historical campaign data comes from a specific bank and period. The model should not be used to make financial eligibility decisions. A production marketing system would require current consent rules, fairness/privacy review, rolling validation, calibrated contact costs and live drift monitoring.
