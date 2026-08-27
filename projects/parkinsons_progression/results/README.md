# Evaluation results

`run.py` writes `latest_metrics.json` here after a real grouped-validation run.

The JSON is intentionally generated rather than pre-filled. Any headline metric used in the portfolio should come from a reproducible run on the UCI Parkinson's Telemonitoring data, with subjects separated between training and holdout sets.

Expected report sections:

- data-cleaning counts
- selected feature policy
- train / holdout row and subject counts
- explicit subject-overlap check
- grouped cross-validation results for each model
- selected model
- untouched holdout metrics (MAE, RMSE and R²)
