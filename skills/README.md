# Data & AI Foundations Lab

This folder is the **DataCamp-style foundation layer** of the portfolio: compact, interview-friendly projects that prove one skill clearly from objective to result before the larger end-to-end systems.

These are deliberately **not** disguised as production applications. Each foundation should be easy to inspect in one sitting: objective → reproducible data/input → inspection/cleaning where needed → direct analysis/visualisation → method/model → metric/output → short conclusion.

Each focused skill has both formats:

- `.ipynb` for explanation, revision and Jupyter/Colab review
- `.py` for normal Python execution and code review

The larger projects under [`../projects/`](../projects/) are a different tier: full professional applications with real datasets, deeper EDA, stronger validation, robustness/error analysis, testing and role-relevant engineering.

**[Read the final three-tier portfolio standard →](../docs/PORTFOLIO_STANDARD.md)**

## Level 1 — Data foundations

| # | Project | What it proves |
| --- | --- | --- |
| 01 | [Data cleaning & preprocessing](01_data_cleaning_preprocessing.ipynb) · [Python](01_data_cleaning_preprocessing.py) | missing values, duplicates, type conversion, dates, encoding, scaling and leakage-safe preprocessing |
| 02 | [NumPy for machine learning](02_numpy_for_machine_learning.ipynb) · [Python](02_numpy_for_machine_learning.py) | arrays, shapes, broadcasting, vectorisation and matrix multiplication |
| 10 | [SQL analytics fundamentals](10_sql_analytics_fundamentals.ipynb) · [Python](10_sql_analytics_fundamentals.py) | joins, aggregation, CTEs, grouping and window functions |

**Real-data follow-on:** [Retail Customer Data Cleaning & Segmentation](../projects/retail_customer_segmentation/) processes **541,909 raw transaction rows** and makes the cleaning/validation rules auditable.

## Level 2 — Core machine learning

| # | Project | What it proves |
| --- | --- | --- |
| 08 | **[Linear Regression & Ridge fundamentals](08_regression_fundamentals.ipynb)** · [Python](08_regression_fundamentals.py) | ordinary least-squares `LinearRegression`, median baseline, mixed-type preprocessing, missing-value imputation, Ridge regularisation, MAE, RMSE, R², residual diagnostics and inference |
| 03 | [Classification fundamentals](03_sklearn_end_to_end_classification.ipynb) · [Python](03_sklearn_end_to_end_classification.py) | train/test split, preprocessing pipeline, dummy baseline, logistic regression and classification metrics |
| 09 | [Clustering fundamentals](09_clustering_fundamentals.ipynb) · [Python](09_clustering_fundamentals.py) | scaling, KMeans, candidate-k comparison, silhouette score and cluster interpretation |

**Real-data follow-ons:** [Flight Delay Risk](../projects/flight_delay_risk/), [Customer Churn](../projects/customer_churn_prediction/), [UK House Prices](../projects/uk_house_price_prediction/), [Statistical Marketing Mix](../projects/statistical_marketing_mix/) and [Retail Segmentation](../projects/retail_customer_segmentation/).

## Level 3 — Deep learning & unstructured data

| # | Project | What it proves |
| --- | --- | --- |
| 04 | [PyTorch neural-network fundamentals](04_pytorch_neural_network_fundamentals.ipynb) · [Python](04_pytorch_neural_network_fundamentals.py) | tensors, layers, forward pass, loss, backpropagation, optimiser and evaluation |
| 05 | [LSTM sequence modelling](05_lstm_sequence_modelling.ipynb) · [Python](05_lstm_sequence_modelling.py) | sequence shapes, hidden state and recurrent modelling |
| 06 | [Text classification with TF-IDF](06_text_classification_tfidf.ipynb) · [Python](06_text_classification_tfidf.py) | text preprocessing, sparse features, logistic regression and text evaluation |
| 07 | [CNN image fundamentals](07_cnn_image_fundamentals.ipynb) · [Python](07_cnn_image_fundamentals.py) | convolution, channels, pooling, flattening and a compact CNN training loop |

**Real-data / engineered follow-ons:** [Energy Demand Forecasting](../projects/energy_demand_forecasting/), [Image Classification with Confidence](../projects/image_classification_confidence/) and [Grounded RAG](../projects/grounded_rag/).

## Progression

```text
PROTECTED ORIGINALS
university/course work kept exactly as evidence
        ↓
FOUNDATIONS
DataCamp-style compact skill notebook + Python exercise
        ↓
PROFESSIONAL APPLICATIONS
end-to-end systems with real analysis, evaluation, decisions and engineering
```

The goal is not to pretend a focused learning exercise is production software. The goal is to make the fundamentals easy to inspect, then prove them again at a higher standard in the main portfolio.
