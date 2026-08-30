# Data & AI Foundations Lab

This folder is the **foundation layer** of the portfolio: small, interview-friendly projects that prove I understand the building blocks before the larger end-to-end systems.

Each focused skill has both formats:

- `.ipynb` for explanation, revision and Jupyter/Colab review
- `.py` for normal Python execution and code review

These are deliberately compact. The larger projects under [`../projects/`](../projects/) show the same skills on real datasets with stronger validation, testing and engineering.

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
| 08 | [Regression fundamentals](08_regression_fundamentals.ipynb) · [Python](08_regression_fundamentals.py) | baseline regression, mixed-type preprocessing, regularisation, MAE and R² |
| 03 | [Classification fundamentals](03_sklearn_end_to_end_classification.ipynb) · [Python](03_sklearn_end_to_end_classification.py) | train/test split, preprocessing pipeline, dummy baseline, logistic regression and classification metrics |
| 09 | [Clustering fundamentals](09_clustering_fundamentals.ipynb) · [Python](09_clustering_fundamentals.py) | scaling, KMeans, candidate-k comparison, silhouette score and cluster interpretation |

**Real-data follow-ons:** [Flight Delay Risk](../projects/flight_delay_risk/), [Customer Churn](../projects/customer_churn_prediction/), [UK House Prices](../projects/uk_house_price_prediction/) and [Retail Segmentation](../projects/retail_customer_segmentation/).

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
FOUNDATIONS
small, explainable notebook + Python exercises
        ↓
INTERMEDIATE
real datasets + SQL + pipelines + tests + APIs + CI
        ↓
ADVANCED
end-to-end systems that combine modelling and engineering
```

The goal is not to pretend a ten-minute learning exercise is production software. The goal is to make the fundamentals easy to inspect, then prove them again at a higher standard in the main portfolio.
