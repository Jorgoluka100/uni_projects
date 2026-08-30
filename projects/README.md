# Data & AI Projects — Intermediate & Advanced

The **foundation layer** lives in [`../skills/`](../skills/). This folder starts where the small exercises stop: real datasets, realistic validation, SQL/data engineering, testing, APIs, model monitoring and deployment-oriented work.

**[See the full Foundation → Intermediate → Advanced roadmap →](../docs/HIRING_PORTFOLIO.md)**

## Advanced flagship

### [Flight Delay Risk Platform](flight_delay_risk/)

The strongest end-to-end system in the portfolio:

```text
official 2026 BTS data
    ↓
data quality + leakage-safe features
    ↓
CatBoost + chronological validation
    ↓
untouched 180,000-flight test set
    ↓
verified release metadata
    ↓
FastAPI single / batch inference
    ↓
Docker + CI
```

This is the project to start with for **ML / AI Engineering** or a more engineering-heavy Data Science role.

## Intermediate — start by role

| Target role | Best projects | Evidence |
| --- | --- | --- |
| **Data Scientist** | [Customer Churn](customer_churn_prediction/) · [UK House Prices](uk_house_price_prediction/) · [Parkinson's Progression](parkinsons_progression/) · [Retail Segmentation](retail_customer_segmentation/) | real-data cleaning, feature engineering, baselines, leakage controls, grouped/temporal validation, calibration, uncertainty and clustering |
| **Data Engineer / Analytics Engineer** | [Reliable Event Pipeline](reliable_event_pipeline/) · [PySpark Clickstream](pyspark_clickstream_analytics/) · [E-commerce SQL + dbt](ecommerce_sql_analytics/) | ingestion, schema validation, idempotency, distributed transformations, semantic grain, dbt models and data-quality tests |
| **ML / AI Engineer** | [Grounded RAG](grounded_rag/) · [Image Classification](image_classification_confidence/) · [Energy Forecasting](energy_demand_forecasting/) · [ModelWatch](model_watch/) | retrieval, PyTorch, TensorFlow, APIs, Docker, model export, uncertainty and drift/calibration monitoring |
| **Data Analyst / Product Analyst** | [E-commerce SQL + dbt](ecommerce_sql_analytics/) · [ExperimentLab](experiment_lab/) · [Retail Segmentation](retail_customer_segmentation/) | SQL, KPI logic, cohorts, experimentation, customer analysis and business interpretation |

## Data handling first

### [Retail Customer Data Cleaning & Segmentation](retail_customer_segmentation/)

The strongest proof that I can work with messy data before modelling. The project begins with **541,909 raw transaction rows**, audits duplicates, missing customer identifiers, cancellations and invalid quantity/price values, applies explicit cleaning rules, validates the final transaction table and only then builds customer-level RFM features and clustering.

## Data / analytics engineering

### [Reliable Event Pipeline](reliable_event_pipeline/)

A compact engineering pipeline covering schema validation, reject handling, duplicate control, late-arriving data, idempotent loads, SQL reconciliation, audit metrics and automated tests.

**Verified demo:** 10 input rows → 7 clean warehouse events, with invalid and duplicate records handled explicitly.

### [E-commerce SQL + dbt](ecommerce_sql_analytics/)

End-to-end analytics over **98,199 commercial orders**, including explicit relational grain, reconciliation, cohorts, window functions and join-safety checks. The project now includes a **dbt + DuckDB analytics-engineering layer** with staging/intermediate/mart models, relationships, custom data tests and CI.

### [PySpark Clickstream Analytics](pyspark_clickstream_analytics/)

Distributed transformations and data-quality logic over **165,474 real events**, plus a separate one-million-row load test.

## Applied machine learning

### [Customer Churn Prediction](customer_churn_prediction/)

Classification with a protected grouped holdout, out-of-fold probability calibration, cost-aware threshold selection and bootstrap uncertainty.

### [UK House Price Prediction](uk_house_price_prediction/)

Official HM Land Registry data with a genuine time-based 2026 holdout, a strong local/property baseline and CatBoost regression.

### [Parkinson's Progression](parkinsons_progression/)

Regression with explicit feature policy, meaningful baselines, patient-grouped holdout separation and GroupKFold validation so repeated measurements from one person cannot leak across train and evaluation data.

### [ExperimentLab](experiment_lab/)

Experimentation covering treatment effects, confidence intervals, CUPED, guardrails and statistical power.

## Deep learning & applied AI

### [Grounded RAG](grounded_rag/)

Hybrid retrieval, source attribution, abstention on weak evidence, prompt-injection checks, read-only tool routing, FastAPI and Docker. A frozen evaluation fixture makes retrieval and routing behaviour testable rather than demo-only.

### [Image Classification with Confidence](image_classification_confidence/)

PyTorch / EfficientNet-B0 with uncertainty checks, selective prediction, Grad-CAM and verified model export.

**Retained test accuracy:** **85.9%**.

### [Energy Demand Forecasting](energy_demand_forecasting/)

TensorFlow Conv1D + LSTM time-series forecasting with chronological splitting, a strong weekly seasonal baseline, uncertainty intervals and saved-model reload checks.

### [ModelWatch](model_watch/)

Monitoring checks for data drift, discrimination and calibration against a reference dataset, with deliberately introduced shifts used to test whether monitoring rules react correctly.

## What employers should be able to inspect

Across the stronger projects I make the following explicit rather than hiding them behind a final metric:

- where the data came from and what was wrong with it
- how missing values, duplicates, joins or invalid records were handled
- why the train / validation / test strategy matches the problem
- what leakage or unrealistic evaluation would look like
- which baseline the model or method must beat
- how the result is measured and retained
- what tests or checks protect the implementation
- how an engineered project is packaged or served where relevant
- what the project does **not** claim to prove

The intended progression is therefore:

```text
skills/       → fundamentals I can explain
projects/     → real-data intermediate evidence
flight_delay  → advanced end-to-end flagship
```