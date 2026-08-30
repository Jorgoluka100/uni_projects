# Hiring Portfolio — Foundation → Intermediate → Advanced

This repository is deliberately structured as a progression rather than a flat list of projects.

The target audience is **internship, graduate and junior hiring** across Data Science, Data Engineering, Analytics / BI and ML / AI Engineering.

## Why the portfolio is structured this way

My earlier background was outside STEM before completing an **MSc in Artificial Intelligence & Data Science with Distinction**. I therefore do not rely on the degree title alone to make the transition credible. The repository is designed so a reviewer can inspect the evidence in sequence: fundamentals first, then end-to-end projects, then stronger engineering and release controls.

The aim is to make the transition measurable: **can I clean data, reason with SQL, build and evaluate models, communicate through BI, and turn selected work into tested software?** Each layer below exists to answer one of those questions directly.

## 1. Foundation — can I work with data and explain the basics?

The [`skills/`](../skills/) lab contains compact notebook + Python pairs that are easy to inspect in an interview.

### Data handling

- data cleaning and preprocessing
- missing values and duplicates
- type conversion and dates
- NumPy/vectorisation
- SQL joins, aggregation, CTEs and window functions

### Core machine learning

- regression with a baseline
- classification with a baseline
- clustering and model-selection diagnostics
- train/test separation
- preprocessing pipelines
- meaningful metrics rather than accuracy alone

### Deep learning and unstructured data

- PyTorch neural-network fundamentals
- LSTM / sequence modelling
- TF-IDF text classification
- CNN image fundamentals

These exercises are intentionally small. They establish that the concepts are understood before the same skills are demonstrated on larger real datasets.

## 2. Intermediate — can I build reliable end-to-end work and communicate it?

### Data Science

- [`retail_customer_segmentation`](../projects/retail_customer_segmentation/) — 541,909 raw rows, auditable cleaning, RFM features, clustering diagnostics and tests
- [`customer_churn_prediction`](../projects/customer_churn_prediction/) — grouped holdout, calibration, cost-aware thresholding and bootstrap uncertainty
- [`uk_house_price_prediction`](../projects/uk_house_price_prediction/) — official 2025–2026 data, temporal holdout, strong baseline and CatBoost
- [`experiment_lab`](../projects/experiment_lab/) — treatment effects, CUPED, confidence intervals, guardrails and power

### Business Intelligence / Analytics

- [`executive_commerce_bi`](../projects/executive_commerce_bi/) — Power BI + Tableau over a governed shared KPI layer; PBIP/PBIR, TMDL/DAX, Tableau workbook source, dashboard storytelling, data-contract tests and CI
- [`ecommerce_sql_analytics`](../projects/ecommerce_sql_analytics/) — explicit semantic grain, reconciliation, cohorts, windows and join-safety tests over 98,199 commercial orders

The BI project is intentionally designed as a bridge between technical analysis and commercial communication. It makes the same verified metrics usable as executive dashboards rather than stopping at notebooks or SQL output.

### Data / Analytics Engineering

- [`reliable_event_pipeline`](../projects/reliable_event_pipeline/) — schema validation, reject handling, deduplication, late data, idempotency, SQL checks and tests
- [`pyspark_clickstream_analytics`](../projects/pyspark_clickstream_analytics/) — distributed transformations over real event data and a one-million-row load test
- [`ecommerce_sql_analytics/dbt_project`](../projects/ecommerce_sql_analytics/dbt_project/) — dbt sources, staging/intermediate/mart models, relationship tests, custom data tests and CI

### ML / AI Engineering

- [`grounded_rag`](../projects/grounded_rag/) — retrieval evaluation, citations, abstention, prompt-injection checks, FastAPI and Docker
- [`image_classification_confidence`](../projects/image_classification_confidence/) — PyTorch/EfficientNet, uncertainty, Grad-CAM and export verification
- [`energy_demand_forecasting`](../projects/energy_demand_forecasting/) — TensorFlow time series, seasonal baseline, interval coverage and saved-model reload checks
- [`model_watch`](../projects/model_watch/) — drift, discrimination and calibration monitoring

## 3. Advanced — can I combine modelling and engineering into a system?

### Advanced flagship: Flight Delay Risk Platform

[`projects/flight_delay_risk/`](../projects/flight_delay_risk/)

The core model uses official **2026 U.S. Bureau of Transportation Statistics** data with a chronological train/validation/test split and an untouched 180,000-flight May test set.

The advanced layer adds:

```text
official monthly data
      ↓
data/schema checks
      ↓
leakage-safe schedule features
      ↓
CatBoost training + temporal validation
      ↓
validation-selected review threshold
      ↓
untouched out-of-time test
      ↓
verified model + release metadata
      ↓
FastAPI single/batch inference
      ↓
Docker service image
      ↓
CI tests + container build
```

The serving layer reconstructs the exact training feature schema from schedule-time inputs and refuses to load a release unless the accompanying metadata has `verification_pass=true`.

That project is intended to be the clearest example of the full progression: **data → ML → evaluation → release controls → API → container → CI**.

## What this progression is meant to prove

A recruiter should be able to move through the repository and answer, in order:

1. **Can this candidate clean and manipulate data?**
2. **Can they write SQL and reason about data grain?**
3. **Can they turn governed data into Power BI / Tableau reporting and define business KPIs clearly?**
4. **Can they build regression, classification and clustering models?**
5. **Can they work with neural networks, sequences, text and images?**
6. **Can they test pipelines and models properly?**
7. **Can they package an AI/ML system behind an API and container?**
8. **Can they explain limitations and avoid overstating results?**

The portfolio is therefore intentionally not a collection of unrelated "AI apps". Each layer exists to provide a different hiring signal.
