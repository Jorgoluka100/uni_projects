# End-to-end Data & AI projects

These projects are the recruiter-facing part of the portfolio. **Every project must stand on its own as a complete portfolio piece**: a recruiter should be able to open one project folder and understand the problem, data, implementation, evaluation, results, limitations and reproduction path without depending on another project.

**[Open the Notebook + Dataset index →](../docs/NOTEBOOKS_AND_DATASETS.md)** — direct `.ipynb` links and the dataset/source for every project.

## Portfolio standard

For this repository, a project is not considered complete because it has a long notebook. Each project/application should contain, where technically relevant:

1. A clear business, analytical or engineering problem and success criteria.
2. Dataset provenance, schema/quality checks and a reproducible data route.
3. Cleaning, preprocessing, EDA and feature/data transformations.
4. A substantive implementation in `project_notebook.ipynb` plus canonical Python/SQL source.
5. Baselines and multiple modelling/analytical approaches when that genuinely improves the project.
6. Leakage-aware validation, metrics, error analysis and uncertainty where appropriate.
7. Explainability, decision logic or stakeholder interpretation where appropriate.
8. Application/engineering depth such as inference, APIs, pipelines, testing, monitoring, dbt/SQL or deployment patterns when relevant to the role.
9. Retained results/evidence rather than unsupported performance claims.
10. Limitations, risks, next steps and a reproducible run path.

The notebook depth target is roughly **800–1,200 meaningful lines of code, ideally around 1,000**, but line count is never used as permission to add filler. Smaller projects must gain genuine functionality before being described as major portfolio projects.

## Choose the route closest to the role

| Hiring route | Start here | Then inspect |
| --- | --- | --- |
| **Data Engineer / Analytics Engineer** | [Reliable Event Pipeline](reliable_event_pipeline/) | [Apache Spark Retail Intelligence](apache_spark_retail_intelligence/) · [PySpark Clickstream](pyspark_clickstream_analytics/) · [E-commerce SQL + dbt](ecommerce_sql_analytics/) |
| **Data Scientist** | [Flight Delay Risk](flight_delay_risk/) | [XGBoost Bike Demand](xgboost_bike_demand/) · [Customer Churn](customer_churn_prediction/) · [Statistical Marketing Mix](statistical_marketing_mix/) · [KNN Product Quality](knn_product_quality/) |
| **ML / AI Engineer** | [Grounded RAG](grounded_rag/) | [Deep Learning Marketing Response](deep_learning_marketing_response/) · [NLP Document Intelligence](nlp_document_intelligence/) · [Image Classification](image_classification_confidence/) · [Energy Forecasting](energy_demand_forecasting/) |
| **Data Analyst / BI / Product Analyst** | [Executive Commerce Intelligence](executive_commerce_bi/) | [Statistical Marketing Mix](statistical_marketing_mix/) · [E-commerce SQL + dbt](ecommerce_sql_analytics/) · [ExperimentLab](experiment_lab/) |

## Project map

| Project | Primary hiring signal | Inspectable evidence |
| --- | --- | --- |
| **[Flight Delay Risk Platform](flight_delay_risk/)** | Data Science · ML Engineering | Official 2026 BTS data, chronological validation, untouched 180,000-flight test set, model card, FastAPI, Docker and CI |
| **[Reliable Event Pipeline](reliable_event_pipeline/)** | Data Engineering | Schema contracts, reject handling, deduplication, late-arriving data, idempotent loads, SQL reconciliation and unit tests |
| **[Apache Spark Retail Intelligence](apache_spark_retail_intelligence/)** | Data Engineering · Apache Spark | Explicit schemas, deterministic million-row workload, data-quality contracts, window functions, Customer 360 features, Spark ML and partition-aware Parquet output |
| **[E-commerce SQL + dbt](ecommerce_sql_analytics/)** | Analytics Engineering · SQL | Explicit relational grains, financial reconciliation, cohorts/windows, staging-to-mart dbt models and data-quality tests over 98,199 orders |
| **[PySpark Clickstream](pyspark_clickstream_analytics/)** | Data Engineering · Distributed Analytics | 165,474 real events, session transformations, a clearly labelled one-million-row load test and leakage-aware Spark ML |
| **[Executive Commerce Intelligence](executive_commerce_bi/)** | BI · Analytics | Governed KPI layer, Power BI PBIP/PBIR + TMDL/DAX, Tableau source, retained dashboard evidence and CI |
| **[Retail Cleaning & Segmentation](retail_customer_segmentation/)** | Data Science · Data Quality | Auditable cleaning of 541,909 transaction rows, validated RFM features, clustering diagnostics and tests |
| **[Customer Churn](customer_churn_prediction/)** | Data Science | Grouped holdout, out-of-fold calibration, proxy-feature policy, cost-aware threshold and bootstrap intervals |
| **[KNN Product Quality](knn_product_quality/)** | Data Science · K-Nearest Neighbours | Leakage-safe scaling, KNN hyperparameter search, scaling ablation, nearest-neighbour explanations, confidence review policy and saved inference pipeline |
| **[XGBoost Bike Demand](xgboost_bike_demand/)** | Data Science · XGBoost | Public UCI bike-demand data, chronological validation, leakage controls, seasonal baseline, boosted-tree tuning, error slices and capacity-planning decisions |
| **[Statistical Marketing Mix](statistical_marketing_mix/)** | Data Science · Statistical Modelling | OLS with HC3 errors, confidence intervals, VIF, residual diagnostics, bootstrap uncertainty, future holdout and budget scenarios on known synthetic ground truth |
| **[UK House Prices](uk_house_price_prediction/)** | Data Science · Regression | Official Land Registry data, temporal holdout, strong location baseline, CatBoost and honest negative/small-gain reporting |
| **[Energy Demand Forecasting](energy_demand_forecasting/)** | ML Engineering · TensorFlow | Chronological forecasting, strong seasonal baseline, Conv1D + LSTM, interval coverage and saved-model reload check |
| **[Deep Learning Marketing Response](deep_learning_marketing_response/)** | ML Engineering · PyTorch | Real UCI campaign data, leakage review, logistic baseline, trained MLP, class imbalance handling, AdamW, early stopping, calibration/error slices and saved checkpoint |
| **[Image Classification](image_classification_confidence/)** | ML Engineering · Computer Vision | PyTorch/EfficientNet-B0, calibration, selective review, Grad-CAM, bootstrap intervals and export parity |
| **[NLP Document Intelligence](nlp_document_intelligence/)** | AI Engineering · NLP | Text cleaning, TF-IDF n-grams, Naive Bayes baseline, calibrated linear SVM, confidence routing, category keywords, confusion/error analysis and persisted inference |
| **[Grounded RAG](grounded_rag/)** | AI Engineering · NLP | Hybrid retrieval, citations, abstention, tool routing, prompt-injection checks, FastAPI and Docker |
| **[ModelWatch](model_watch/)** | MLOps | PSI/KS drift, discrimination and calibration checks, subgroup summaries and an explicit retraining policy |
| **[ExperimentLab](experiment_lab/)** | Product Data Science | CUPED, bootstrap uncertainty, guardrails, power and a machine-readable ship/hold decision on labelled synthetic data |
| **[Parkinson's Progression](parkinsons_progression/)** | Data Science · Validation Design | Subject-grouped validation, schema/cleaning controls and clear educational/non-clinical scope |

## How to inspect a project

Each project is expected to expose the same recruiter-friendly evidence pattern:

1. `project_notebook.ipynb` — the complete notebook/project story, not a thin index.
2. `README.md` plus a data card/data-model description where needed — problem framing, dataset provenance and reproduction instructions.
3. `src/`, `run.py`, APIs and/or SQL — production-style implementation outside notebook cells.
4. `tests/` and GitHub Actions — checks for important data, modelling or engineering contracts.
5. `results/`, `outputs/`, `artifacts/` and/or `verified/` — retained evidence where a metric or result is promoted.

`verification_pass=true` means the repository's stated checks passed; it is not presented as an external audit. Synthetic fixtures, load-test replication and historical datasets are labelled where they are used.

## Verify the portfolio gates

From the repository root:

```bash
python scripts/validate_portfolio_manifest.py
python scripts/validate_skill_notebooks.py
python scripts/validate_notebook_coverage.py
python scripts/validate_full_portfolio_projects.py
python scripts/validate_new_projects.py
python scripts/validate_portfolio.py
```

The original university and laboratory notebooks remain available at the repository root for direct inspection. This folder is the recommended route for the strengthened, production-style work.
