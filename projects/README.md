# 21 End-to-End Data & AI Applications

These are the recruiter-facing projects. **Every project must stand on its own**: open one folder and understand the problem, data, analysis, implementation, evaluation, decision/solution, limitations and reproduction path.

**[Open every notebook + dataset/source →](../docs/NOTEBOOKS_AND_DATASETS.md)**

## Portfolio standard

A project is not complete because it has a long notebook. Where technically relevant, each application should show:

1. problem, stakeholder and success criteria
2. dataset provenance and reproducible acquisition
3. schema/data-quality checks
4. cleaning and preprocessing
5. **real EDA and visualisations inside the notebook**
6. feature engineering / transformations
7. baseline and meaningful model/approach comparison
8. leakage-aware validation and tuning
9. metrics, residual/error analysis and robustness checks
10. uncertainty/calibration and explainability where appropriate
11. inference or an operational decision layer
12. tests and retained results/evidence
13. APIs, pipelines, SQL/dbt, monitoring or deployment when relevant
14. limitations, risks and next steps

For major professional applications the working depth target is **roughly 1,000 meaningful visible notebook code lines when the problem genuinely supports it**. This is not permission to pad. A notebook should grow through useful analysis, plots, modelling, engineering and decision logic.

The notebook is the main recruiter artifact. Modular `.py`, `src/`, SQL, APIs and tests are supporting engineering evidence, not a place to hide the work.

## Choose the route closest to the role

| Hiring route | Start here | Then inspect |
| --- | --- | --- |
| **Data Scientist** | [Flight Delay Risk](flight_delay_risk/) | [Linear Regression Energy](linear_regression_energy_efficiency/) · [XGBoost Bike Demand](xgboost_bike_demand/) · [Customer Churn](customer_churn_prediction/) · [Statistical Marketing Mix](statistical_marketing_mix/) · [KNN Product Quality](knn_product_quality/) |
| **Data Engineer / Analytics Engineer** | [Reliable Event Pipeline](reliable_event_pipeline/) | [Apache Spark Retail Intelligence](apache_spark_retail_intelligence/) · [PySpark Clickstream](pyspark_clickstream_analytics/) · [E-commerce SQL + dbt](ecommerce_sql_analytics/) |
| **ML / AI Engineer** | [Grounded RAG](grounded_rag/) | [Deep Learning Marketing Response](deep_learning_marketing_response/) · [NLP Document Intelligence](nlp_document_intelligence/) · [Image Classification](image_classification_confidence/) · [Energy Forecasting](energy_demand_forecasting/) |
| **Data Analyst / BI / Product Analyst** | [Executive Commerce Intelligence](executive_commerce_bi/) | [Statistical Marketing Mix](statistical_marketing_mix/) · [E-commerce SQL + dbt](ecommerce_sql_analytics/) · [ExperimentLab](experiment_lab/) |

## Project map

| Project | Primary hiring signal | Inspectable evidence |
| --- | --- | --- |
| **[Flight Delay Risk Platform](flight_delay_risk/)** | Data Science · ML Engineering | Official 2026 BTS data, chronological validation, untouched test period, CatBoost, decisioning, FastAPI, Docker and CI |
| **[Linear Regression — Building Energy Efficiency](linear_regression_energy_efficiency/)** | Data Science · Linear Regression | UCI Energy Efficiency, direct EDA/plots, median baseline, OLS LinearRegression, Ridge/Lasso/polynomial comparison, CV, residual/error slices, coefficient analysis, bootstrap uncertainty and design-review scenarios |
| **[Customer Churn](customer_churn_prediction/)** | Data Science | Grouped holdout, calibration, proxy-feature policy, cost-aware threshold and bootstrap intervals |
| **[UK House Prices](uk_house_price_prediction/)** | Data Science · Regression | Official Land Registry data, temporal holdout, location baseline, CatBoost and error analysis |
| **[Retail Cleaning & Segmentation](retail_customer_segmentation/)** | Data Science · Data Quality | 541,909 transactions, auditable cleaning, RFM, cluster diagnostics, personas and tests |
| **[KNN Product Quality](knn_product_quality/)** | Data Science · KNN | Scaling, KNN hyperparameter search, ablation, nearest-neighbour evidence and confidence review |
| **[XGBoost Bike Demand](xgboost_bike_demand/)** | Data Science · XGBoost | UCI bike-demand data, chronological validation, leakage controls, seasonal baseline, boosted-tree tuning and capacity decisions |
| **[Statistical Marketing Mix](statistical_marketing_mix/)** | Data Science · Statistics | OLS/HC3, confidence intervals, VIF, residual diagnostics, bootstrap uncertainty, holdout and budget scenarios |
| **[ExperimentLab](experiment_lab/)** | Product Data Science | CUPED, bootstrap uncertainty, guardrails, power and ship/hold decision logic |
| **[Parkinson's Progression](parkinsons_progression/)** | Data Science · Validation | Subject-grouped validation, schema controls, regression and explicit non-clinical limitations |
| **[Grounded RAG](grounded_rag/)** | AI Engineering | Hybrid retrieval, citations, abstention, tool routing, prompt-injection checks, FastAPI and Docker |
| **[Deep Learning Marketing Response](deep_learning_marketing_response/)** | ML Engineering · PyTorch | UCI campaign data, leakage review, logistic baseline, actually trained MLP, AdamW, early stopping, calibration and checkpoint |
| **[NLP Document Intelligence](nlp_document_intelligence/)** | AI Engineering · NLP | Text cleaning, TF-IDF, Naive Bayes baseline, calibrated SVM, confidence routing, keywords and error analysis |
| **[Image Classification](image_classification_confidence/)** | ML Engineering · Computer Vision | EfficientNet transfer learning, calibration, selective review, Grad-CAM, uncertainty and export checks |
| **[Energy Demand Forecasting](energy_demand_forecasting/)** | ML Engineering · TensorFlow | Chronological forecasting, seasonal baseline, Conv1D/LSTM, interval coverage and model reload checks |
| **[ModelWatch](model_watch/)** | MLOps | PSI/KS drift, data quality, discrimination, calibration, subgroup monitoring and retraining policy |
| **[Reliable Event Pipeline](reliable_event_pipeline/)** | Data Engineering | Schema contracts, reject handling, deduplication, late data, idempotency, reconciliation and tests |
| **[Apache Spark Retail Intelligence](apache_spark_retail_intelligence/)** | Data Engineering · Spark | Explicit schemas, million-row workload, data-quality contracts, window functions, Customer 360, Spark ML and Parquet |
| **[PySpark Clickstream](pyspark_clickstream_analytics/)** | Data Engineering · Distributed Analytics | Real clickstream events, session/funnel transformations, one-million-row load test and Spark ML |
| **[E-commerce SQL + dbt](ecommerce_sql_analytics/)** | Analytics Engineering · SQL | Relational grain, financial reconciliation, cohorts/windows, dbt marts and data-quality tests |
| **[Executive Commerce Intelligence](executive_commerce_bi/)** | BI · Analytics | Governed KPIs, Power BI PBIP/PBIR/TMDL/DAX, Tableau assets and retained dashboard evidence |

## How to inspect a project

1. `project_notebook.ipynb` — direct analysis, plots and complete project story.
2. `README.md` — objective, dataset/provenance and reproduction route.
3. `run.py`, `src/`, SQL, API or BI files — reusable implementation.
4. `tests/` + GitHub Actions — important contracts and checks.
5. `results/`, `outputs/`, `artifacts/` or `verified/` — retained evidence.

The original university/course notebooks remain at repository root and are linked directly from the main [README](../README.md).
