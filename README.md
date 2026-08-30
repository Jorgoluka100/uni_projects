# Jorgo Luka — Data & AI Engineering Portfolio

**MSc Artificial Intelligence & Data Science (Distinction)**  
Python · SQL · Data Science · Data Engineering · Machine Learning · Applied AI

This portfolio is structured to answer the questions I would expect in an internship, graduate or junior technical interview: **can I work with raw data, build models, evaluate them properly and turn the stronger work into reliable software?**

[![Portfolio checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)
[![Flight delay project](https://github.com/Jorgoluka100/uni_projects/actions/workflows/flight-delay-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/flight-delay-ci.yml)
[![dbt analytics engineering](https://github.com/Jorgoluka100/uni_projects/actions/workflows/dbt-analytics-engineering.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/dbt-analytics-engineering.yml)

## Start here

**[Foundation → Intermediate → Advanced hiring roadmap](docs/HIRING_PORTFOLIO.md)**

| Level | Purpose | Evidence |
| --- | --- | --- |
| **Foundation** | Prove the building blocks individually | [Data & AI Foundations Lab](skills/) — cleaning, NumPy, regression, classification, clustering, SQL, PyTorch, LSTM, text and CNNs |
| **Intermediate** | Apply those skills to real data and engineering problems | [End-to-end projects](projects/) — real datasets, SQL, PySpark, pipelines, testing, dbt, APIs, Docker and monitoring |
| **Advanced** | Combine modelling and engineering into a releaseable system | [Flight Delay Risk Platform](projects/flight_delay_risk/) — official 2026 data → temporal ML evaluation → verified model → FastAPI → Docker → CI |

## Start with the role you are hiring for

| Role | Best evidence | What it demonstrates |
| --- | --- | --- |
| **Data Scientist** | [Flight Delay Risk](projects/flight_delay_risk/) · [Customer Churn](projects/customer_churn_prediction/) · [UK House Prices](projects/uk_house_price_prediction/) · [Parkinson's Progression](projects/parkinsons_progression/) | cleaning, feature engineering, baselines, supervised ML, leakage control, realistic validation, calibration and uncertainty |
| **Data Engineer / Analytics Engineer** | [Reliable Event Pipeline](projects/reliable_event_pipeline/) · [PySpark Clickstream](projects/pyspark_clickstream_analytics/) · [E-commerce SQL + dbt](projects/ecommerce_sql_analytics/) | ingestion, schema contracts, deduplication, idempotency, SQL grain, dbt modelling, data-quality tests and scalable transformations |
| **ML / AI Engineer** | [Flight Delay Risk Platform](projects/flight_delay_risk/) · [Grounded RAG](projects/grounded_rag/) · [Image Classification](projects/image_classification_confidence/) · [ModelWatch](projects/model_watch/) | model serving, FastAPI, Docker, retrieval, PyTorch, model export, uncertainty, drift monitoring and CI |
| **Data Analyst / Product Analyst** | [Retail Dashboard](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp/tree/main/dashboard) · [E-commerce SQL Analytics](projects/ecommerce_sql_analytics/) · [ExperimentLab](projects/experiment_lab/) | KPI logic, SQL, reporting, experimentation, data cleaning and business interpretation |

## Foundation — individual skills I can explain from code

**[Open the full Foundations Lab →](skills/README.md)**

| Area | Direct evidence |
| --- | --- |
| Data cleaning / preprocessing | [focused project](skills/01_data_cleaning_preprocessing.ipynb) → [541,909-row real-data follow-on](projects/retail_customer_segmentation/) |
| NumPy | [NumPy for ML](skills/02_numpy_for_machine_learning.ipynb) |
| Classification | [scikit-learn classification](skills/03_sklearn_end_to_end_classification.ipynb) |
| Neural networks / PyTorch | [PyTorch fundamentals](skills/04_pytorch_neural_network_fundamentals.ipynb) |
| Sequence modelling / LSTM | [LSTM fundamentals](skills/05_lstm_sequence_modelling.ipynb) |
| Text data | [TF-IDF text classification](skills/06_text_classification_tfidf.ipynb) → [Grounded RAG](projects/grounded_rag/) |
| Image data / CNNs | [CNN fundamentals](skills/07_cnn_image_fundamentals.ipynb) → [confidence-aware image classification](projects/image_classification_confidence/) |
| Regression | [regression fundamentals](skills/08_regression_fundamentals.ipynb) → [UK House Prices](projects/uk_house_price_prediction/) |
| Clustering | [clustering fundamentals](skills/09_clustering_fundamentals.ipynb) → [Retail Segmentation](projects/retail_customer_segmentation/) |
| SQL | [SQL fundamentals](skills/10_sql_analytics_fundamentals.ipynb) → [SQL + dbt analytics engineering](projects/ecommerce_sql_analytics/) |

## Intermediate — strongest end-to-end evidence

### Data work

**[Retail Customer Data Cleaning & Segmentation](projects/retail_customer_segmentation/)** starts with **541,909 raw transaction rows** and explicitly audits duplicates, missing customer IDs, cancellations and invalid quantities/prices before building validated RFM features and clustering customers.

### SQL + analytics engineering

**[E-commerce SQL Analytics](projects/ecommerce_sql_analytics/)** reconciles **98,199 commercial orders** across different relational grains rather than allowing many-to-many joins to inflate revenue. The project now also contains a **[dbt + DuckDB extension](projects/ecommerce_sql_analytics/dbt_project/)** with sources, staging/intermediate/mart models, relationship tests, custom data-quality tests and its own CI workflow.

### Data engineering

**[Reliable Event Pipeline](projects/reliable_event_pipeline/)** demonstrates ingestion, schema validation, reject handling, late-arriving data, idempotency, SQL reconciliation, audit metrics and automated tests. **[PySpark Clickstream](projects/pyspark_clickstream_analytics/)** adds distributed transformations over **165,474 real events** and a separate one-million-row load test.

### ML / deep learning

- **[Customer Churn](projects/customer_churn_prediction/)** — grouped holdout, out-of-fold calibration and cost-aware threshold selection
- **[Energy Demand Forecasting](projects/energy_demand_forecasting/)** — TensorFlow Conv1D + LSTM against a strong seasonal baseline
- **[Image Classification](projects/image_classification_confidence/)** — EfficientNet-B0, bootstrap uncertainty, selective prediction, Grad-CAM and export verification
- **[Grounded RAG](projects/grounded_rag/)** — hybrid retrieval, citations, abstention, prompt-injection checks, FastAPI and Docker
- **[ModelWatch](projects/model_watch/)** — drift, discrimination and calibration monitoring

## Advanced flagship — Flight Delay Risk Platform

**[Open project →](projects/flight_delay_risk/)**

The model uses official **2026 U.S. Bureau of Transportation Statistics** flight data and keeps May 2026 as an untouched **180,000-flight** out-of-time test set.

```text
official monthly data
      ↓
schema / data-quality checks
      ↓
leakage-safe schedule features
      ↓
CatBoost + chronological validation
      ↓
validation-selected review capacity
      ↓
untouched May 2026 test
      ↓
verified model + release metadata
      ↓
FastAPI single / batch inference
      ↓
Docker service
      ↓
unit tests + container-build CI
```

The serving layer reconstructs the exact training feature schema from schedule-time inputs and refuses to load a model release unless the accompanying metadata reports `verification_pass=true`.

Retained test evidence: **PR-AUC 0.291 vs 0.215 prevalence baseline**, with the highest-risk 10% of flights reaching **1.58×** the normal delay rate. I leave the moderate predictive performance visible rather than overselling the model.

## Engineering standard

Across the stronger projects I make these visible rather than leaving them implicit:

- source provenance, schema and quality checks
- missing-value, duplicate and join-grain controls
- validation / holdout design that matches the intended use
- leakage checks
- meaningful baselines before more complex models
- uncertainty, calibration, threshold or stability checks where relevant
- machine-readable retained results
- unit tests and self-tests
- GitHub Actions CI
- dbt tests / model lineage where appropriate
- APIs and Docker where deployment adds value
- limitations and negative results stated explicitly

## Original university and laboratory work

I keep the original executed notebooks rather than rewriting history. They are indexed in **[docs/UNIVERSITY_PROJECTS.md](docs/UNIVERSITY_PROJECTS.md)** and **[docs/PROJECT_CATALOG.md](docs/PROJECT_CATALOG.md)**, while the recruiter-facing path above points to the strengthened work.

This includes the original UK house-price, SQL/customer, churn, CNN, TensorFlow forecasting, PySpark and R projects, plus retained laboratory notebooks such as `Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb`, `AeroFlow_AI_Engine.ipynb`, `Aviation_Strategy_PostgreSQL_Optimization.ipynb`, `CineIntelligence_NoSQL_DataEngineering.ipynb`, `Clustering_Models.ipynb`, `KDDCup.ipynb`, `LLM_Mastery_Hands_on_Code.ipynb`, `LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb`, `Logistic_Regression_PySpark.ipynb`, `Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb`, `NYC_Airbnb_Market_Analysis (1).ipynb`, `Naive_Bayes_PySpark.ipynb`, `Parkinsons_Progression_ML.ipynb`, `Pathfinding.ipynb`, `PyTorch_medical_AI_xray_diagnosis.ipynb`, `Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb` and `financial_fraud_aml_detection_system.ipynb`.

## Main stack

**Data:** Python · SQL · Pandas · NumPy · PostgreSQL · DuckDB · PySpark · dbt  
**ML:** scikit-learn · CatBoost · PyTorch · TensorFlow/Keras  
**Applied AI:** NLP · retrieval/RAG · computer vision · FastAPI  
**Engineering:** Docker · Git · GitHub Actions · testing · data/model validation · model monitoring

## Supporting analyst portfolio

The separate **[Data Analyst Bootcamp portfolio](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp)** keeps dashboard, reporting and analyst-focused work available without crowding this repository.

## Licence

My own code and documentation are MIT-licensed. External datasets and pretrained models retain their original licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
