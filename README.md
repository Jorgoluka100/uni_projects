# Jorgo Luka — Graduate Data & AI Portfolio

**MSc Artificial Intelligence & Data Science (Distinction)**  
Python · SQL · Power BI · Tableau · Data Science · Data Engineering · Machine Learning · Applied AI

This portfolio is structured to answer the questions I would expect in a graduate or junior technical interview: **can I work with raw data, build models, evaluate them properly, communicate results to a business and turn stronger work into reliable software?**

[![Portfolio checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)
[![BI portfolio project](https://github.com/Jorgoluka100/uni_projects/actions/workflows/bi-portfolio-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/bi-portfolio-ci.yml)
[![Flight delay project](https://github.com/Jorgoluka100/uni_projects/actions/workflows/flight-delay-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/flight-delay-ci.yml)
[![dbt analytics engineering](https://github.com/Jorgoluka100/uni_projects/actions/workflows/dbt-analytics-engineering.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/dbt-analytics-engineering.yml)

## Start here

**[Open every recruiter-facing `.ipynb` + its dataset/source →](docs/NOTEBOOKS_AND_DATASETS.md)**  
**[Foundation → Intermediate → Advanced hiring roadmap](docs/HIRING_PORTFOLIO.md)**

| Level | Purpose | Evidence |
| --- | --- | --- |
| **Foundation** | Prove the building blocks individually | [Data & AI Foundations Lab](skills/) — cleaning, NumPy, regression, classification, clustering, SQL, PyTorch, LSTM, text and CNNs |
| **Intermediate** | Apply those skills to real data and business/engineering problems | [End-to-end projects](projects/) — real datasets, Power BI, Tableau, SQL, PySpark, pipelines, testing, dbt, APIs, Docker and monitoring |
| **Advanced** | Combine modelling and engineering into a releaseable system | [Flight Delay Risk Platform](projects/flight_delay_risk/) — official 2026 data → temporal ML evaluation → verified model → FastAPI → Docker → CI |

## Start with the role you are hiring for

| Role | Best evidence | What it demonstrates |
| --- | --- | --- |
| **Data Scientist** | [Flight Delay Risk](projects/flight_delay_risk/) · [Customer Churn](projects/customer_churn_prediction/) · [UK House Prices](projects/uk_house_price_prediction/) · [Parkinson's Progression](projects/parkinsons_progression/) | cleaning, features, baselines, supervised ML, leakage control, realistic validation, calibration and uncertainty |
| **Data Engineer / Analytics Engineer** | [Reliable Event Pipeline](projects/reliable_event_pipeline/) · [PySpark Clickstream](projects/pyspark_clickstream_analytics/) · [E-commerce SQL + dbt](projects/ecommerce_sql_analytics/) | ingestion, schema contracts, deduplication, idempotency, SQL grain, dbt modelling, data-quality tests and scalable transformations |
| **ML / AI Engineer** | [Flight Delay Risk Platform](projects/flight_delay_risk/) · [Grounded RAG](projects/grounded_rag/) · [Image Classification](projects/image_classification_confidence/) · [ModelWatch](projects/model_watch/) | serving, FastAPI, Docker, retrieval, PyTorch, model export, uncertainty, drift monitoring and CI |
| **Data Analyst / BI / Product Analyst** | [Executive Commerce Intelligence — Power BI + Tableau](projects/executive_commerce_bi/) · [E-commerce SQL + dbt](projects/ecommerce_sql_analytics/) · [ExperimentLab](projects/experiment_lab/) | dashboarding, KPI design, Power BI/TMDL, Tableau, SQL, experimentation and business interpretation |

**[Browse all 14 end-to-end projects by role and evidence →](projects/)**

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

### Data handling

**[Retail Customer Data Cleaning & Segmentation](projects/retail_customer_segmentation/)** starts with **541,909 raw transaction rows** and audits duplicates, missing customer IDs, cancellations and invalid quantities/prices before building validated RFM features and clustering customers.

### Business intelligence — Power BI + Tableau

**[Executive Commerce Intelligence](projects/executive_commerce_bi/)** closes the gap between technical analysis and business communication. Both dashboard tools consume the same governed exports from the verified e-commerce warehouse so KPI definitions stay consistent across tools.

The project contains:

- a Power BI **PBIP / PBIR + TMDL** semantic-model project with DAX measures and report source
- a Tableau **`.twb` workbook source** and calculation pack
- a shared KPI dictionary and dashboard-story contract
- reproducible BI exports with required-column checks, hashes and row counts
- a static GitHub dashboard preview using retained verified values
- CI that validates the data contract, Power BI bindings and Tableau XML structure

The underlying commercial evidence covers **98,199 orders**, **94,983 customers**, **R$13.49M merchandise value**, customer repeat behaviour, category performance, delivery/review quality and seller concentration.

### SQL + analytics engineering

**[E-commerce SQL Analytics](projects/ecommerce_sql_analytics/)** reconciles **98,199 commercial orders** across different relational grains rather than allowing many-to-many joins to inflate revenue. It also contains a **[dbt + DuckDB extension](projects/ecommerce_sql_analytics/dbt_project/)** with sources, staging/intermediate/mart models, relationship tests, custom data-quality tests and CI.

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
- shared KPI definitions for BI work
- validation / holdout design that matches intended use
- leakage checks and meaningful baselines
- uncertainty, calibration, threshold or stability checks where relevant
- machine-readable retained results
- unit tests and self-tests
- GitHub Actions CI
- dbt tests / model lineage where appropriate
- source-controlled Power BI / Tableau assets
- APIs and Docker where deployment adds value
- limitations and negative results stated explicitly

## Original university and laboratory work

I keep the original executed notebooks rather than rewriting history. They are indexed in **[docs/UNIVERSITY_PROJECTS.md](docs/UNIVERSITY_PROJECTS.md)** and **[docs/PROJECT_CATALOG.md](docs/PROJECT_CATALOG.md)**, while the recruiter-facing path above points to strengthened work.

## Main stack

**Data & BI:** Python · SQL · Pandas · NumPy · Power BI · DAX · TMDL · Tableau · PostgreSQL · DuckDB · PySpark · dbt  
**ML:** scikit-learn · CatBoost · PyTorch · TensorFlow/Keras  
**Applied AI:** NLP · retrieval/RAG · computer vision · FastAPI  
**Engineering:** Docker · Git · GitHub Actions · testing · data/model validation · model monitoring

## Supporting analyst portfolio

The separate **[Data Analyst Bootcamp portfolio](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp)** keeps additional dashboard, reporting, cleaning and SQL work available without crowding this repository.

## Licence

My own code and documentation are MIT-licensed. External datasets and pretrained models retain their original licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
