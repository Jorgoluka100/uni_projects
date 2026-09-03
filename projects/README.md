# 22 End-to-End Data, Machine Learning & AI Applications

These are the recruiter-facing projects. The portfolio deliberately separates **Data Science**, **Machine Learning**, **Deep Learning / Computer Vision**, **NLP / LLM / Applied AI**, **Data Engineering / MLOps**, and **Analytics / BI** so the breadth is obvious.

**[Open every notebook + dataset/source →](../docs/NOTEBOOKS_AND_DATASETS.md)**

## Capability map

| Area | What is clearly demonstrated | Best evidence |
| --- | --- | --- |
| **Data Science** | EDA, data cleaning, statistics, feature engineering, regression, experiments, error analysis, decision science | Flight Delay · Linear Regression Energy · Customer Churn · House Prices · Segmentation · Marketing Mix · ExperimentLab |
| **Machine Learning** | Linear Regression, Logistic Regression, KNN, Naive Bayes, clustering, XGBoost, CatBoost, SVM, calibration, cross-validation, tuning | Linear Regression Energy · KNN Product Quality · XGBoost Bike Demand · Customer Churn · Flight Delay · NLP Document Intelligence |
| **Deep Learning** | MLPs, CNNs, BatchNorm, Dropout, augmentation, transfer learning, Conv1D, LSTM, AdamW, learning curves | CNN Retail Image Classification · Image Classification Confidence · Deep Learning Marketing Response · Energy Forecasting |
| **Computer Vision** | custom CNNs, ResNet18, EfficientNet, transfer learning, confidence routing, Grad-CAM/explainability | CNN Retail Image Classification · Image Classification Confidence · original university CNN notebook |
| **NLP / LLM / Applied AI** | TF-IDF, Naive Bayes, calibrated SVM, embeddings, retrieval, RAG, citations, abstention, prompt-injection checks | NLP Document Intelligence · Grounded RAG · original CN7030 NLP work · LLM notebooks |
| **Data Engineering** | PySpark, Apache Spark, SQL/dbt, schemas, pipelines, deduplication, partitions, Parquet, distributed analytics | Reliable Event Pipeline · Spark Retail · PySpark Clickstream · E-commerce SQL + dbt |
| **MLOps / ML Engineering** | testing, model persistence, monitoring, drift, calibration, inference, APIs, Docker, CI | ModelWatch · Flight Delay · Grounded RAG · image projects |
| **Analytics / BI** | SQL, governed KPIs, Power BI, DAX, Tableau, cohorts and executive decision support | Executive Commerce Intelligence · E-commerce SQL + dbt |

## Portfolio standard

Every full application must show the complete chain where relevant:

1. problem / stakeholder / success criteria
2. dataset provenance
3. schema and data-quality checks
4. cleaning / preprocessing
5. EDA and visualisation
6. feature engineering
7. baseline
8. model / approach comparison
9. validation and tuning
10. metrics and failure analysis
11. uncertainty / calibration / explainability where appropriate
12. operational decision or inference layer
13. tests and retained evidence
14. deployment / API / pipeline / monitoring when relevant
15. limitations and next steps

The working notebook depth target is roughly **1,000 meaningful visible lines when justified by the problem**, never filler.

# Data Science

- [Flight Delay Risk Platform](flight_delay_risk/) — temporal validation, CatBoost, risk decisions.
- [Linear Regression — Building Energy Efficiency](linear_regression_energy_efficiency/) — ordinary linear regression, Ridge/Lasso, residuals and uncertainty.
- [Customer Churn](customer_churn_prediction/) — classification, calibration and retention economics.
- [UK House Prices](uk_house_price_prediction/) — regression with time-aware validation.
- [Retail Customer Segmentation](retail_customer_segmentation/) — RFM, clustering and personas.
- [KNN Product Quality](knn_product_quality/) — distance-based learning, scaling, tuning and confidence.
- [XGBoost Bike Demand](xgboost_bike_demand/) — boosted trees, chronological validation and operations decisions.
- [Statistical Marketing Mix](statistical_marketing_mix/) — OLS, robust inference, diagnostics and scenarios.
- [ExperimentLab](experiment_lab/) — A/B testing, CUPED, power and guardrails.
- [Parkinson's Progression](parkinsons_progression/) — grouped regression validation and leakage control.

# Machine Learning

The portfolio makes individual ML methods easy to find rather than only naming “machine learning” broadly.

| Technique | Evidence |
| --- | --- |
| **Linear Regression** | [Building Energy Efficiency](linear_regression_energy_efficiency/) + restored `Linear_Regression_PySpark_CN7030.ipynb` |
| **Logistic Regression** | Customer Churn · Deep Learning baseline · original PySpark KDD Logistic Regression |
| **K-Nearest Neighbours** | [KNN Product Quality](knn_product_quality/) |
| **Naive Bayes** | [NLP Document Intelligence](nlp_document_intelligence/) + original PySpark KDD Naive Bayes |
| **Clustering** | [Retail Segmentation](retail_customer_segmentation/) + original `Clustering_Models.ipynb` |
| **XGBoost / boosted trees** | [XGBoost Bike Demand](xgboost_bike_demand/) |
| **CatBoost** | [Flight Delay Risk](flight_delay_risk/) · House Prices |
| **SVM / calibrated classification** | [NLP Document Intelligence](nlp_document_intelligence/) |
| **Cross-validation / tuning / calibration** | used across the Data Science and ML projects |

# Deep Learning & Computer Vision

- **[CNN Retail Image Classification & Confidence Routing](cnn_retail_image_classification/)** — custom CNN, deeper CNN, convolution blocks, ReLU, pooling, BatchNorm, Dropout, augmentation, AdamW, ResNet18 transfer learning, calibration and review routing.
- [Image Classification Confidence](image_classification_confidence/) — EfficientNet transfer learning, calibration, selective prediction and Grad-CAM.
- [Deep Learning Marketing Response](deep_learning_marketing_response/) — trained PyTorch MLP, logistic baseline, AdamW and calibration.
- [Energy Demand Forecasting](energy_demand_forecasting/) — TensorFlow Conv1D/LSTM forecasting.

The original university notebook [`04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb`](../04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) remains visible separately as academic evidence.

# NLP, LLMs & Applied AI

- [NLP Document Intelligence](nlp_document_intelligence/) — text cleaning, TF-IDF, Naive Bayes, SVM, confidence routing and error analysis.
- [Grounded RAG](grounded_rag/) — hybrid retrieval, citations, abstention, prompt-injection checks, FastAPI and Docker.
- Original CN7030 NLP notebooks and LLM/Udemy notebooks remain at repository root.

# Data Engineering & MLOps

- [Reliable Event Pipeline](reliable_event_pipeline/) — schema contracts, rejects, idempotency and reconciliation.
- [Apache Spark Retail Intelligence](apache_spark_retail_intelligence/) — million-row Spark workload, windows, Customer 360 and Parquet.
- [PySpark Clickstream](pyspark_clickstream_analytics/) — distributed sessions/funnels and load testing.
- [E-commerce SQL + dbt](ecommerce_sql_analytics/) — analytical modelling, marts and data-quality tests.
- [ModelWatch](model_watch/) — drift, data quality, calibration and retraining policy.

# Analytics & BI

- [Executive Commerce Intelligence](executive_commerce_bi/) — Power BI PBIP/PBIR/TMDL, DAX, Tableau and governed KPIs.
- [E-commerce SQL + dbt](ecommerce_sql_analytics/) — SQL, cohorts, commercial KPIs and analytics engineering.
- [ExperimentLab](experiment_lab/) and [Statistical Marketing Mix](statistical_marketing_mix/) support product / decision analytics roles.

## How to inspect a project

1. `project_notebook.ipynb` — visible analysis and complete project story.
2. `README.md` — objective, dataset and reproduction route.
3. `run.py`, `src/`, SQL, API or BI files — engineering implementation.
4. `tests/` + GitHub Actions — validation.
5. `results/`, `outputs/`, `artifacts/` or `verified/` — evidence.

The original university/course notebooks are preserved at repository root and in the historical archive.
