# 22 End-to-End Data, Machine Learning & AI Applications

These are the recruiter-facing projects. The portfolio deliberately separates **Data Science**, **Machine Learning**, **Deep Learning / Computer Vision**, **NLP / LLM / Applied AI**, **Data Engineering / MLOps**, and **Analytics / BI** so the breadth is obvious.

**[Open every notebook + dataset/source →](../docs/NOTEBOOKS_AND_DATASETS.md)**  
**[Open the complete ML / Deep Learning algorithm coverage map →](../docs/ML_ALGORITHM_COVERAGE.md)**

## Capability map

| Area | What is clearly demonstrated | Best evidence |
| --- | --- | --- |
| **Data Science** | EDA, data cleaning, statistics, feature engineering, regression, experiments, error analysis, decision science | Flight Delay · Linear Regression Energy · Customer Churn · House Prices · Segmentation · Marketing Mix · ExperimentLab |
| **Machine Learning** | Linear/Ridge/Lasso, Logistic Regression, KNN, Naive Bayes, SVM, Decision Trees, Random Forests, Gradient Boosting, XGBoost, CatBoost, clustering, calibration, cross-validation, tuning | foundations + Linear Regression Energy · KNN · XGBoost · Churn · Flight Delay · NLP |
| **Deep Learning** | MLPs, 2D CNNs, Conv2d, padding, pooling, BatchNorm, Dropout, augmentation, transfer learning, Conv1D, LSTM, AdamW | CNN 2D Image Classification · Image Classification Confidence · Deep Learning Marketing · Energy Forecasting |
| **Computer Vision** | custom CNNs, ResNet18, EfficientNet, transfer learning, confidence routing, Grad-CAM/explainability | CNN 2D Image Classification · Image Classification Confidence · original university CNN notebook |
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

Individual algorithms are easy to find rather than hidden behind the label “machine learning.” The concise foundations live in [`../skills/`](../skills/) and the full applications prove the methods again at a larger scale.

| Technique | Foundation / original evidence | Applied evidence |
| --- | --- | --- |
| **Linear Regression** | [regression foundations](../skills/08_regression_fundamentals.ipynb) · restored [`Linear_Regression_PySpark_CN7030.ipynb`](../Linear_Regression_PySpark_CN7030.ipynb) | [Building Energy Efficiency](linear_regression_energy_efficiency/) |
| **Ridge / Lasso** | regression foundations | Building Energy Efficiency |
| **Logistic Regression** | [classification foundations](../skills/03_sklearn_end_to_end_classification.ipynb) · original PySpark KDD Logistic Regression | Customer Churn · Deep Learning baseline |
| **K-Nearest Neighbours** | classical classification route | [KNN Product Quality](knn_product_quality/) |
| **Naive Bayes** | original PySpark KDD Naive Bayes | [NLP Document Intelligence](nlp_document_intelligence/) |
| **SVM — Linear / RBF** | **[SVM foundation](../skills/11_support_vector_machines.ipynb)** | calibrated `LinearSVC` in [NLP Document Intelligence](nlp_document_intelligence/) |
| **Decision Tree** | **[Tree & ensemble foundation](../skills/12_tree_and_ensemble_models.ipynb)** | model-family comparison evidence |
| **Random Forest / bagging** | Tree & ensemble foundation | ensemble / feature-importance evidence |
| **Gradient Boosting** | Tree & ensemble foundation | advanced follow-ons below |
| **XGBoost** | boosted-tree foundation route | [XGBoost Bike Demand](xgboost_bike_demand/) |
| **CatBoost** | applied evidence | [Flight Delay Risk](flight_delay_risk/) · [UK House Prices](uk_house_price_prediction/) |
| **K-Means clustering** | [clustering foundation](../skills/09_clustering_fundamentals.ipynb) · original `Clustering_Models.ipynb` | [Retail Segmentation](retail_customer_segmentation/) |
| **Cross-validation / tuning / calibration** | foundations | used throughout the professional ML projects |

# Deep Learning & Computer Vision

- **[CNN 2D Image Classification — Convolutions, Padding & Confidence Routing](cnn_retail_image_classification/)** — `Conv2d`, 3×3 kernels, `padding=1`, ReLU, `MaxPool2d`, BatchNorm2d, Dropout, adaptive pooling, augmentation, AdamW, ResNet18 transfer learning, calibration and human-review routing.
- [Image Classification Confidence](image_classification_confidence/) — EfficientNet transfer learning, calibration, selective prediction and Grad-CAM.
- [Deep Learning Marketing Response](deep_learning_marketing_response/) — trained PyTorch MLP, logistic baseline, AdamW and calibration.
- [Energy Demand Forecasting](energy_demand_forecasting/) — TensorFlow Conv1D/LSTM forecasting.

The original university notebook [`04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb`](../04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) remains visible separately as academic evidence.

# NLP, LLMs & Applied AI

- [NLP Document Intelligence](nlp_document_intelligence/) — text cleaning, TF-IDF, Naive Bayes, calibrated SVM, confidence routing and error analysis.
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

The original university/course notebooks are preserved at repository root, and the complete historical repository state is retained under [`../originals/pre_cleanup_2026_08_11/`](../originals/pre_cleanup_2026_08_11/).
