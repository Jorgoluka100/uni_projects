# Jorgo Luka — Complete Data, AI & University Portfolio

**MSc Artificial Intelligence & Data Science (Distinction)**  
Python · SQL · R · Apache Spark/PySpark · Power BI · Tableau · Machine Learning · Deep Learning · NLP · Applied AI

> **This repository is the one link for my complete portfolio.** It brings together my original MSc/university work, LLM/Udemy course learning, standalone industry-ready applications, foundations, datasets/provenance, results, tests and engineering evidence.

[![Portfolio checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)

## Start here — everything from one link

| Area | What it contains | Open |
| --- | --- | --- |
| **MSc / university work** | Original executed academic notebooks, including the KDD network-intrusion/cyber-attack suite | **[University projects →](docs/UNIVERSITY_PROJECTS.md)** |
| **LLM / Udemy & specialist learning** | Hands-on LLM training, Llama-style architecture/alignment and algorithm foundations | **[Course & specialist work ↓](#llm--udemy-course--specialist-learning)** |
| **20 professional projects / applications** | Standalone Data Science, Data Engineering, ML/AI Engineering, Analytics and BI portfolio pieces | **[All professional projects →](projects/)** |
| **Notebooks + datasets** | Main recruiter notebooks with source/provenance and reproduction routes | **[Notebook & dataset index →](docs/NOTEBOOKS_AND_DATASETS.md)** |
| **Foundations** | Cleaning, NumPy, **Linear Regression**, classification, clustering, SQL, PyTorch, LSTM, NLP and CNNs | **[Foundations Lab →](skills/)** |
| **Complete inventory** | All retained notebooks, professional projects and supporting evidence | **[Project catalog →](docs/PROJECT_CATALOG.md)** |

## Original MSc / university projects

The original work is deliberately preserved. It shows the academic foundation before the later production-style rebuilds.

| University work | Original evidence | Progression / strengthened evidence |
| --- | --- | --- |
| **Network Intrusion / Cyber-Attack Detection — KDD Cup** | **[KDD Cup notebook](KDDCup.ipynb)** · **[PySpark Logistic Regression](Logistic_Regression_PySpark.ipynb)** · **[PySpark Naive Bayes](Naive_Bayes_PySpark.ipynb)** | [KDD hardened extension](extensions/kdd_intrusion_v2.py) · [verified intrusion evidence](verified/kdd_intrusion/) · [verified Spark KDD evidence](verified/spark_kdd/) |
| **Parkinson's Progression Modelling** | [Original notebook](Parkinsons_Progression_ML.ipynb) | [Production version](projects/parkinsons_progression/) |
| **UK House Price Analysis & Prediction** | [Original notebook](01_UK_House_Price_Analysis_and_Prediction.ipynb) | [Production version](projects/uk_house_price_prediction/) |
| **SQL Sales & Customer Analysis** | [Original notebook](02_SQL_Sales_and_Customer_Analysis.ipynb) | [SQL + dbt application](projects/ecommerce_sql_analytics/) |
| **Customer Churn Prediction** | [Original notebook](03_Customer_Churn_Prediction.ipynb) | [Production version](projects/customer_churn_prediction/) |
| **Image Classification with CNNs & Transfer Learning** | [Original notebook](04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) | [Confidence-aware CV application](projects/image_classification_confidence/) |
| **Energy Demand Forecasting with TensorFlow** | [Original notebook](05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) | [Production version](projects/energy_demand_forecasting/) |
| **Clickstream Analysis with PySpark** | [Original notebook](06_Clickstream_Analysis_with_PySpark.ipynb) | [Production PySpark application](projects/pyspark_clickstream_analytics/) |
| **London Air Quality Analysis with R** | [Original notebook](07_London_Air_Quality_Analysis_with_R.ipynb) | Retained as R/statistical-analysis evidence |

**[Open the full university index →](docs/UNIVERSITY_PROJECTS.md)**

## LLM / Udemy course & specialist learning

These are shown separately from university work so the source of the learning is clear.

| Area | Direct evidence | What it shows / progression |
| --- | --- | --- |
| **LLM Mastery — hands-on training** | **[LLM Mastery Hands-on Code](LLM_Mastery_Hands_on_Code.ipynb)** | PyTorch transformer training, tokenisation, architecture/training parameters, checkpoints and experiment tracking; progresses into [Grounded RAG](projects/grounded_rag/) and LLM evaluation assets |
| **LLM architecture & alignment** | **[Align and Master LLMs](LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb)** | Llama-style internals including RMSNorm, RoPE and attention plus the supplied 138M-parameter FineWeb-Edu pretrained workflow for alignment study |
| **K-Nearest Neighbours** | [classification foundations](skills/03_sklearn_end_to_end_classification.ipynb) | **[KNN Product Quality Decision System](projects/knn_product_quality/)** — scaling, tuning, CV, neighbour explanations and confidence routing |
| **Linear Regression** | **[Linear Regression & Ridge notebook](skills/08_regression_fundamentals.ipynb)** · [Python](skills/08_regression_fundamentals.py) | ordinary least-squares `LinearRegression`, median baseline, leakage-safe preprocessing, Ridge comparison, MAE/RMSE/R², residual analysis and inference; progresses into [UK House Prices](projects/uk_house_price_prediction/) and [Statistical Marketing Mix](projects/statistical_marketing_mix/) |
| **Apache Spark / PySpark** | [university Clickstream notebook](06_Clickstream_Analysis_with_PySpark.ipynb) | [Apache Spark Retail Intelligence](projects/apache_spark_retail_intelligence/) · [PySpark Clickstream](projects/pyspark_clickstream_analytics/) |
| **XGBoost** | regression/model-selection foundations | [XGBoost Bike Demand](projects/xgboost_bike_demand/) |
| **Neural networks / deep learning** | [PyTorch fundamentals](skills/04_pytorch_neural_network_fundamentals.ipynb) | [trained PyTorch Marketing Response model](projects/deep_learning_marketing_response/) · [Image Classification](projects/image_classification_confidence/) |
| **NLP** | [TF-IDF text classification](skills/06_text_classification_tfidf.ipynb) | [NLP Document Intelligence](projects/nlp_document_intelligence/) · [Grounded RAG](projects/grounded_rag/) |
| **Statistical modelling** | regression/statistics foundations | [Statistical Marketing Mix](projects/statistical_marketing_mix/) · [ExperimentLab](projects/experiment_lab/) |

## Industry-ready portfolio — 20 standalone applications

Each `projects/<name>/` folder is intended to stand alone: problem, dataset/provenance, recruiter-facing `.ipynb`, visible implementation, cleaning/preprocessing, modelling or engineering, evaluation, results, limitations and reproducibility. Supporting `.py`, tests, APIs, pipelines, CI and deployment assets stay alongside the notebook where appropriate.

### Data Science & Decision Science

- **[Flight Delay Risk Platform](projects/flight_delay_risk/)** — official 2026 BTS data, temporal evaluation, CatBoost, decisioning, FastAPI, Docker, CI
- **[Customer Churn Prediction](projects/customer_churn_prediction/)** — cleaning, grouped validation, calibration and cost-aware retention decisions
- **[UK House Price Prediction](projects/uk_house_price_prediction/)** — Land Registry data, regression, time-aware evaluation and uncertainty
- **[Retail Customer Segmentation](projects/retail_customer_segmentation/)** — 541,909 transactions, auditable cleaning, RFM and clustering
- **[KNN Product Quality Decision System](projects/knn_product_quality/)** — KNN, scaling, cross-validation, nearest-neighbour evidence and review policy
- **[XGBoost Bike Demand](projects/xgboost_bike_demand/)** — chronological forecasting, boosted-tree tuning and operations decisions
- **[Statistical Marketing Mix](projects/statistical_marketing_mix/)** — OLS, HC3 inference, diagnostics, bootstrap uncertainty and scenarios
- **[ExperimentLab](projects/experiment_lab/)** — experimentation, CUPED, power, uncertainty and decision policy
- **[Parkinson's Progression](projects/parkinsons_progression/)** — subject-grouped validation and regression

### ML / AI Engineering

- **[Grounded RAG](projects/grounded_rag/)** — retrieval, citations, abstention, prompt-injection checks, evaluation, FastAPI and Docker
- **[Deep Learning Marketing Response](projects/deep_learning_marketing_response/)** — real UCI data, logistic baseline and actually trained PyTorch MLP
- **[NLP Document Intelligence](projects/nlp_document_intelligence/)** — TF-IDF, Naive Bayes, calibrated SVM, confidence routing and error analysis
- **[Image Classification Confidence](projects/image_classification_confidence/)** — transfer learning, calibration, selective prediction and Grad-CAM
- **[Energy Demand Forecasting](projects/energy_demand_forecasting/)** — TensorFlow Conv1D/LSTM forecasting against seasonal baselines
- **[ModelWatch](projects/model_watch/)** — drift, data quality, calibration, performance monitoring and retraining policy

### Data Engineering / Analytics Engineering

- **[Reliable Event Pipeline](projects/reliable_event_pipeline/)** — schema validation, reject handling, deduplication, idempotency, reconciliation and tests
- **[Apache Spark Retail Intelligence](projects/apache_spark_retail_intelligence/)** — explicit schemas, million-row workload, windows, Customer 360, Spark ML and Parquet
- **[PySpark Clickstream Analytics](projects/pyspark_clickstream_analytics/)** — real distributed clickstream transformations plus load testing
- **[E-commerce SQL + dbt Analytics](projects/ecommerce_sql_analytics/)** — relational grain control, SQL analytics, dbt modelling and data-quality tests

### Analytics / BI

- **[Executive Commerce Intelligence](projects/executive_commerce_bi/)** — governed KPI data, Power BI PBIP/PBIR/TMDL, Tableau source and validation

## Other original / laboratory notebooks retained

Nothing has been thrown away. Additional original work remains directly accessible here:

[ConsultAI AI Opportunity Engine](01_ConsultAI_AI_Opportunity_Engine.ipynb) · [VisionForge PyTorch Visual Inspection](12_VisionForge_PyTorch_Visual_Inspection.ipynb) · [Advanced Multi-Modal Health Analytics](Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb) · [AeroFlow AI Engine](AeroFlow_AI_Engine.ipynb) · [Aviation PostgreSQL Optimisation](Aviation_Strategy_PostgreSQL_Optimization.ipynb) · [CineIntelligence NoSQL](CineIntelligence_NoSQL_DataEngineering.ipynb) · [Clustering Models](Clustering_Models.ipynb) · [Movie Recommendation System](Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb) · [NYC Airbnb Analysis](NYC_Airbnb_Market_Analysis%20(1).ipynb) · [Pathfinding](Pathfinding.ipynb) · [PyTorch Medical X-ray AI](PyTorch_medical_AI_xray_diagnosis.ipynb) · [Telecom Churn + SQL](Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb) · [Financial Fraud / AML](financial_fraud_aml_detection_system.ipynb)

For the full inventory and classification of every retained asset, use **[PROJECT_CATALOG.md](docs/PROJECT_CATALOG.md)**.

## Recruiter route by role

| Role | Inspect first |
| --- | --- |
| **Data Scientist** | [Flight Delay](projects/flight_delay_risk/) · [XGBoost Bike Demand](projects/xgboost_bike_demand/) · [Customer Churn](projects/customer_churn_prediction/) · [Statistical Marketing Mix](projects/statistical_marketing_mix/) |
| **Data Engineer / Analytics Engineer** | [Reliable Event Pipeline](projects/reliable_event_pipeline/) · [Apache Spark Retail](projects/apache_spark_retail_intelligence/) · [PySpark Clickstream](projects/pyspark_clickstream_analytics/) · [SQL + dbt](projects/ecommerce_sql_analytics/) |
| **ML / AI Engineer** | [Grounded RAG](projects/grounded_rag/) · [Deep Learning](projects/deep_learning_marketing_response/) · [NLP](projects/nlp_document_intelligence/) · [Image Classification](projects/image_classification_confidence/) |
| **Data Analyst / BI** | [Executive Commerce BI](projects/executive_commerce_bi/) · [Statistical Marketing Mix](projects/statistical_marketing_mix/) · [ExperimentLab](projects/experiment_lab/) · [SQL + dbt](projects/ecommerce_sql_analytics/) |

## Portfolio standard

The strengthened projects prioritise substantive end-to-end depth rather than artificial line-count padding. Where appropriate they show raw-data handling, validation, EDA, feature engineering, baselines, multiple modelling approaches, tuning, realistic holdouts, error analysis, explainability, inference/decision logic, tests, monitoring/deployment, results and limitations.

**[Open every main professional notebook with its dataset/source →](docs/NOTEBOOKS_AND_DATASETS.md)**

## Main stack

**Data & BI:** Python · SQL · R · Pandas · NumPy · Power BI · DAX · TMDL · Tableau · PostgreSQL · DuckDB · Apache Spark/PySpark · dbt  
**ML & Statistics:** scikit-learn · Linear Regression · KNN · XGBoost · CatBoost · statsmodels · SciPy · PyTorch · TensorFlow/Keras  
**Applied AI:** NLP · LLMs · retrieval/RAG · computer vision · FastAPI  
**Engineering:** Docker · Git · GitHub Actions · testing · data/model validation · monitoring

## Additional analyst work

The supporting **[Data Analyst Bootcamp portfolio](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp)** is linked here as additional evidence without replacing this repository as the single primary portfolio URL.

## Licence

My own code and documentation are MIT-licensed. External datasets, course materials and pretrained models retain their original licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
