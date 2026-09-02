# Jorgo Luka — Data, AI & University Portfolio

**MSc Artificial Intelligence & Data Science (Distinction)**  
Python · SQL · R · Apache Spark/PySpark · Power BI · Tableau · Machine Learning · Deep Learning · NLP · Applied AI

> **One link. Everything is here.** Original MSc/university work, LLM/Udemy learning, focused foundations and **21 strengthened end-to-end applications**.

[![Portfolio checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)

## Start here

| What you want to inspect | Open |
| --- | --- |
| **Original university/course work** | Direct links are immediately below ↓ |
| **21 professional applications** | [projects/](projects/) |
| **Every recruiter notebook + dataset/source** | [docs/NOTEBOOKS_AND_DATASETS.md](docs/NOTEBOOKS_AND_DATASETS.md) |
| **Focused ML/data foundations** | [skills/](skills/) |
| **Complete inventory** | [docs/PROJECT_CATALOG.md](docs/PROJECT_CATALOG.md) |

## Original MSc / university work — direct access

The original notebooks are preserved. They show the academic work first; the newer project folders show how the same skills were developed into stronger job-facing applications.

| Original project | Notebook | Stronger / follow-on evidence |
| --- | --- | --- |
| **KDD Cup Network Intrusion / Cyber-Attack Detection** | [KDDCup.ipynb](KDDCup.ipynb) | [Hardened KDD extension](extensions/kdd_intrusion_v2.py) · [verified evidence](verified/kdd_intrusion/) |
| **KDD Intrusion — PySpark Logistic Regression** | [Logistic_Regression_PySpark.ipynb](Logistic_Regression_PySpark.ipynb) | [verified Spark KDD evidence](verified/spark_kdd/) |
| **KDD Intrusion — PySpark Naive Bayes** | [Naive_Bayes_PySpark.ipynb](Naive_Bayes_PySpark.ipynb) | [verified Spark KDD evidence](verified/spark_kdd/) |
| **Parkinson's Progression Modelling** | [Parkinsons_Progression_ML.ipynb](Parkinsons_Progression_ML.ipynb) | [Production version](projects/parkinsons_progression/) |
| **UK House Price Analysis & Prediction** | [01_UK_House_Price_Analysis_and_Prediction.ipynb](01_UK_House_Price_Analysis_and_Prediction.ipynb) | [Production version](projects/uk_house_price_prediction/) |
| **SQL Sales & Customer Analysis** | [02_SQL_Sales_and_Customer_Analysis.ipynb](02_SQL_Sales_and_Customer_Analysis.ipynb) | [SQL + dbt application](projects/ecommerce_sql_analytics/) |
| **Customer Churn Prediction** | [03_Customer_Churn_Prediction.ipynb](03_Customer_Churn_Prediction.ipynb) | [Production version](projects/customer_churn_prediction/) |
| **Image Classification — CNNs & Transfer Learning** | [04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb](04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) | [Confidence-aware CV application](projects/image_classification_confidence/) |
| **Energy Demand Forecasting — TensorFlow** | [05_Energy_Demand_Forecasting_with_TensorFlow.ipynb](05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) | [Production version](projects/energy_demand_forecasting/) |
| **Clickstream Analysis — PySpark** | [06_Clickstream_Analysis_with_PySpark.ipynb](06_Clickstream_Analysis_with_PySpark.ipynb) | [Production PySpark application](projects/pyspark_clickstream_analytics/) |
| **London Air Quality Analysis — R** | [07_London_Air_Quality_Analysis_with_R.ipynb](07_London_Air_Quality_Analysis_with_R.ipynb) | Original R/statistical-analysis evidence retained |

[Full university index →](docs/UNIVERSITY_PROJECTS.md)

## LLM / Udemy & specialist learning — direct access

| Area | Direct evidence | Stronger application |
| --- | --- | --- |
| **LLM hands-on training** | [LLM_Mastery_Hands_on_Code.ipynb](LLM_Mastery_Hands_on_Code.ipynb) | [Grounded RAG](projects/grounded_rag/) |
| **Llama-style architecture & alignment** | [LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb](LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb) | [Grounded RAG](projects/grounded_rag/) + retained LLM evaluation assets |
| **Linear Regression** | [Linear Regression foundations notebook](skills/08_regression_fundamentals.ipynb) | **[Building Energy Efficiency Decision Model](projects/linear_regression_energy_efficiency/)** |
| **K-Nearest Neighbours** | [classification foundations](skills/03_sklearn_end_to_end_classification.ipynb) | [KNN Product Quality](projects/knn_product_quality/) |
| **Apache Spark / PySpark** | [University Clickstream](06_Clickstream_Analysis_with_PySpark.ipynb) | [Apache Spark Retail Intelligence](projects/apache_spark_retail_intelligence/) |
| **XGBoost** | [regression foundations](skills/08_regression_fundamentals.ipynb) | [XGBoost Bike Demand](projects/xgboost_bike_demand/) |
| **Neural Networks / Deep Learning** | [PyTorch fundamentals](skills/04_pytorch_neural_network_fundamentals.ipynb) | [Trained PyTorch Marketing Response](projects/deep_learning_marketing_response/) |
| **NLP** | [TF-IDF text classification](skills/06_text_classification_tfidf.ipynb) | [NLP Document Intelligence](projects/nlp_document_intelligence/) |
| **Statistical Modelling** | [regression foundations](skills/08_regression_fundamentals.ipynb) | [Statistical Marketing Mix](projects/statistical_marketing_mix/) · [ExperimentLab](projects/experiment_lab/) |

## 21 strengthened professional applications

Each project is designed to stand alone. The recruiter notebook leads with the **problem → data → audit/cleaning → EDA/visualisation → modelling/engineering → evaluation → error analysis → decision/solution → limitations**. Modular code, tests, APIs and CI remain supporting evidence rather than a substitute for the notebook.

### Data Science & Decision Science

| Project | What it demonstrates |
| --- | --- |
| **[Flight Delay Risk Platform](projects/flight_delay_risk/)** | 2026 BTS data · temporal validation · CatBoost · decisioning · FastAPI · Docker · CI |
| **[Linear Regression — Building Energy Efficiency](projects/linear_regression_energy_efficiency/)** | UCI Energy Efficiency · EDA/plots · ordinary LinearRegression · Ridge/Lasso · polynomial comparison · CV · residuals · bootstrap uncertainty · scenario decisions |
| **[Customer Churn Prediction](projects/customer_churn_prediction/)** | cleaning · grouped validation · calibration · cost-aware retention decisions |
| **[UK House Price Prediction](projects/uk_house_price_prediction/)** | HM Land Registry · regression · temporal evaluation · uncertainty |
| **[Retail Customer Segmentation](projects/retail_customer_segmentation/)** | 541,909 transactions · cleaning · RFM · clustering · commercial segmentation |
| **[KNN Product Quality](projects/knn_product_quality/)** | scaling · KNN tuning · CV · neighbour evidence · confidence review |
| **[XGBoost Bike Demand](projects/xgboost_bike_demand/)** | chronological forecasting · boosted trees · leakage controls · operations decisions |
| **[Statistical Marketing Mix](projects/statistical_marketing_mix/)** | OLS · HC3 robust inference · diagnostics · bootstrap uncertainty · scenarios |
| **[ExperimentLab](projects/experiment_lab/)** | A/B testing · CUPED · power · guardrails · ship/hold decision |
| **[Parkinson's Progression](projects/parkinsons_progression/)** | subject-grouped validation · regression · leakage controls · non-clinical limitations |

### ML / AI Engineering

| Project | What it demonstrates |
| --- | --- |
| **[Grounded RAG](projects/grounded_rag/)** | retrieval · citations · abstention · prompt-injection checks · FastAPI · Docker |
| **[Deep Learning Marketing Response](projects/deep_learning_marketing_response/)** | real UCI data · logistic baseline · executed PyTorch MLP · calibration · saved checkpoint |
| **[NLP Document Intelligence](projects/nlp_document_intelligence/)** | TF-IDF · Naive Bayes · calibrated SVM · confidence routing · error analysis |
| **[Image Classification Confidence](projects/image_classification_confidence/)** | transfer learning · calibration · selective prediction · Grad-CAM |
| **[Energy Demand Forecasting](projects/energy_demand_forecasting/)** | TensorFlow Conv1D/LSTM · seasonal baselines · temporal validation |
| **[ModelWatch](projects/model_watch/)** | drift · data quality · discrimination · calibration · retraining policy |

### Data Engineering / Analytics Engineering

| Project | What it demonstrates |
| --- | --- |
| **[Reliable Event Pipeline](projects/reliable_event_pipeline/)** | schema contracts · rejects · deduplication · idempotency · reconciliation · tests |
| **[Apache Spark Retail Intelligence](projects/apache_spark_retail_intelligence/)** | explicit Spark schemas · million-row workload · windows · Customer 360 · Spark ML · Parquet |
| **[PySpark Clickstream Analytics](projects/pyspark_clickstream_analytics/)** | distributed clickstream transformations · sessions/funnels · Spark ML · load testing |
| **[E-commerce SQL + dbt Analytics](projects/ecommerce_sql_analytics/)** | relational grain · SQL · DuckDB/dbt · cohorts · data-quality tests |

### Analytics / BI

| Project | What it demonstrates |
| --- | --- |
| **[Executive Commerce Intelligence](projects/executive_commerce_bi/)** | governed KPIs · Power BI PBIP/PBIR/TMDL · DAX · Tableau · validation |

## Other original / laboratory work — also retained

Nothing has been deleted. These notebooks remain directly visible from the same repository:

- [ConsultAI AI Opportunity Engine](01_ConsultAI_AI_Opportunity_Engine.ipynb)
- [VisionForge PyTorch Visual Inspection](12_VisionForge_PyTorch_Visual_Inspection.ipynb)
- [Advanced Multi-Modal Health Analytics](Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb)
- [AeroFlow AI Engine](AeroFlow_AI_Engine.ipynb)
- [Aviation PostgreSQL Optimisation](Aviation_Strategy_PostgreSQL_Optimization.ipynb)
- [CineIntelligence NoSQL](CineIntelligence_NoSQL_DataEngineering.ipynb)
- [Clustering Models](Clustering_Models.ipynb)
- [Movie Recommendation System](Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb)
- [NYC Airbnb Market Analysis](NYC_Airbnb_Market_Analysis%20(1).ipynb)
- [Pathfinding](Pathfinding.ipynb)
- [PyTorch Medical X-ray AI](PyTorch_medical_AI_xray_diagnosis.ipynb)
- [Strategic Telecom Churn + SQL](Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb)
- [Financial Fraud / AML Detection](financial_fraud_aml_detection_system.ipynb)

## Recruiter route by role

| Role | Inspect first |
| --- | --- |
| **Data Scientist** | [Flight Delay](projects/flight_delay_risk/) · [Linear Regression Energy](projects/linear_regression_energy_efficiency/) · [XGBoost Bike Demand](projects/xgboost_bike_demand/) · [Customer Churn](projects/customer_churn_prediction/) |
| **Data Engineer / Analytics Engineer** | [Reliable Event Pipeline](projects/reliable_event_pipeline/) · [Apache Spark Retail](projects/apache_spark_retail_intelligence/) · [PySpark Clickstream](projects/pyspark_clickstream_analytics/) · [SQL + dbt](projects/ecommerce_sql_analytics/) |
| **ML / AI Engineer** | [Grounded RAG](projects/grounded_rag/) · [Deep Learning](projects/deep_learning_marketing_response/) · [NLP](projects/nlp_document_intelligence/) · [Image Classification](projects/image_classification_confidence/) |
| **Data Analyst / BI / Product Analyst** | [Executive Commerce BI](projects/executive_commerce_bi/) · [Statistical Marketing Mix](projects/statistical_marketing_mix/) · [ExperimentLab](projects/experiment_lab/) · [SQL + dbt](projects/ecommerce_sql_analytics/) |

## Notebook standard

For major professional applications the working depth target is **roughly 1,000 meaningful visible code lines when the problem genuinely supports it**. It is not a padding target. Projects grow through real data work: acquisition, validation, cleaning, EDA, visualisation, features, baselines, model comparison, tuning, leakage control, error analysis, explainability, uncertainty, inference, testing, monitoring/deployment and decision logic.

The notebook must show the work directly. Reusable `.py`, `src/`, SQL, tests and APIs remain because they are valuable engineering evidence, but a recruiter should not need to hunt through them to understand the project.

**[Open every professional notebook with its dataset/source →](docs/NOTEBOOKS_AND_DATASETS.md)**

## Main stack

**Data & BI:** Python · SQL · R · Pandas · NumPy · Power BI · DAX · TMDL · Tableau · PostgreSQL · DuckDB · Apache Spark/PySpark · dbt  
**ML & Statistics:** scikit-learn · Linear Regression · Logistic Regression · KNN · Naive Bayes · XGBoost · CatBoost · statsmodels · SciPy · PyTorch · TensorFlow/Keras  
**Applied AI:** NLP · LLMs · retrieval/RAG · computer vision · FastAPI  
**Engineering:** Docker · Git · GitHub Actions · testing · data/model validation · monitoring

## Supporting analyst portfolio

[Data Analyst Bootcamp portfolio](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp) remains additional evidence; this repository is the single primary portfolio URL.

## Licence

My own code and documentation are MIT-licensed. External datasets, course materials and pretrained models retain their original licences and terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
