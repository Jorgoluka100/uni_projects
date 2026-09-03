# Notebooks + Datasets — Recruiter Quick Access

Every strengthened project has a Jupyter notebook entry point and an explicit data route. There are now **22 professional applications**.

> **Data policy:** large, third-party, licensed or frequently refreshed raw datasets are not copied into Git history just for appearance. Projects name the source and keep a reproducible download/load path. Small fixtures, synthetic data and compact derived evidence are committed when that is the right engineering choice.

| Project | Jupyter notebook | Dataset | Data / reproduction route |
| --- | --- | --- | --- |
| **Flight Delay Risk Platform** | [`project_notebook.ipynb`](../projects/flight_delay_risk/project_notebook.ipynb) | U.S. DOT/BTS On-Time Reporting, Jan–May 2026 | [`DATA_CARD.md`](../projects/flight_delay_risk/DATA_CARD.md) · [`run.py`](../projects/flight_delay_risk/run.py) |
| **Linear Regression — Building Energy Efficiency** | [`project_notebook.ipynb`](../projects/linear_regression_energy_efficiency/project_notebook.ipynb) | UCI Energy Efficiency, 768 building configurations | Reproducible UCI loader in [`run.py`](../projects/linear_regression_energy_efficiency/run.py). |
| **CNN Retail Image Classification** | [`project_notebook.ipynb`](../projects/cnn_retail_image_classification/project_notebook.ipynb) | **CIFAR-10**, 60,000 labelled colour images / 10 classes | `torchvision.datasets.CIFAR10` in [`run.py`](../projects/cnn_retail_image_classification/run.py); custom CNN, regularised CNN and ResNet18 transfer learning. |
| **Reliable Event Pipeline** | [`project_notebook.ipynb`](../projects/reliable_event_pipeline/project_notebook.ipynb) | Deterministic ingestion fixtures | Project fixtures and pipeline tests. |
| **E-commerce SQL + dbt** | [`project_notebook.ipynb`](../projects/ecommerce_sql_analytics/project_notebook.ipynb) | Olist Brazilian E-commerce | [`run.py`](../projects/ecommerce_sql_analytics/run.py) + [`DATA_MODEL.md`](../projects/ecommerce_sql_analytics/DATA_MODEL.md). |
| **PySpark Clickstream** | [`project_notebook.ipynb`](../projects/pyspark_clickstream_analytics/project_notebook.ipynb) | Historical UCI e-commerce/clickstream sources | Dataset roles and load-test route in project README. |
| **Apache Spark Retail Intelligence** | [`project_notebook.ipynb`](../projects/apache_spark_retail_intelligence/project_notebook.ipynb) | Deterministic synthetic high-volume retail stream | Generated in Spark by [`run.py`](../projects/apache_spark_retail_intelligence/run.py). |
| **Executive Commerce Intelligence** | [`project_notebook.ipynb`](../projects/executive_commerce_bi/project_notebook.ipynb) | Governed Olist warehouse exports | Rebuildable governed BI exports. |
| **Retail Cleaning & Segmentation** | [`project_notebook.ipynb`](../projects/retail_customer_segmentation/project_notebook.ipynb) | UCI Online Retail, 541,909 rows | Runtime download in [`run.py`](../projects/retail_customer_segmentation/run.py). |
| **Customer Churn** | [`project_notebook.ipynb`](../projects/customer_churn_prediction/project_notebook.ipynb) | UCI Iranian Churn, 3,150 records | Official source route in project code. |
| **KNN Product Quality** | [`project_notebook.ipynb`](../projects/knn_product_quality/project_notebook.ipynb) | scikit-learn Wine | `load_wine` in [`run.py`](../projects/knn_product_quality/run.py). |
| **XGBoost Bike Demand** | [`project_notebook.ipynb`](../projects/xgboost_bike_demand/project_notebook.ipynb) | UCI Bike Sharing | Public UCI archive in [`run.py`](../projects/xgboost_bike_demand/run.py). |
| **Deep Learning Marketing Response** | [`project_notebook.ipynb`](../projects/deep_learning_marketing_response/project_notebook.ipynb) | UCI Bank Marketing | [`run.py`](../projects/deep_learning_marketing_response/run.py) downloads data and trains PyTorch MLP. |
| **NLP Document Intelligence** | [`project_notebook.ipynb`](../projects/nlp_document_intelligence/project_notebook.ipynb) | scikit-learn 20 Newsgroups | `fetch_20newsgroups` in [`run.py`](../projects/nlp_document_intelligence/run.py). |
| **Statistical Marketing Mix** | [`project_notebook.ipynb`](../projects/statistical_marketing_mix/project_notebook.ipynb) | Deterministic synthetic weekly marketing data | Generated with known ground truth in project code. |
| **UK House Price Prediction** | [`project_notebook.ipynb`](../projects/uk_house_price_prediction/project_notebook.ipynb) | HM Land Registry Price Paid | Source/filtering route in project README and `run.py`. |
| **Energy Demand Forecasting** | [`project_notebook.ipynb`](../projects/energy_demand_forecasting/project_notebook.ipynb) | Open Power System Data | Source/split in project README and `run.py`. |
| **Image Classification + Confidence** | [`project_notebook.ipynb`](../projects/image_classification_confidence/project_notebook.ipynb) | Makerere Beans | Dataset/evidence in project README and results. |
| **Grounded RAG** | [`project_notebook.ipynb`](../projects/grounded_rag/project_notebook.ipynb) | Synthetic deterministic knowledge/evaluation fixture | Fixture ships with project. |
| **ModelWatch** | [`project_notebook.ipynb`](../projects/model_watch/project_notebook.ipynb) | Deterministic monitoring simulation | Generated by project code. |
| **ExperimentLab** | [`project_notebook.ipynb`](../projects/experiment_lab/project_notebook.ipynb) | Reproducible synthetic randomized experiment | Generated with known treatment effect. |
| **Parkinson's Progression** | [`project_notebook.ipynb`](../projects/parkinsons_progression/project_notebook.ipynb) | UCI Parkinson's Telemonitoring | UCI/local data route with subject-grouped validation. |

## Original university/course work

Original notebooks are preserved at repository root. This includes the restored **`Linear_Regression_PySpark_CN7030.ipynb`**, KDD Logistic Regression / Naive Bayes work, CN7030 NLP feature-extraction notebooks, CNN / transfer-learning coursework, TensorFlow, PySpark, SQL, R and the LLM notebooks.

## Recruiter starting routes

- **Data Science:** [Flight Delay](../projects/flight_delay_risk/project_notebook.ipynb) → [Linear Regression Energy](../projects/linear_regression_energy_efficiency/project_notebook.ipynb) → [XGBoost](../projects/xgboost_bike_demand/project_notebook.ipynb) → [Customer Churn](../projects/customer_churn_prediction/project_notebook.ipynb)
- **Machine Learning:** [Linear Regression](../projects/linear_regression_energy_efficiency/project_notebook.ipynb) → [KNN](../projects/knn_product_quality/project_notebook.ipynb) → [XGBoost](../projects/xgboost_bike_demand/project_notebook.ipynb) → [NLP classification](../projects/nlp_document_intelligence/project_notebook.ipynb)
- **Deep Learning / Computer Vision:** [CNN Retail](../projects/cnn_retail_image_classification/project_notebook.ipynb) → [Image Classification Confidence](../projects/image_classification_confidence/project_notebook.ipynb) → [Deep Learning Marketing](../projects/deep_learning_marketing_response/project_notebook.ipynb) → [Energy Forecasting](../projects/energy_demand_forecasting/project_notebook.ipynb)
- **NLP / LLM / AI:** [NLP Document Intelligence](../projects/nlp_document_intelligence/project_notebook.ipynb) → [Grounded RAG](../projects/grounded_rag/project_notebook.ipynb)
- **Data Engineer / Analytics Engineer:** [Reliable Event Pipeline](../projects/reliable_event_pipeline/project_notebook.ipynb) → [Apache Spark Retail](../projects/apache_spark_retail_intelligence/project_notebook.ipynb) → [PySpark Clickstream](../projects/pyspark_clickstream_analytics/project_notebook.ipynb) → [SQL + dbt](../projects/ecommerce_sql_analytics/project_notebook.ipynb)
- **Data Analyst / BI:** [Executive Commerce BI](../projects/executive_commerce_bi/project_notebook.ipynb) → [Statistical Marketing Mix](../projects/statistical_marketing_mix/project_notebook.ipynb) → [ExperimentLab](../projects/experiment_lab/project_notebook.ipynb)

For every professional project, `project_notebook.ipynb` is the main recruiter-facing application; source, tests, APIs and pipelines are supporting engineering evidence.
