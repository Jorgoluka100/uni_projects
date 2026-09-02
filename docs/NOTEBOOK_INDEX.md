# Notebook & Python Code Index

This is the single place to find the repository's Jupyter notebooks and the Python code that supports the newer work.

The repository intentionally keeps three layers:

1. **Original / university / course notebooks** — retained as evidence of the work and learning that came first.
2. **Focused skills notebooks + `.py` scripts** — compact interview-revision examples.
3. **21 strengthened projects** — each Python-backed project has a recruiter-facing `project_notebook.ipynb` plus normal Python/SQL/BI source files.

## 1. Original university and course notebooks

These files remain at repository root. They have not been replaced by the new project notebooks.

### MSc / university evidence

- [`KDDCup.ipynb`](../KDDCup.ipynb) — KDD Cup network-intrusion / cyber-attack analysis.
- [`Logistic_Regression_PySpark.ipynb`](../Logistic_Regression_PySpark.ipynb) — PySpark Logistic Regression on KDD intrusion data.
- [`Naive_Bayes_PySpark.ipynb`](../Naive_Bayes_PySpark.ipynb) — PySpark Naive Bayes on KDD intrusion data.
- [`Parkinsons_Progression_ML.ipynb`](../Parkinsons_Progression_ML.ipynb)
- [`01_UK_House_Price_Analysis_and_Prediction.ipynb`](../01_UK_House_Price_Analysis_and_Prediction.ipynb)
- [`02_SQL_Sales_and_Customer_Analysis.ipynb`](../02_SQL_Sales_and_Customer_Analysis.ipynb)
- [`03_Customer_Churn_Prediction.ipynb`](../03_Customer_Churn_Prediction.ipynb)
- [`04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb`](../04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb)
- [`05_Energy_Demand_Forecasting_with_TensorFlow.ipynb`](../05_Energy_Demand_Forecasting_with_TensorFlow.ipynb)
- [`06_Clickstream_Analysis_with_PySpark.ipynb`](../06_Clickstream_Analysis_with_PySpark.ipynb)
- [`07_London_Air_Quality_Analysis_with_R.ipynb`](../07_London_Air_Quality_Analysis_with_R.ipynb)

### LLM / Udemy & specialist learning

- [`LLM_Mastery_Hands_on_Code.ipynb`](../LLM_Mastery_Hands_on_Code.ipynb)
- [`LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb`](../LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb)

### Other retained original / laboratory notebooks

- [`01_ConsultAI_AI_Opportunity_Engine.ipynb`](../01_ConsultAI_AI_Opportunity_Engine.ipynb)
- [`12_VisionForge_PyTorch_Visual_Inspection.ipynb`](../12_VisionForge_PyTorch_Visual_Inspection.ipynb)
- [`Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb`](../Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb)
- [`AeroFlow_AI_Engine.ipynb`](../AeroFlow_AI_Engine.ipynb)
- [`Aviation_Strategy_PostgreSQL_Optimization.ipynb`](../Aviation_Strategy_PostgreSQL_Optimization.ipynb)
- [`CineIntelligence_NoSQL_DataEngineering.ipynb`](../CineIntelligence_NoSQL_DataEngineering.ipynb)
- [`Clustering_Models.ipynb`](../Clustering_Models.ipynb)
- [`Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb`](../Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb)
- [`NYC_Airbnb_Market_Analysis (1).ipynb`](../NYC_Airbnb_Market_Analysis%20(1).ipynb)
- [`Pathfinding.ipynb`](../Pathfinding.ipynb)
- [`PyTorch_medical_AI_xray_diagnosis.ipynb`](../PyTorch_medical_AI_xray_diagnosis.ipynb)
- [`Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb`](../Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb)
- [`financial_fraud_aml_detection_system.ipynb`](../financial_fraud_aml_detection_system.ipynb)

### Export any notebook to plain Python

To create a `.py` view of all notebooks without changing the originals:

```bash
python scripts/export_notebooks_to_python.py --root . --output notebook_python_exports
```

## 2. Focused skills — notebook + Python pairs

| Skill | Notebook | Python script |
| --- | --- | --- |
| Data cleaning / preprocessing | [`01_data_cleaning_preprocessing.ipynb`](../skills/01_data_cleaning_preprocessing.ipynb) | [`01_data_cleaning_preprocessing.py`](../skills/01_data_cleaning_preprocessing.py) |
| NumPy for ML | [`02_numpy_for_machine_learning.ipynb`](../skills/02_numpy_for_machine_learning.ipynb) | [`02_numpy_for_machine_learning.py`](../skills/02_numpy_for_machine_learning.py) |
| scikit-learn classification | [`03_sklearn_end_to_end_classification.ipynb`](../skills/03_sklearn_end_to_end_classification.ipynb) | [`03_sklearn_end_to_end_classification.py`](../skills/03_sklearn_end_to_end_classification.py) |
| PyTorch neural networks | [`04_pytorch_neural_network_fundamentals.ipynb`](../skills/04_pytorch_neural_network_fundamentals.ipynb) | [`04_pytorch_neural_network_fundamentals.py`](../skills/04_pytorch_neural_network_fundamentals.py) |
| LSTM sequence modelling | [`05_lstm_sequence_modelling.ipynb`](../skills/05_lstm_sequence_modelling.ipynb) | [`05_lstm_sequence_modelling.py`](../skills/05_lstm_sequence_modelling.py) |
| Text classification / NLP | [`06_text_classification_tfidf.ipynb`](../skills/06_text_classification_tfidf.ipynb) | [`06_text_classification_tfidf.py`](../skills/06_text_classification_tfidf.py) |
| CNN image fundamentals | [`07_cnn_image_fundamentals.ipynb`](../skills/07_cnn_image_fundamentals.ipynb) | [`07_cnn_image_fundamentals.py`](../skills/07_cnn_image_fundamentals.py) |
| **Linear Regression + regression fundamentals** | [`08_regression_fundamentals.ipynb`](../skills/08_regression_fundamentals.ipynb) | [`08_regression_fundamentals.py`](../skills/08_regression_fundamentals.py) |
| Clustering fundamentals | [`09_clustering_fundamentals.ipynb`](../skills/09_clustering_fundamentals.ipynb) | [`09_clustering_fundamentals.py`](../skills/09_clustering_fundamentals.py) |
| SQL analytics fundamentals | [`10_sql_analytics_fundamentals.ipynb`](../skills/10_sql_analytics_fundamentals.ipynb) | [`10_sql_analytics_fundamentals.py`](../skills/10_sql_analytics_fundamentals.py) |

## 3. Strengthened projects — 21 notebook-first applications

Every project is intended to stand alone. The notebook leads with the problem, direct data inspection, EDA/visualisation, modelling or engineering evidence, evaluation and the decision/solution. Modular source remains additional engineering evidence rather than a substitute for visible notebook work.

| Project | Notebook | Python / project code |
| --- | --- | --- |
| Executive Commerce Intelligence — Power BI + Tableau | [`project_notebook.ipynb`](../projects/executive_commerce_bi/project_notebook.ipynb) | [`projects/executive_commerce_bi/`](../projects/executive_commerce_bi/) |
| Parkinson's Progression | [`project_notebook.ipynb`](../projects/parkinsons_progression/project_notebook.ipynb) | [`projects/parkinsons_progression/`](../projects/parkinsons_progression/) |
| Reliable Event Pipeline | [`project_notebook.ipynb`](../projects/reliable_event_pipeline/project_notebook.ipynb) | [`projects/reliable_event_pipeline/`](../projects/reliable_event_pipeline/) |
| Apache Spark Retail Intelligence | [`project_notebook.ipynb`](../projects/apache_spark_retail_intelligence/project_notebook.ipynb) | [`projects/apache_spark_retail_intelligence/`](../projects/apache_spark_retail_intelligence/) |
| Flight Delay Risk | [`project_notebook.ipynb`](../projects/flight_delay_risk/project_notebook.ipynb) | [`projects/flight_delay_risk/`](../projects/flight_delay_risk/) |
| E-commerce SQL Analytics | [`project_notebook.ipynb`](../projects/ecommerce_sql_analytics/project_notebook.ipynb) | [`projects/ecommerce_sql_analytics/`](../projects/ecommerce_sql_analytics/) |
| Customer Churn | [`project_notebook.ipynb`](../projects/customer_churn_prediction/project_notebook.ipynb) | [`projects/customer_churn_prediction/`](../projects/customer_churn_prediction/) |
| **Linear Regression — Building Energy Efficiency** | [`project_notebook.ipynb`](../projects/linear_regression_energy_efficiency/project_notebook.ipynb) | [`projects/linear_regression_energy_efficiency/`](../projects/linear_regression_energy_efficiency/) |
| KNN Product Quality | [`project_notebook.ipynb`](../projects/knn_product_quality/project_notebook.ipynb) | [`projects/knn_product_quality/`](../projects/knn_product_quality/) |
| XGBoost Bike Demand | [`project_notebook.ipynb`](../projects/xgboost_bike_demand/project_notebook.ipynb) | [`projects/xgboost_bike_demand/`](../projects/xgboost_bike_demand/) |
| Statistical Marketing Mix | [`project_notebook.ipynb`](../projects/statistical_marketing_mix/project_notebook.ipynb) | [`projects/statistical_marketing_mix/`](../projects/statistical_marketing_mix/) |
| Grounded RAG | [`project_notebook.ipynb`](../projects/grounded_rag/project_notebook.ipynb) | [`projects/grounded_rag/`](../projects/grounded_rag/) |
| NLP Document Intelligence | [`project_notebook.ipynb`](../projects/nlp_document_intelligence/project_notebook.ipynb) | [`projects/nlp_document_intelligence/`](../projects/nlp_document_intelligence/) |
| Deep Learning Marketing Response | [`project_notebook.ipynb`](../projects/deep_learning_marketing_response/project_notebook.ipynb) | [`projects/deep_learning_marketing_response/`](../projects/deep_learning_marketing_response/) |
| Image Classification | [`project_notebook.ipynb`](../projects/image_classification_confidence/project_notebook.ipynb) | [`projects/image_classification_confidence/`](../projects/image_classification_confidence/) |
| ExperimentLab | [`project_notebook.ipynb`](../projects/experiment_lab/project_notebook.ipynb) | [`projects/experiment_lab/`](../projects/experiment_lab/) |
| PySpark Clickstream | [`project_notebook.ipynb`](../projects/pyspark_clickstream_analytics/project_notebook.ipynb) | [`projects/pyspark_clickstream_analytics/`](../projects/pyspark_clickstream_analytics/) |
| UK House Price Prediction | [`project_notebook.ipynb`](../projects/uk_house_price_prediction/project_notebook.ipynb) | [`projects/uk_house_price_prediction/`](../projects/uk_house_price_prediction/) |
| Energy Demand Forecasting | [`project_notebook.ipynb`](../projects/energy_demand_forecasting/project_notebook.ipynb) | [`projects/energy_demand_forecasting/`](../projects/energy_demand_forecasting/) |
| ModelWatch | [`project_notebook.ipynb`](../projects/model_watch/project_notebook.ipynb) | [`projects/model_watch/`](../projects/model_watch/) |
| Retail Customer Segmentation | [`project_notebook.ipynb`](../projects/retail_customer_segmentation/project_notebook.ipynb) | [`projects/retail_customer_segmentation/`](../projects/retail_customer_segmentation/) |

## Why both `.ipynb` and modular source exist

The notebook is the recruiter-facing project/application: problem, data, cleaning/preprocessing, EDA and plots, modelling or engineering, evaluation, results, limitations and decision logic. The repository also keeps modular source because tests, APIs, pipelines and production reuse are stronger in normal files.

```text
project_notebook.ipynb → complete readable project + direct analysis + visible code
.py / src / SQL        → reusable implementation + tests + production structure
```
