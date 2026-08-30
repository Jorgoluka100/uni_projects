# Notebook & Python Code Index

This is the single place to find the repository's Jupyter notebooks and the Python code that sits behind the newer work.

The repository intentionally keeps three layers:

1. **Original / previously uploaded notebooks** — retained as historical and university evidence.
2. **Focused skills notebooks + `.py` scripts** — small interview-revision examples.
3. **Strengthened projects** — each Python-backed project has a `project_notebook.ipynb` companion while the full implementation remains in normal Python/SQL source files.

## 1. Original and previously uploaded notebooks

These files remain at repository root. They have not been replaced by the new project notebooks.

### University / verified notebook evidence

- [`01_UK_House_Price_Analysis_and_Prediction.ipynb`](../01_UK_House_Price_Analysis_and_Prediction.ipynb)
- [`02_SQL_Sales_and_Customer_Analysis.ipynb`](../02_SQL_Sales_and_Customer_Analysis.ipynb)
- [`03_Customer_Churn_Prediction.ipynb`](../03_Customer_Churn_Prediction.ipynb)
- [`04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb`](../04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb)
- [`05_Energy_Demand_Forecasting_with_TensorFlow.ipynb`](../05_Energy_Demand_Forecasting_with_TensorFlow.ipynb)
- [`06_Clickstream_Analysis_with_PySpark.ipynb`](../06_Clickstream_Analysis_with_PySpark.ipynb)
- [`07_London_Air_Quality_Analysis_with_R.ipynb`](../07_London_Air_Quality_Analysis_with_R.ipynb)
- [`01_ConsultAI_AI_Opportunity_Engine.ipynb`](../01_ConsultAI_AI_Opportunity_Engine.ipynb)
- [`12_VisionForge_PyTorch_Visual_Inspection.ipynb`](../12_VisionForge_PyTorch_Visual_Inspection.ipynb)

### Other uploaded / laboratory notebooks

- [`Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb`](../Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb)
- [`AeroFlow_AI_Engine.ipynb`](../AeroFlow_AI_Engine.ipynb)
- [`Aviation_Strategy_PostgreSQL_Optimization.ipynb`](../Aviation_Strategy_PostgreSQL_Optimization.ipynb)
- [`CineIntelligence_NoSQL_DataEngineering.ipynb`](../CineIntelligence_NoSQL_DataEngineering.ipynb)
- [`Clustering_Models.ipynb`](../Clustering_Models.ipynb)
- [`KDDCup.ipynb`](../KDDCup.ipynb)
- [`LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb`](../LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb)
- [`LLM_Mastery_Hands_on_Code.ipynb`](../LLM_Mastery_Hands_on_Code.ipynb)
- [`Logistic_Regression_PySpark.ipynb`](../Logistic_Regression_PySpark.ipynb)
- [`Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb`](../Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb)
- [`NYC_Airbnb_Market_Analysis (1).ipynb`](../NYC_Airbnb_Market_Analysis%20(1).ipynb)
- [`Naive_Bayes_PySpark.ipynb`](../Naive_Bayes_PySpark.ipynb)
- [`Parkinsons_Progression_ML.ipynb`](../Parkinsons_Progression_ML.ipynb)
- [`Pathfinding.ipynb`](../Pathfinding.ipynb)
- [`PyTorch_medical_AI_xray_diagnosis.ipynb`](../PyTorch_medical_AI_xray_diagnosis.ipynb)
- [`Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb`](../Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb)
- [`financial_fraud_aml_detection_system.ipynb`](../financial_fraud_aml_detection_system.ipynb)

### Export any notebook to plain Python

To create a `.py` view of **all notebooks**, including the older uploaded files, without changing the original `.ipynb` files:

```bash
python scripts/export_notebooks_to_python.py --root . --output notebook_python_exports
```

The exporter mirrors notebook paths and writes code cells in order with `# %%` markers. The originals remain untouched.

## 2. Focused skills — notebook + Python pairs

| Skill | Notebook | Python script |
| --- | --- | --- |
| Data cleaning / preprocessing | [`01_data_cleaning_preprocessing.ipynb`](../skills/01_data_cleaning_preprocessing.ipynb) | [`01_data_cleaning_preprocessing.py`](../skills/01_data_cleaning_preprocessing.py) |
| NumPy for ML | [`02_numpy_for_machine_learning.ipynb`](../skills/02_numpy_for_machine_learning.ipynb) | [`02_numpy_for_machine_learning.py`](../skills/02_numpy_for_machine_learning.py) |
| scikit-learn classification | [`03_sklearn_end_to_end_classification.ipynb`](../skills/03_sklearn_end_to_end_classification.ipynb) | [`03_sklearn_end_to_end_classification.py`](../skills/03_sklearn_end_to_end_classification.py) |
| PyTorch neural networks | [`04_pytorch_neural_network_fundamentals.ipynb`](../skills/04_pytorch_neural_network_fundamentals.ipynb) | [`04_pytorch_neural_network_fundamentals.py`](../skills/04_pytorch_neural_network_fundamentals.py) |
| LSTM sequence modelling | [`05_lstm_sequence_modelling.ipynb`](../skills/05_lstm_sequence_modelling.ipynb) | [`05_lstm_sequence_modelling.py`](../skills/05_lstm_sequence_modelling.py) |
| Text classification | [`06_text_classification_tfidf.ipynb`](../skills/06_text_classification_tfidf.ipynb) | [`06_text_classification_tfidf.py`](../skills/06_text_classification_tfidf.py) |
| CNN image fundamentals | [`07_cnn_image_fundamentals.ipynb`](../skills/07_cnn_image_fundamentals.ipynb) | [`07_cnn_image_fundamentals.py`](../skills/07_cnn_image_fundamentals.py) |

## 3. Strengthened projects — notebook + implementation

The notebook is the Jupyter/Colab walkthrough. The project folder contains the actual Python package, tests, SQL, API code and/or retained results.

| Project | Notebook | Python / project code |
| --- | --- | --- |
| Parkinson's Progression | [`project_notebook.ipynb`](../projects/parkinsons_progression/project_notebook.ipynb) | [`projects/parkinsons_progression/`](../projects/parkinsons_progression/) |
| Reliable Event Pipeline | [`project_notebook.ipynb`](../projects/reliable_event_pipeline/project_notebook.ipynb) | [`projects/reliable_event_pipeline/`](../projects/reliable_event_pipeline/) |
| Flight Delay Risk | [`project_notebook.ipynb`](../projects/flight_delay_risk/project_notebook.ipynb) | [`projects/flight_delay_risk/`](../projects/flight_delay_risk/) |
| E-commerce SQL Analytics | [`project_notebook.ipynb`](../projects/ecommerce_sql_analytics/project_notebook.ipynb) | [`projects/ecommerce_sql_analytics/`](../projects/ecommerce_sql_analytics/) |
| Customer Churn | [`project_notebook.ipynb`](../projects/customer_churn_prediction/project_notebook.ipynb) | [`projects/customer_churn_prediction/`](../projects/customer_churn_prediction/) |
| Grounded RAG | [`project_notebook.ipynb`](../projects/grounded_rag/project_notebook.ipynb) | [`projects/grounded_rag/`](../projects/grounded_rag/) |
| Image Classification | [`project_notebook.ipynb`](../projects/image_classification_confidence/project_notebook.ipynb) | [`projects/image_classification_confidence/`](../projects/image_classification_confidence/) |
| ExperimentLab | [`project_notebook.ipynb`](../projects/experiment_lab/project_notebook.ipynb) | [`projects/experiment_lab/`](../projects/experiment_lab/) |
| PySpark Clickstream | [`project_notebook.ipynb`](../projects/pyspark_clickstream_analytics/project_notebook.ipynb) | [`projects/pyspark_clickstream_analytics/`](../projects/pyspark_clickstream_analytics/) |
| UK House Price Prediction | [`project_notebook.ipynb`](../projects/uk_house_price_prediction/project_notebook.ipynb) | [`projects/uk_house_price_prediction/`](../projects/uk_house_price_prediction/) |
| Energy Demand Forecasting | [`project_notebook.ipynb`](../projects/energy_demand_forecasting/project_notebook.ipynb) | [`projects/energy_demand_forecasting/`](../projects/energy_demand_forecasting/) |
| ModelWatch | [`project_notebook.ipynb`](../projects/model_watch/project_notebook.ipynb) | [`projects/model_watch/`](../projects/model_watch/) |
| Retail Customer Segmentation | [`project_notebook.ipynb`](../projects/retail_customer_segmentation/project_notebook.ipynb) | [`projects/retail_customer_segmentation/`](../projects/retail_customer_segmentation/) |

## Why the code is not duplicated inside every notebook

For the strengthened projects, the notebook is a companion rather than a second copy of the application. Duplicating hundreds of lines into `.ipynb` and `.py` files creates two implementations that can drift apart. The project notebook instead points to and inspects the normal Python source, tests and result files.

That gives recruiters both views without weakening the engineering structure:

```text
.ipynb  → explanation / exploration / Colab review
.py     → reusable implementation / tests / normal execution
```
