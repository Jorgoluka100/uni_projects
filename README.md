# Jorgo Luka — Data & AI Portfolio

**MSc Artificial Intelligence & Data Science (Distinction), University of East London**  
Python · SQL · machine learning · applied AI · analytics

This is the work I would actually open in an interview. I keep the code, the test setup and the final numbers together so I can explain where a result came from instead of just quoting a metric. The strongest projects are below. Older university work and experiments are still in the repository, but I keep them separate from the projects I would lead with.

[![Portfolio checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)
[![Python project checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml)
[![Retail cleaning checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/retail-segmentation-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/retail-segmentation-ci.yml)

## Selected work

### Flight delay prediction and risk analysis

I started this as a regression problem, but the first version did not beat a simple baseline consistently. I changed the question rather than forcing a weak result: can schedule-time information identify flights that are more likely to arrive 15+ minutes late?

- Untouched May test set: **180,000 flights**
- PR-AUC: **0.291** vs **0.215** delay rate
- Highest-risk 10%: **1.58x** the normal delay rate
- Temporal validation, leakage checks, calibration, capacity analysis and CI

[Project](projects/flight_delay_risk/) · [Model card](projects/flight_delay_risk/MODEL_CARD.md) · [Evidence](projects/flight_delay_risk/results/verified_test_metrics.json)

### E-commerce sales and customer analysis

The SQL was not the difficult part here. The main issue was making sure orders, items, payments and reviews were joined at the right grain before trusting any revenue or customer KPI.

- **98,199** commercial orders and **94,983** customers
- **R$13.49M** merchandise value after cross-grain reconciliation
- Repeat-customer rate: **3.03%**
- SQL marts, window functions, cohorts, reconciliation and join-safety tests

[Project](projects/ecommerce_sql_analytics/) · [SQL](projects/ecommerce_sql_analytics/sql/) · [Data model](projects/ecommerce_sql_analytics/DATA_MODEL.md) · [Evidence](projects/ecommerce_sql_analytics/results/verified_summary.json)

### Retail data cleaning and customer segmentation

This project starts with the messy transaction table rather than the model. I audit the source, make row-removal rules explicit, validate the clean purchase table, build RFM features and then test the clustering solution rather than choosing a segment count by eye.

- Raw source: **541,909 transaction rows**
- Clean valid purchases: **392,692 rows** across **4,338 customers**
- Explicit duplicate, missing-ID, cancellation, quantity and price checks
- Selected KMeans: **k=2**, silhouette **0.4335**
- Initialization stability: adjusted Rand index **0.9963–0.9991** across eight additional seeds

[Project](projects/retail_customer_segmentation/) · [Project card](projects/retail_customer_segmentation/PROJECT_CARD.md) · [Evidence](projects/retail_customer_segmentation/results/verification.json)

### Customer churn prediction and retention screening

This project is about deciding who should be reviewed by a retention team, not just getting a high classification score. The model choice, probability calibration and threshold are fixed before the final holdout is scored.

- Protected holdout: **628 customers**
- PR-AUC: **0.955**; ROC-AUC: **0.990**
- Training-selected threshold: **0.15**
- Recall **94.9%**, precision **73.2%**, review rate **20.2%**
- Grouped validation, out-of-fold calibration, bootstrap uncertainty and cost-aware thresholding

[Project](projects/customer_churn_prediction/) · [Model card](projects/customer_churn_prediction/MODEL_CARD.md) · [Evidence](projects/customer_churn_prediction/results/verified_metrics.json)

### Image classification with confidence checks

I fine-tuned EfficientNet-B0 for bean-leaf classification, then added confidence checks because a model that is sometimes uncertain should be able to say so.

- Test accuracy: **85.9%**; macro-F1: **85.8%**
- Accuracy 95% bootstrap interval: **79.7%–91.4%**
- **90.4%** accuracy on accepted predictions while routing **10.9%** to review
- Grad-CAM plus verified TorchScript and ONNX parity

[Project](projects/image_classification_confidence/) · [Model card](projects/image_classification_confidence/MODEL_CARD.md) · [Evidence](projects/image_classification_confidence/results/verified_metrics.json)

## More verified work

- **[UK house price prediction](projects/uk_house_price_prediction/):** 995,059 modelling transactions and an untouched 216,564-sale 2026 test set; CatBoost MAE **£81,805** vs **£82,804** for a strong postcode/property baseline. The improvement is small, so I report it that way.
- **[Energy demand forecasting](projects/energy_demand_forecasting/):** 14-day TensorFlow forecast with MAE **43.51 GWh** vs **53.18 GWh** for the 7-day seasonal baseline, an **18.2%** improvement; includes validation-based uncertainty intervals and model reload checks.
- **[PySpark clickstream analytics](projects/pyspark_clickstream_analytics/):** **165,474** real events across **24,026** sessions plus a separately labelled one-million-row replicated load test; conversion model PR-AUC **0.351** vs **0.155** prevalence.
- **[Retrieval-augmented support assistant](projects/grounded_rag/):** local RAG application with source attribution, weak-match abstention, read-only FastAPI tooling and prompt-injection checks. Perfect scores on the small frozen fixture are treated as software checks, not real-world LLM performance.
- **[A/B test analysis](projects/experiment_lab/):** treatment-effect estimation, confidence intervals, CUPED, guardrails and power calculations on a simulated dataset with a known effect; CUPED reduced retained-run variance by **50.7%**.
- **[Model monitoring and drift detection](projects/model_watch/):** feature drift, discrimination and calibration checks with deliberately shifted batches so the alert logic can be tested.
- **NYC Airbnb 2026:** MAE **$68.97** vs **$121.90** median baseline on unseen neighbourhoods.
- **Movie recommendation:** Recall@10 **0.501** vs **0.463** popularity baseline after removing future-interaction leakage.
- **London air quality:** R analysis using official 2025–2026 monitoring data.

For smaller analysis exercises, see my [Data Analyst Bootcamp repository](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp).

## What I check before I trust a result

- Where the data came from and whether basic quality checks pass
- Whether the train/test split matches the way the model would be used
- Leakage and duplicate-profile checks where they matter
- A baseline that is genuinely worth beating
- Threshold selection before opening the final holdout
- Error analysis, uncertainty and limitations
- Machine-readable result files so the headline numbers can be checked
- Tests or self-checks for packaged projects
- GitHub Actions for the main project folders

## Repository guide

- [`projects/`](projects/) — the projects I would lead with in an interview.
- [`verified/`](verified/) — saved result files and verification outputs from earlier project versions.
- [`extensions/`](extensions/) — rerun or verification code for older notebooks.
- [`docs/PROJECT_CATALOG.md`](docs/PROJECT_CATALOG.md) — complete repository catalogue.
- Root `.ipynb` files — executed notebooks, university work and older learning projects.

Some older medical-imaging, LLM and coursework notebooks remain because they show what I was learning at the time. I do **not** use unverified results from those notebooks as production or clinical claims.

<details>
<summary><strong>Complete notebook inventory</strong></summary>

### Verified notebook projects

- [`01_UK_House_Price_Analysis_and_Prediction.ipynb`](01_UK_House_Price_Analysis_and_Prediction.ipynb)
- [`02_SQL_Sales_and_Customer_Analysis.ipynb`](02_SQL_Sales_and_Customer_Analysis.ipynb)
- [`03_Customer_Churn_Prediction.ipynb`](03_Customer_Churn_Prediction.ipynb)
- [`04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb`](04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb)
- [`05_Energy_Demand_Forecasting_with_TensorFlow.ipynb`](05_Energy_Demand_Forecasting_with_TensorFlow.ipynb)
- [`06_Clickstream_Analysis_with_PySpark.ipynb`](06_Clickstream_Analysis_with_PySpark.ipynb)
- [`07_London_Air_Quality_Analysis_with_R.ipynb`](07_London_Air_Quality_Analysis_with_R.ipynb)
- [`01_ConsultAI_AI_Opportunity_Engine.ipynb`](01_ConsultAI_AI_Opportunity_Engine.ipynb)
- [`12_VisionForge_PyTorch_Visual_Inspection.ipynb`](12_VisionForge_PyTorch_Visual_Inspection.ipynb)

### Advanced / historical laboratory notebooks

- [`Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb`](Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb)
- [`AeroFlow_AI_Engine.ipynb`](AeroFlow_AI_Engine.ipynb)
- [`Aviation_Strategy_PostgreSQL_Optimization.ipynb`](Aviation_Strategy_PostgreSQL_Optimization.ipynb)
- [`CineIntelligence_NoSQL_DataEngineering.ipynb`](CineIntelligence_NoSQL_DataEngineering.ipynb)
- [`Clustering_Models.ipynb`](Clustering_Models.ipynb)
- [`KDDCup.ipynb`](KDDCup.ipynb)
- [`LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb`](LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb)
- [`LLM_Mastery_Hands_on_Code.ipynb`](LLM_Mastery_Hands_on_Code.ipynb)
- [`Logistic_Regression_PySpark.ipynb`](Logistic_Regression_PySpark.ipynb)
- [`Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb`](Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb)
- [`NYC_Airbnb_Market_Analysis (1).ipynb`](NYC_Airbnb_Market_Analysis%20%281%29.ipynb)
- [`Naive_Bayes_PySpark.ipynb`](Naive_Bayes_PySpark.ipynb)
- [`Parkinsons_Progression_ML.ipynb`](Parkinsons_Progression_ML.ipynb)
- [`Pathfinding.ipynb`](Pathfinding.ipynb)
- [`PyTorch_medical_AI_xray_diagnosis.ipynb`](PyTorch_medical_AI_xray_diagnosis.ipynb)
- [`Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb`](Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb)
- [`financial_fraud_aml_detection_system.ipynb`](financial_fraud_aml_detection_system.ipynb)

</details>

## Main tools

Python, SQL, Pandas, NumPy, scikit-learn, PyTorch, TensorFlow/Keras, CatBoost, DuckDB, PostgreSQL, PySpark, FastAPI, Docker and GitHub Actions.

## Licence

My own code and documentation are MIT-licensed. External datasets and pretrained models retain their original licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
