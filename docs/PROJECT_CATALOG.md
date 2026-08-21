# Project Catalog

This file is the complete inventory used by the portfolio integrity gate. The root README is intentionally recruiter-first and does not list every historical notebook in detail.

## Verified notebook flagships

- `01_UK_House_Price_Analysis_and_Prediction.ipynb` — original executed market analysis and training record retained alongside `projects/uk_house_price_prediction/`.
- `02_SQL_Sales_and_Customer_Analysis.ipynb` — original executed exploration retained alongside the production-style `projects/ecommerce_sql_analytics/` package.
- `03_Customer_Churn_Prediction.ipynb` — original executed modelling/evaluation retained alongside `projects/customer_churn_prediction/`.
- `04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb`
- `05_Energy_Demand_Forecasting_with_TensorFlow.ipynb` — original executed forecasting/training record retained alongside `projects/energy_demand_forecasting/`.
- `06_Clickstream_Analysis_with_PySpark.ipynb`
- `07_London_Air_Quality_Analysis_with_R.ipynb`
- `01_ConsultAI_AI_Opportunity_Engine.ipynb`
- `12_VisionForge_PyTorch_Visual_Inspection.ipynb` — original executed training/evaluation retained alongside `projects/image_classification_confidence/`.

## Advanced / laboratory notebooks

These are retained for inspection and learning but are not automatically promoted as verified performance evidence.

- `Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb`
- `AeroFlow_AI_Engine.ipynb` — historical predecessor to the verified `projects/flight_delay_risk/` package.
- `Aviation_Strategy_PostgreSQL_Optimization.ipynb`
- `CineIntelligence_NoSQL_DataEngineering.ipynb`
- `Clustering_Models.ipynb`
- `KDDCup.ipynb`
- `LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb`
- `LLM_Mastery_Hands_on_Code.ipynb`
- `Logistic_Regression_PySpark.ipynb`
- `Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb`
- `NYC_Airbnb_Market_Analysis (1).ipynb`
- `Naive_Bayes_PySpark.ipynb`
- `Parkinsons_Progression_ML.ipynb`
- `Pathfinding.ipynb`
- `PyTorch_medical_AI_xray_diagnosis.ipynb`
- `Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb`
- `financial_fraud_aml_detection_system.ipynb`

## Production-style projects

- `projects/flight_delay_risk/` — official BTS 2026 data, temporal holdout, leakage-safe schedule-time features, CatBoost ranking, validation-selected review capacity, calibration checks, carrier slices, tests and retained evidence.
- `projects/ecommerce_sql_analytics/` — pinned Olist v7 source, explicit order/item semantic grains, DuckDB marts, financial reconciliation, cohorts, window functions, marketplace operations, tests and retained evidence.
- `projects/image_classification_confidence/` — EfficientNet-B0 evaluation, bootstrap uncertainty, calibration, selective prediction, Grad-CAM, export-parity checks, tests, model card and retained evidence.
- `projects/uk_house_price_prediction/` — official HM Land Registry 2025–2026 data, strict temporal holdout, strong geographical/property baseline, CatBoost, residual uncertainty intervals, tests, model card and retained evidence.
- `projects/customer_churn_prediction/` — pinned UCI source, duplicate-profile grouped holdout, proxy-feature exclusion, histogram gradient boosting, grouped OOF calibration, cost-aware retention threshold, bootstrap uncertainty, tests, model card and retained evidence.
- `projects/energy_demand_forecasting/` — chronological 60-to-14-day forecasting, seasonal baselines, Conv1D+LSTM, validation-calibrated residual intervals, artifact reload parity, tests, model card and retained evidence.
- `projects/grounded_rag/` — hybrid sparse+dense retrieval, deterministic LSA vector indexing, grounded citations, abstention, allow-listed read-only tool routing, prompt-injection blocking, FastAPI and Docker.
- `projects/careerlens_ai/` — NLP / information retrieval and skill-gap ranking.
- `projects/experiment_lab/` — experimentation, CUPED, bootstrap uncertainty, guardrails and power.
- `projects/model_watch/` — drift, calibration, performance monitoring and retraining policy.

## Hardened extensions and evidence

Additional hardened runners live under `extensions/`; machine-readable evidence lives under `verified/` or inside an individual production project when that keeps the code and its retained evidence easier to inspect together. The integrity workflow treats `verification_pass=true` as a minimum contract, not a substitute for interpreting scope and limitations.
