# AI & Data Science Projects

Hi, I'm Jorgo. I completed an MSc in Artificial Intelligence & Data Science with Distinction, and I'm looking for my first role in data science, machine learning, AI or analytics.

This repo is where I keep the projects I would be comfortable talking through in an interview. I have gone back over a lot of my earlier work to fix things such as data leakage, weak train/test splits and missing baselines. I also keep failed approaches when they taught me something useful instead of hiding them.

[![Portfolio checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)
[![Python project checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml)

## Projects I would start with

| Project | What I did | Result | Tools |
|---|---|---|---|
| **[Retrieval-Augmented Generation Application](projects/grounded_rag/)** | Built a small RAG system that searches policy documents, returns sources, refuses weak answers and can route a read-only ticket analytics tool. I also added tests for prompt-injection attempts. | All retrieval, routing and safety checks pass on the small frozen test set. I treat these as software tests, not as proof of real-world LLM performance. | Python, BM25, TF-IDF, LSA, FastAPI, Docker |
| **[Flight Delay Prediction & Risk Analysis](extensions/aeroflow_delay_risk_v3.py)** | Used official 2026 US flight data to predict which flights are most likely to arrive 15+ minutes late. My first regression version did not beat its baseline, so I changed the problem to classification/ranking. | PR-AUC **0.291** vs **0.215** delay rate; the highest-risk 10% of flights had **1.58x** the normal delay rate on the untouched May test set. | Python, CatBoost, Pandas |
| **[Image Classification with EfficientNet](12_VisionForge_PyTorch_Visual_Inspection.ipynb)** | Fine-tuned an EfficientNet model for bean-leaf image classification, then added confidence checks, Grad-CAM and model export tests. | **85.9%** test accuracy, **85.8%** macro-F1; **90.4%** accuracy on the predictions the model accepted at the chosen confidence threshold. | PyTorch, EfficientNet, Grad-CAM, ONNX |
| **[E-commerce Customer & Sales Analysis](02_SQL_Sales_and_Customer_Analysis.ipynb)** | Cleaned and modelled an e-commerce dataset, checked keys and joins, and built customer and sales reporting queries. | **98,199** orders, **94,983** customers and **R$13.49M** merchandise value after reconciliation checks. | SQL, DuckDB, Pandas |
| **[A/B Testing & Experiment Analysis](projects/experiment_lab/)** | Built a small A/B-test analysis project with CUPED, confidence intervals, guardrails and power calculations. | On the simulated 20,000-row experiment, CUPED reduced outcome variance by **50.7%**. | Python, statistics, bootstrap, CUPED |
| **[Model Monitoring & Drift Detection](projects/model_watch/)** | Built a monitoring demo for drift, model performance and calibration. | Stable data stays green while the deliberately shifted batches trigger the expected alerts. | Python, PSI, KS, ROC/PR, Brier, ECE |

## More data science work

- **UK house prices:** 995,059 modelling transactions with an untouched 2026 test set of 216,564 sales. CatBoost reached MAE **£81,805** and R² **0.604**.
- **Customer churn:** PR-AUC **0.955** and ROC-AUC **0.990** on the retained test results.
- **NYC Airbnb 2026:** MAE **$68.97** compared with a **$121.90** median baseline on unseen neighbourhoods.
- **Movie recommender:** found and removed future-interaction leakage before rerunning the ranking evaluation; Recall@10 **0.501** vs **0.463** popularity baseline.
- **Energy forecasting:** TensorFlow model MAE **43.51** vs **53.18** seasonal baseline.
- **PySpark clickstream:** **165,474** events across **24,026** sessions; PR-AUC **0.351** vs **0.155** positive rate.
- **London air quality:** R analysis using official 2025–2026 monitoring data.

My separate [Data Analyst Bootcamp repository](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp) has three smaller projects focused on cleaning, EDA, regression, SQL and business reporting.

## How I work on projects

I try to keep the process simple:

1. Check the data before modelling.
2. Keep training and test data separate in a way that matches the real problem.
3. Compare against a simple baseline.
4. Save the actual results rather than quoting numbers I cannot reproduce.
5. Look at errors and limitations, not only the best metric.
6. Keep the code understandable enough that I can explain it without relying on the notebook output.

The `verified/` folder contains saved result files for projects that have separate checks, and the GitHub Actions workflows rerun the main repository and Python-project checks.

## Main tools used in this repo

Python, SQL, R, PySpark, Pandas, NumPy, scikit-learn, CatBoost, TensorFlow/Keras, PyTorch, PostgreSQL, DuckDB, Spark SQL, FastAPI, Docker, GitHub Actions, NLP / information retrieval, computer vision, recommendation systems, time series, experimentation and model monitoring.

## Repository layout

- [`projects/`](projects/) — smaller Python projects.
- [`verified/`](verified/) — saved result and test files.
- [`extensions/`](extensions/) — rerun / verification code added to some notebook projects.
- [`docs/PROJECT_CATALOG.md`](docs/PROJECT_CATALOG.md) — full project list.
- Root `.ipynb` files — the original notebook projects.

## Older / laboratory notebooks

Some older notebooks are kept because they show what I was learning at the time. I do not use their results on my CV unless I have gone back and checked the evaluation properly. In particular, the medical-imaging and older LLM-training notebooks should be treated as learning projects rather than clinical or production claims.

<details>
<summary><strong>Complete notebook list used by the repository checks</strong></summary>

Verified notebooks: `01_UK_House_Price_Analysis_and_Prediction.ipynb` · `02_SQL_Sales_and_Customer_Analysis.ipynb` · `03_Customer_Churn_Prediction.ipynb` · `04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb` · `05_Energy_Demand_Forecasting_with_TensorFlow.ipynb` · `06_Clickstream_Analysis_with_PySpark.ipynb` · `07_London_Air_Quality_Analysis_with_R.ipynb` · `01_ConsultAI_AI_Opportunity_Engine.ipynb` · `12_VisionForge_PyTorch_Visual_Inspection.ipynb`

Other notebooks: `Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb` · `AeroFlow_AI_Engine.ipynb` · `Aviation_Strategy_PostgreSQL_Optimization.ipynb` · `CineIntelligence_NoSQL_DataEngineering.ipynb` · `Clustering_Models.ipynb` · `KDDCup.ipynb` · `LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb` · `LLM_Mastery_Hands_on_Code.ipynb` · `Logistic_Regression_PySpark.ipynb` · `Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb` · `NYC_Airbnb_Market_Analysis (1).ipynb` · `Naive_Bayes_PySpark.ipynb` · `Parkinsons_Progression_ML.ipynb` · `Pathfinding.ipynb` · `PyTorch_medical_AI_xray_diagnosis.ipynb` · `Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb` · `financial_fraud_aml_detection_system.ipynb`

</details>

## Licence

My own code and documentation in this repository are MIT-licensed. External datasets and pretrained models keep their original licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
