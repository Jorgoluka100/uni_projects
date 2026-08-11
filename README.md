# AI & Data Science Portfolio

## Jorgo Luka — MSc Artificial Intelligence & Data Science (Distinction)

I build end-to-end data and AI projects that start with a real decision, audit and clean the source data, establish an honest baseline, evaluate the result properly, and communicate limitations instead of hiding them.

This repository deliberately contains **two levels of work**:

- **Verified flagships** — executed end to end with retained outputs and defensible measured results.
- **Advanced project laboratory** — substantial AI, ML, data-engineering and optimisation notebooks that are useful evidence of breadth. These projects are kept on `main`, quality-checked as notebook/code artefacts, and are being promoted to verified status only after a clean full rerun and evidence review.

That distinction lets the portfolio keep good code without presenting unverified outputs as proven results.

## Start here — verified flagships

| Project | Problem solved | Verified evidence | Main tools | Open |
|---|---|---|---|---|
| **UK House Price Analysis and Prediction** | Produce a forward-looking residential price range while being explicit about uncertainty and missing property characteristics. | 995,059 modelling transactions; 216,564-sale untouched 2026 test; CatBoost MAE **£81,805**, R² **0.604**; 90% interval coverage **91.6%**. | Python, Pandas, NumPy, DuckDB SQL, CatBoost | [Notebook](01_UK_House_Price_Analysis_and_Prediction.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/01_UK_House_Price_Analysis_and_Prediction.ipynb) |
| **SQL Sales and Customer Analysis** | Build trustworthy marketplace KPIs from nine relational tables without multiplying money across incompatible grains. | **98,199** commercial orders, **94,983** customers and **R$13.49M** merchandise value reconciled; source hashes, key checks, semantic-layer tests and artifact tests all passed. | SQL, DuckDB, Python, Pandas, Parquet | [Notebook](02_SQL_Sales_and_Customer_Analysis.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/02_SQL_Sales_and_Customer_Analysis.ipynb) |
| **Customer Churn Prediction** | Build a decision-ready churn screening system while controlling duplicate-profile leakage, calibration and review capacity. | Holdout PR-AUC **0.955** (95% bootstrap **0.927–0.980**) and ROC-AUC **0.990**; **94.9%** recall at **73.2%** precision while flagging **20.2%** of customers. | Python, scikit-learn, calibration, bootstrap diagnostics | [Notebook](03_Customer_Churn_Prediction.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/03_Customer_Churn_Prediction.ipynb) |
| **Image Classification with CNNs and Transfer Learning** | Classify pet breeds and route uncertain predictions to human review instead of automating every case. | 7,349 real images and 37 breeds; test accuracy **58.5%**, macro-F1 **56.3%**, top-3 accuracy **83.1%**; accepted-case accuracy **80.9%** at **49.1%** coverage. | PyTorch, torchvision, MobileNetV3, Grad-CAM | [Notebook](04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) |
| **Energy Demand Forecasting with TensorFlow** | Forecast a chronological 14-day electricity-demand window and test whether the neural model actually beats honest seasonal baselines. | LSTM test MAE **43.51** versus **53.18** for the 7-day seasonal baseline — an **18.2%** improvement — with prediction intervals and saved-model verification. | TensorFlow, Keras, LSTM, time-series validation | [Notebook](05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) |
| **Clickstream Analysis with PySpark** | Measure engagement drop-off at Spark scale and identify higher-intent shopping sessions without using post-purchase leakage. | 165,474 real click events across 24,026 sessions; separate 12,330-session conversion dataset; test PR-AUC **0.351** versus **0.155** prevalence and ROC-AUC **0.763**. | PySpark, Spark SQL, Spark ML | [Notebook](06_Clickstream_Analysis_with_PySpark.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/06_Clickstream_Analysis_with_PySpark.ipynb) |
| **London Air Quality Analysis with R** | Turn messy, daily-updated government files into an auditable monitoring panel and test whether past-only features improve hourly NO₂ estimates. | Official 2025–2026 data; 143,102 observed station-hours; untouched 2026 MAE **10.48** and R² **0.056**; 90% interval coverage **88.2%**. | R, data.table, ggplot2, mgcv | [Notebook](07_London_Air_Quality_Analysis_with_R.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/07_London_Air_Quality_Analysis_with_R.ipynb) |

## Advanced project laboratory

These notebooks are intentionally visible because the code adds useful hiring signals. They are **not given verified performance claims in this README until they pass a fresh end-to-end rerun**.

| Project | Capability demonstrated | Open |
|---|---|---|
| **ConsultAI — AI Opportunity Prioritisation Engine** | Decision science, Monte Carlo uncertainty, constrained portfolio optimisation, governance artefacts and application engineering | [Notebook](01_ConsultAI_AI_Opportunity_Engine.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/01_ConsultAI_AI_Opportunity_Engine.ipynb) |
| **VisionForge — PyTorch Visual Inspection** | Computer vision, PyTorch, industrial-style visual inspection workflow | [Notebook](12_VisionForge_PyTorch_Visual_Inspection.ipynb) |
| **Advanced Multi-Modal Health Analytics Suite** | Multi-modal health analytics and model integration | [Notebook](Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb) |
| **AeroFlow AI Engine** | Aviation intelligence, predictive modelling and decision support | [Notebook](AeroFlow_AI_Engine.ipynb) |
| **Aviation Strategy PostgreSQL Optimisation** | PostgreSQL analytics, query optimisation and aviation strategy | [Notebook](Aviation_Strategy_PostgreSQL_Optimization.ipynb) |
| **CineIntelligence NoSQL Data Engineering** | NoSQL/data-engineering concepts applied to entertainment data | [Notebook](CineIntelligence_NoSQL_DataEngineering.ipynb) |
| **Clustering Models** | Unsupervised learning, clustering comparison and segmentation | [Notebook](Clustering_Models.ipynb) |
| **KDD Cup Analysis** | Large-dataset ML workflow and benchmark-style experimentation | [Notebook](KDDCup.ipynb) |
| **LLM Mastery — Alignment** | LLM alignment concepts and hands-on implementation | [Notebook](LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb) |
| **LLM Mastery — Core** | Transformer/LLM implementation and experimentation | [Notebook](LLM_Mastery_Hands_on_Code.ipynb) |
| **Logistic Regression with PySpark** | Distributed classification with Spark ML | [Notebook](Logistic_Regression_PySpark.ipynb) |
| **Hybrid Deep-Learning Movie Recommender** | Recommendation systems and hybrid deep-learning pipeline design | [Notebook](Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb) |
| **NYC Airbnb Market Analysis** | EDA, market analytics and applied modelling | [Notebook](NYC_Airbnb_Market_Analysis%20(1).ipynb) |
| **Naive Bayes with PySpark** | Distributed probabilistic classification with Spark ML | [Notebook](Naive_Bayes_PySpark.ipynb) |
| **Parkinson's Progression ML** | Healthcare regression/progression modelling and evaluation | [Notebook](Parkinsons_Progression_ML.ipynb) |
| **Pathfinding** | Search algorithms, optimisation and algorithmic reasoning | [Notebook](Pathfinding.ipynb) |
| **PyTorch Medical AI — X-ray Diagnosis** | Medical imaging, deep learning and explainability-oriented workflow | [Notebook](PyTorch_medical_AI_xray_diagnosis.ipynb) |
| **Strategic Telecom Churn + Predictive SQL** | Churn analytics, SQL and commercial decision support | [Notebook](Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb) |
| **Financial Fraud / AML Detection System** | Imbalanced classification, fraud risk and AML-oriented analytics | [Notebook](financial_fraud_aml_detection_system.ipynb) |

## Completion standard

A notebook is promoted from the advanced laboratory to **verified flagship** only when it passes all of the following:

1. **Decision framing** — clear user, business/technical decision, target and failure cost.
2. **Data provenance** — source, licence/usage constraints, date coverage and freshness stated explicitly.
3. **Data quality** — schema, missingness, duplicates, invalid ranges and join integrity checked.
4. **Leakage control** — preprocessing fitted on training data only; temporal/group leakage addressed where relevant.
5. **Baseline** — complex models must beat or materially complement a simple benchmark.
6. **Untouched evaluation** — a genuine held-out test set with decision-relevant metrics.
7. **Uncertainty/error analysis** — calibration, intervals, slices, confidence, failure modes or human-review policy where appropriate.
8. **Reproducibility** — deterministic seeds/config, dependency/setup guidance and clean restart/run-all behaviour.
9. **Engineering evidence** — tests, saved artefacts, reload/smoke check, or an API/app where it adds value.
10. **Interview defence** — limitations, what the result does not prove, and the next production step.

The repository validator in [`scripts/validate_portfolio.py`](scripts/validate_portfolio.py) treats verified and advanced projects differently: verified projects must retain executed evidence, while advanced projects must remain valid, documented, non-broken notebooks and are reported separately until fully rerun.

## Data-freshness notes

- **Customer Churn Prediction** uses the historical UCI Iranian Churn dataset. It demonstrates rigorous classification and decision design, not current telecom-market behaviour.
- **Energy Demand Forecasting** uses the official Open Power System Data series from 2006–2017. It is a time-series methodology project, not a description of today's German electricity market.
- Current official data are preferred where the problem supports them; older datasets are labelled rather than presented as current evidence.

## How I build a project

1. Define the decision, user, target, and failure cost.
2. Document source provenance, licence, time coverage, and known limitations.
3. Audit missingness, duplicates, invalid values, ranges, types, and join integrity.
4. Fit preprocessing on training data only and prevent temporal, group, or target leakage.
5. Compare against a simple baseline before selecting a more complex model.
6. Reserve an untouched test set and report metrics that match the decision.
7. Add uncertainty, error analysis, slices, or a human-review policy where appropriate.
8. Reload the saved artifact and run automated acceptance checks.
9. State what the result does **not** prove.

## Skills evidenced

| Capability | Evidence across the portfolio |
|---|---|
| Data cleaning and preprocessing | Explicit schemas, missing-value audits, duplicate controls, range checks, categorical handling and training-only transforms |
| SQL and analytical querying | DuckDB SQL, PostgreSQL, relational grain controls, reconciliation, Spark SQL and session analytics |
| Classical machine learning | CatBoost, gradient boosting, regularised regression, Naive Bayes, clustering, baselines, calibration and error analysis |
| Deep learning | PyTorch, TensorFlow/Keras, CNN transfer learning, LSTMs, medical imaging and recommendation systems |
| LLM / applied AI | LLM implementation/alignment experiments and AI opportunity decision engineering |
| Big-data processing | PySpark DataFrames, windows, sessionisation, Spark SQL and Spark ML |
| Statistics and R | Time-aware modelling, GAMs, conformal intervals, data.table and ggplot2 |
| Decision / governance engineering | Monte Carlo analysis, constrained optimisation, uncertainty, human escalation, audit artefacts and limitations |

## Additional analytics work

My separate [Data Analyst Bootcamp Portfolio](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp) contains applied Python, SQL, data cleaning, regression, reporting and business-analysis work. Its strongest executed case studies are organised under [`notebooks/`](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp/tree/main/notebooks) with source data under [`data/`](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp/tree/main/data).

## Evidence policy

- A result is called **verified** only when the notebook has been run end to end, retains the produced outputs, contains no stored error, and passes its documented checks.
- Advanced notebooks stay visible because breadth and code quality matter, but they do not receive unverified metric claims.
- A complicated model is not automatically better. Every model must earn its place against a relevant baseline.
- Archive branches remain as safety snapshots of earlier portfolio states.
- A project is promoted only when its claims are defensible in a technical interview.

## Current focus

The next upgrades are **depth, execution evidence and production hardening**, not deleting useful work or adding random notebook count. The strongest advanced projects will be rerun, audited against the completion standard above, and then promoted with verified results.

For opportunities in data science, machine learning, applied AI, or analytics, contact me through my [GitHub profile](https://github.com/Jorgoluka100).
