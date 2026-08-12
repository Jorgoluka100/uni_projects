# Jorgo Luka — AI & Data Science Portfolio

**MSc Artificial Intelligence & Data Science (Distinction)**

[![Portfolio integrity](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)
[![New production projects](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml)

I build data and AI systems around a real decision: audit the data, prevent leakage, establish a baseline, evaluate on held-out evidence, retain reproducible artifacts, and state limitations rather than protecting an attractive metric.

**Recruiter shortcut:** [`docs/HIRING_PLAYBOOK.md`](docs/HIRING_PLAYBOOK.md) · **Production-style projects:** [`projects/`](projects/) · **Verification evidence:** [`verified/`](verified/)

## Start here — strongest verified work

| Project | What it proves | Verified evidence | Stack |
|---|---|---|---|
| **VisionForge — Trustworthy Visual Inspection** | End-to-end CV engineering with uncertainty and deployment evidence. | Makerere Beans test accuracy **85.9%**, macro-F1 **85.8%** (95% bootstrap **79.2–91.3%**); selective accuracy **90.4%** at **89.1%** coverage; independent checkpoint recomputation plus TorchScript/ONNX parity. [Open](12_VisionForge_PyTorch_Visual_Inspection.ipynb) · [Evidence](verified/visionforge/verification_metrics.json) | PyTorch, EfficientNet, calibration, Grad-CAM, ONNX, TorchScript |
| **AeroFlow — 2026 Flight Delay Risk** | Current-data temporal modelling and problem reframing when the first model fails its baseline. | Official BTS 2026: **360k** train / **120k** validation / **180k** untouched May test; PR-AUC **0.291** vs **0.215** prevalence; top-risk decile **1.58× lift**. [v3](extensions/aeroflow_delay_risk_v3.py) · [Evidence](verified/aeroflow_delay_risk/verification.json) | Python, CatBoost, temporal validation, ranking |
| **UK House Price Analysis** | Large-scale regression with honest uncertainty. | **995,059** modelling transactions; **216,564-sale** untouched 2026 test; MAE **£81,805**, R² **0.604**, 90% interval coverage **91.6%**. [Notebook](01_UK_House_Price_Analysis_and_Prediction.ipynb) | Python, DuckDB, CatBoost |
| **NYC Airbnb — 2026 Refresh** | Current market analysis with group holdout rather than random neighbourhood mixing. | 14-Jun-2026 Inside Airbnb snapshot; unseen-neighbourhood test MAE **$68.97** vs **$121.90** baseline (**43.4% improvement**), R² **0.483**. [v2](extensions/airbnb_nyc_2026_v2.py) · [Evidence](verified/airbnb_nyc_2026/verification.json) | Python, Pandas, scikit-learn |
| **Movie Recommender — Temporal Ranking** | Recommendation evaluation without future-history leakage. | **609** held-out users; **1,756** future interactions removed; Recall@10 **0.501** vs **0.463** popularity baseline, NDCG@10 **0.336** vs **0.293**. [v2](extensions/recommender_v2.py) · [Evidence](verified/recommender/verification.json) | Python, SciPy, ranking metrics |
| **SQL Sales & Customer Analysis** | Relational grain control and trustworthy commercial KPIs. | **98,199** commercial orders, **94,983** customers and **R$13.49M** merchandise value reconciled; semantic-layer and source-integrity checks passed. [Notebook](02_SQL_Sales_and_Customer_Analysis.ipynb) | SQL, DuckDB, Pandas, Parquet |
| **Customer Churn Prediction** | Calibration, capacity-aware screening and leakage control. | PR-AUC **0.955** (95% bootstrap **0.927–0.980**), ROC-AUC **0.990**; **94.9%** recall at **73.2%** precision. [Notebook](03_Customer_Churn_Prediction.ipynb) | Python, scikit-learn, calibration |
| **Clickstream with PySpark** | Distributed session analytics and Spark ML. | **165,474** click events / **24,026** sessions; separate conversion data; PR-AUC **0.351** vs **0.155** prevalence, ROC-AUC **0.763**. [Notebook](06_Clickstream_Analysis_with_PySpark.ipynb) | PySpark, Spark SQL, Spark ML |

## New production-style projects

These were added specifically to fill gaps that the original notebook portfolio did not cover strongly: **NLP retrieval, experimentation/causal decision science, and MLOps monitoring**. Each has a clean runner, self-test, retained evidence and a Colab launcher.

| Project | Decision | Verified evidence | Run |
|---|---|---|---|
| **CareerLens AI** | Rank jobs for a candidate and expose matched/missing skills without pretending to predict hiring. | Deterministic 4-query retrieval benchmark: **MRR 1.000**, **Recall@5 1.000**, **NDCG@5 0.984**. Built-in corpus is explicitly synthetic; real job CSVs can be supplied. | [Code](projects/careerlens_ai/run.py) · [Evidence](verified/careerlens_ai/verification.json) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/projects/careerlens_ai/CareerLens_AI.ipynb) |
| **ExperimentLab** | Decide whether a randomized product change should ship using effect uncertainty, CUPED, guardrails and power. | **20,000** simulated observations; known effect 2.5, CUPED estimate **2.773** with 95% CI **2.435–3.112**; **50.7% variance reduction**; bootstrap and guardrail checks passed. Synthetic by design. | [Code](projects/experiment_lab/run.py) · [Evidence](verified/experiment_lab/verification.json) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/projects/experiment_lab/ExperimentLab.ipynb) |
| **ModelWatch** | Detect when a deployed classifier needs investigation/retraining under drift. | Stable batch max PSI **0.003** → green; feature-shift PSI **0.264** → red; concept-shift PSI **0.466**, ROC-AUC drop **0.049**, ECE **0.058** → red; saved-model reload parity passed. | [Code](projects/model_watch/run.py) · [Evidence](verified/model_watch/verification.json) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/projects/model_watch/ModelWatch.ipynb) |

## Other verified engineering / methodology evidence

| Area | Evidence |
|---|---|
| **PostgreSQL optimisation** | Genuine PostgreSQL 17, deterministic **1,000,000-row** aviation workload, before/after `EXPLAIN (ANALYZE, BUFFERS)`, targeted index observed and reconciliation passed. [Evidence](verified/aviation_postgres/verification.json) |
| **Distributed classification** | **145,585** deduplicated KDD rows, train-only PySpark pipeline, validation selection, untouched test and serialized `PipelineModel` reload. Historical benchmark only. [Evidence](verified/spark_kdd/verification.json) |
| **Experiment honesty** | Parkinson's complete-subject holdout exposed poor generalisation: RF MAE **10.70** vs **8.13** median baseline. The negative result is retained rather than hidden. [Evidence](verified/parkinsons_grouped/verification.json) |
| **MLOps / decision systems** | Fraud/AML chronological screening, telecom capacity-aware churn, clustering stability, pathfinding optimality, NoSQL parsing/quarantine, LLM evaluation harness and deterministic ConsultAI optimisation all have retained evidence under [`verified/`](verified/). |
| **Deep learning / R / time series** | TensorFlow Energy MAE **43.51** vs **53.18** seasonal baseline; London Air Quality uses official 2025–2026 data; additional PyTorch transfer-learning work remains inspectable. |

## What remains deliberately unpromoted

Only projects that still need genuinely project-specific evidence stay in the laboratory tier:

- `Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb` — requires defensible patient/group-safe provenance and non-clinical evidence.
- `PyTorch_medical_AI_xray_diagnosis.ipynb` — same healthcare safety/evidence requirement.
- `LLM_Mastery_Hands_on_Code.ipynb` and `LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb` — evaluator is verified, but fresh current-checkpoint model-quality evidence is still required.

## Evidence policy

1. Define the decision and failure cost.
2. Document source, licence/usage constraints, dates and freshness.
3. Audit missingness, duplicates, ranges, schema and join grain.
4. Prevent target, temporal and group leakage.
5. Establish a simple baseline before trusting complexity.
6. Reserve genuinely held-out evidence where the task supports it.
7. Add uncertainty, calibration, ranking, slices or human-review policy where appropriate.
8. Save/reload or independently reconstruct important artifacts.
9. Retain negative results when they change the modelling decision.
10. Quote only metrics that can be defended in an interview.

The repository-wide [`Portfolio integrity`](.github/workflows/portfolio-integrity.yml) workflow compile-checks Python code, self-tests critical gates, validates retained notebook evidence, and separately enforces the three new production-project verification records.

## Skills evidenced

**Python · SQL · R · PySpark · Pandas · NumPy · scikit-learn · SciPy · CatBoost · TensorFlow/Keras · PyTorch · PostgreSQL · DuckDB · Spark SQL · NLP/information retrieval · experimentation/CUPED · MLOps monitoring/drift · computer vision · recommendation systems · time-series modelling · classification · regression · clustering · calibration · uncertainty · model persistence · CI/evidence gates · data cleaning/preprocessing · decision engineering**

## Preserved notebook index

The original notebook work remains available rather than being deleted during curation:

`01_ConsultAI_AI_Opportunity_Engine.ipynb` · `01_UK_House_Price_Analysis_and_Prediction.ipynb` · `02_SQL_Sales_and_Customer_Analysis.ipynb` · `03_Customer_Churn_Prediction.ipynb` · `04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb` · `05_Energy_Demand_Forecasting_with_TensorFlow.ipynb` · `06_Clickstream_Analysis_with_PySpark.ipynb` · `07_London_Air_Quality_Analysis_with_R.ipynb` · `12_VisionForge_PyTorch_Visual_Inspection.ipynb` · `Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb` · `AeroFlow_AI_Engine.ipynb` · `Aviation_Strategy_PostgreSQL_Optimization.ipynb` · `CineIntelligence_NoSQL_DataEngineering.ipynb` · `Clustering_Models.ipynb` · `KDDCup.ipynb` · `LLM_Mastery_Hands_on_Code.ipynb` · `LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb` · `Logistic_Regression_PySpark.ipynb` · `Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb` · `NYC_Airbnb_Market_Analysis (1).ipynb` · `Naive_Bayes_PySpark.ipynb` · `Parkinsons_Progression_ML.ipynb` · `Pathfinding.ipynb` · `PyTorch_medical_AI_xray_diagnosis.ipynb` · `Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb` · `financial_fraud_aml_detection_system.ipynb`

## Additional analytics work

The separate [Data Analyst Bootcamp Portfolio](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp) contains executed Python, SQL, data-cleaning, regression, reporting and business-analysis case studies.

## Licence and external assets

Original repository code and documentation are MIT-licensed. Third-party datasets, pretrained model assets and external resources retain their own licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
