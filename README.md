# Jorgo Luka — AI & Data Science Portfolio

**MSc Artificial Intelligence & Data Science (Distinction)**  
**Target roles:** Junior / Graduate Data Scientist · Machine Learning Engineer · AI Engineer · Data Analyst

[![Portfolio integrity](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)
[![Production project evidence](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml)

I build data and AI systems that can be **checked rather than merely presented**: data quality first, leakage-safe validation, explicit baselines, held-out evidence, uncertainty or abstention where appropriate, reproducible artifacts, and limitations that stay visible when a model fails.

**Recruiter shortcuts:** [Hiring playbook](docs/HIRING_PLAYBOOK.md) · [Production-style projects](projects/) · [Verification evidence](verified/) · [Full project catalog](docs/PROJECT_CATALOG.md) · [Data Analyst bootcamp](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp)

## 60-second view — strongest evidence

| Project | What it demonstrates | Verified evidence | Stack |
|---|---|---|---|
| **GroundedRAG v2 — Retrieval, Vector Search, Tool Routing & Safety** | Applied-AI engineering: sparse+dense retrieval, deterministic vector indexing, grounded citations, abstention, allow-listed read-only tool routing, prompt-injection blocking, API and Docker packaging. | Frozen synthetic fixture: Recall@3 / MRR@3 / NDCG@3 **1.000**; citation-or-correct-abstention **1.000**; tool-route **1.000**; tool-result **1.000**; injection block rate **1.000**; tool execution on attack fixture **0.000**. These are deterministic contract tests, **not production generalisation claims**. [Code](projects/grounded_rag/run.py) · [Evidence](verified/grounded_rag/verification.json) | Python, BM25, TF-IDF, LSA embeddings, dense vector index, FastAPI, Docker, RAG/tool evaluation |
| **VisionForge — Trustworthy Visual Inspection** | Computer vision plus calibration, explainability and deployment evidence. | Makerere Beans test accuracy **85.9%**, macro-F1 **85.8%**; selective accuracy **90.4%** at **89.1%** coverage; TorchScript and ONNX parity passed. [Notebook](12_VisionForge_PyTorch_Visual_Inspection.ipynb) · [Evidence](verified/visionforge/verification_metrics.json) | PyTorch, EfficientNet, Grad-CAM, calibration, ONNX |
| **AeroFlow — 2026 Flight Delay Risk** | Current-data temporal modelling and honest problem reframing after a failed regression baseline. | Official BTS 2026: **360k** train / **120k** validation / **180k** untouched May test; PR-AUC **0.291** vs **0.215** prevalence; top-risk decile **1.58× lift**. [Code](extensions/aeroflow_delay_risk_v3.py) · [Evidence](verified/aeroflow_delay_risk/verification.json) | Python, CatBoost, temporal validation, ranking |
| **ExperimentLab** | Product experimentation and decision science rather than prediction alone. | **20,000** deterministic simulated observations; CUPED estimate **2.773** for known +2.5 effect, 95% CI **2.435–3.112**; **50.7% variance reduction**; guardrail and power checks passed. [Code](projects/experiment_lab/run.py) · [Evidence](verified/experiment_lab/verification.json) | Python, CUPED, bootstrap, power/MDE |
| **ModelWatch** | MLOps monitoring with explicit operational policy. | Stable batch max PSI **0.003 → green**; feature shift **0.264 → red**; concept shift **0.466 → red**; performance/calibration metrics and saved-model reload parity retained. [Code](projects/model_watch/run.py) · [Evidence](verified/model_watch/verification.json) | Python, PSI/KS, ROC/PR, Brier, ECE, MLOps |
| **SQL Sales & Customer Analysis** | Relational grain control, KPI reconciliation and commercial analytics. | **98,199** commercial orders, **94,983** customers, **R$13.49M** merchandise value; source/key/semantic-layer and artifact checks passed. [Notebook](02_SQL_Sales_and_Customer_Analysis.ipynb) | SQL, DuckDB, Pandas, Parquet |

## Capabilities by role

### Applied AI / ML Engineering
- GroundedRAG v2: sparse+dense retrieval, deterministic LSA embeddings/vector indexing, source attribution, abstention, constrained tool routing, prompt-injection blocking, FastAPI and Docker.
- VisionForge: PyTorch transfer learning, calibration, Grad-CAM, TorchScript and ONNX parity.
- ModelWatch: drift, calibration, performance monitoring, alert policy and model reload checks.
- LLM evaluation harness retained separately; model-quality claims remain unpromoted until fresh checkpoint evidence exists.

### Data Science
- AeroFlow: official 2026 temporal data, classification/ranking, calibration and business targeting.
- Customer churn: PR-AUC **0.955** (95% bootstrap **0.927–0.980**), ROC-AUC **0.990**.
- NYC Airbnb 2026: unseen-neighbourhood test MAE **$68.97** vs **$121.90** baseline, **43.4% improvement**.
- Movie recommender: temporal leakage fix; Recall@10 **0.501** vs **0.463** popularity baseline.
- TensorFlow energy forecasting: MAE **43.51** vs **53.18** seasonal baseline, **18.2% improvement**.

### Analytics / Data
- SQL sales: trustworthy commercial KPI modelling and reconciliation.
- PySpark clickstream: **165,474** events / **24,026** sessions; PR-AUC **0.351** vs **0.155** prevalence.
- PostgreSQL optimisation: deterministic **1,000,000-row** workload with before/after `EXPLAIN (ANALYZE, BUFFERS)` evidence.
- R air-quality analysis: official 2025–2026 data with limitations retained.
- Separate [Data Analyst Bootcamp Portfolio](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp) demonstrates cleaning, EDA, SQL, regression, reporting and business communication foundations.

## Production-style project layer

[`projects/`](projects/) contains the smaller package-like projects added to close specific hiring gaps:

1. **GroundedRAG v2** — applied-AI retrieval, dense vector indexing, safe tool routing and evaluation.
2. **CareerLens AI** — NLP / information retrieval and inspectable skill matching.
3. **ExperimentLab** — A/B testing, CUPED, uncertainty, guardrails and power.
4. **ModelWatch** — model drift, performance, calibration and retraining policy.

Each project has a runnable implementation and machine-readable evidence. [`new-projects-ci.yml`](.github/workflows/new-projects-ci.yml) reruns self-tests and clean evidence generation before retained metrics are trusted.

## Evidence standard

A project is promoted only when it can answer these questions:

1. **What decision is being supported?**
2. **Where did the data come from and what are its usage limits?**
3. **What data-quality checks were performed?**
4. **How was target, temporal or group leakage prevented?**
5. **What simple baseline had to be beaten?**
6. **What evidence was genuinely held out?**
7. **How are uncertainty, calibration, ranking, abstention or human review handled?**
8. **Can the important artifact be reconstructed or reloaded?**
9. **What failed, and was the failure preserved?**
10. **Can every CV metric be defended in an interview?**

Examples of negative evidence deliberately retained: the first AeroFlow regression formulation failed its seasonal-style baseline and was reframed; grouped Parkinson's validation underperformed a median baseline; temporal leakage was found and removed from the recommender evaluation.

## Technology evidenced

**Python · SQL · R · PySpark · Pandas · NumPy · SciPy · scikit-learn · CatBoost · TensorFlow/Keras · PyTorch · PostgreSQL · DuckDB · Spark SQL · FastAPI · Docker · NLP/information retrieval · BM25 · dense vector retrieval · LSA embeddings · RAG evaluation · safe tool routing · computer vision · experimentation/CUPED · MLOps monitoring · recommendation systems · time series · calibration · uncertainty · model persistence · CI/evidence gates · data cleaning/preprocessing**

## Repository map

- [`projects/`](projects/) — production-style Python projects and engineering demos.
- [`verified/`](verified/) — retained machine-readable evidence.
- [`extensions/`](extensions/) — hardened extensions and verification runners for notebook projects.
- [`docs/HIRING_PLAYBOOK.md`](docs/HIRING_PLAYBOOK.md) — recruiter/interview navigation and defensible project stories.
- [`docs/PROJECT_CATALOG.md`](docs/PROJECT_CATALOG.md) — complete project inventory used by the integrity gate.
- Root `.ipynb` files — original and advanced notebook portfolio, preserved rather than hidden.

## Deliberately not promoted

Healthcare diagnostic notebooks remain laboratory work until patient/group-safe provenance and evaluation are strong enough to justify promotion. LLM training/alignment notebooks remain inspectable, but the repository does not claim current model quality without fresh checkpoint-based generation evidence.

<details>
<summary><strong>Complete notebook inventory (integrity / audit)</strong></summary>

Verified flagships: `01_UK_House_Price_Analysis_and_Prediction.ipynb` · `02_SQL_Sales_and_Customer_Analysis.ipynb` · `03_Customer_Churn_Prediction.ipynb` · `04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb` · `05_Energy_Demand_Forecasting_with_TensorFlow.ipynb` · `06_Clickstream_Analysis_with_PySpark.ipynb` · `07_London_Air_Quality_Analysis_with_R.ipynb` · `01_ConsultAI_AI_Opportunity_Engine.ipynb` · `12_VisionForge_PyTorch_Visual_Inspection.ipynb`

Advanced / laboratory: `Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb` · `AeroFlow_AI_Engine.ipynb` · `Aviation_Strategy_PostgreSQL_Optimization.ipynb` · `CineIntelligence_NoSQL_DataEngineering.ipynb` · `Clustering_Models.ipynb` · `KDDCup.ipynb` · `LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb` · `LLM_Mastery_Hands_on_Code.ipynb` · `Logistic_Regression_PySpark.ipynb` · `Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb` · `NYC_Airbnb_Market_Analysis (1).ipynb` · `Naive_Bayes_PySpark.ipynb` · `Parkinsons_Progression_ML.ipynb` · `Pathfinding.ipynb` · `PyTorch_medical_AI_xray_diagnosis.ipynb` · `Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb` · `financial_fraud_aml_detection_system.ipynb`

</details>

## Licence and external assets

Original repository code and documentation are MIT-licensed. Third-party datasets, pretrained model assets and external resources retain their own licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
