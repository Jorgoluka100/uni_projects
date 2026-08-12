# AI & Data Science Portfolio

## Jorgo Luka — MSc Artificial Intelligence & Data Science (Distinction)

I build end-to-end data and AI systems around a real decision: source and audit the data, prevent leakage, establish a baseline, evaluate on genuinely held-out evidence, retain reproducible artifacts, and state limitations instead of hiding them.

This repository keeps the original work visible while distinguishing **verified results** from **advanced work still awaiting project-specific evidence**.

## Featured verified projects

| Project | Decision / problem | Verified evidence | Stack | Open |
|---|---|---|---|---|
| **VisionForge — Trustworthy Visual Inspection** | Classify bean-leaf disease while escalating uncertain images instead of forcing every decision. | Clean full notebook execution on the public Makerere Beans split. Test accuracy **85.9%**, macro-F1 **85.8%** (95% bootstrap CI **79.2–91.3%**); selective accuracy **90.4%** at **89.1%** coverage. Independent checkpoint recomputation passed; TorchScript and ONNX matched PyTorch. | PyTorch, EfficientNet, calibration, Grad-CAM, ONNX, TorchScript, Gradio | [Notebook](12_VisionForge_PyTorch_Visual_Inspection.ipynb) · [Evidence](verified/visionforge/verification_metrics.json) · [Verifier](extensions/visionforge_verify_v2.py) |
| **AeroFlow — 2026 Flight Delay Risk** | Prioritise flights at risk of arriving ≥15 minutes late using information available at schedule time. | Official BTS 2026 data: **360k** Jan–Mar train, **120k** Apr validation, **180k** untouched May test. PR-AUC **0.291** vs **0.215** prevalence, ROC-AUC **0.618**; top-10% risk bucket had **34.1%** delays vs **21.5%** population (**1.58× lift**). An earlier delay-minute regression that failed the MAE baseline is deliberately retained as a negative result; the decision was then reframed. | Python, CatBoost, temporal validation, operational thresholding | [Original notebook](AeroFlow_AI_Engine.ipynb) · [Current v3](extensions/aeroflow_delay_risk_v3.py) · [Evidence](verified/aeroflow_delay_risk/verification.json) |
| **UK House Price Analysis & Prediction** | Produce a forward-looking residential price range while being explicit about uncertainty and missing property characteristics. | **995,059** modelling transactions; **216,564-sale** untouched 2026 test; CatBoost MAE **£81,805**, R² **0.604**; 90% interval coverage **91.6%**. | Python, Pandas, DuckDB SQL, CatBoost | [Notebook](01_UK_House_Price_Analysis_and_Prediction.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/01_UK_House_Price_Analysis_and_Prediction.ipynb) |
| **NYC Airbnb Market Analysis — 2026 Refresh** | Estimate snapshot listing prices while testing transfer to neighbourhoods unseen during training. | Inside Airbnb **14 Jun 2026** snapshot; **30,259** raw listings. Untouched **3,787-row / 45-neighbourhood** test: MAE **$68.97** vs median baseline **$121.90** (**43.4% improvement**), R² **0.483**. Listing price is explicitly not presented as realised revenue. | Python, Pandas, scikit-learn, grouped validation | [Original notebook](NYC_Airbnb_Market_Analysis%20(1).ipynb) · [Current v2](extensions/airbnb_nyc_2026_v2.py) · [Evidence](verified/airbnb_nyc_2026/verification.json) |
| **Movie Recommender — Temporal Ranking v2** | Test whether a recommender can rank each user's next positive item rather than merely minimise rating error. | **609** held-out users; evaluated-user histories truncated strictly before the target, removing **1,756** later interactions that would otherwise leak. Latent SVD Recall@10 **0.501** vs popularity **0.463**; NDCG@10 **0.336** vs **0.293**. | Python, SciPy, scikit-learn, recommendation evaluation | [Original notebook](Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb) · [Current v2](extensions/recommender_v2.py) · [Evidence](verified/recommender/verification.json) |
| **SQL Sales & Customer Analysis** | Build trustworthy marketplace KPIs from nine relational tables without multiplying money across incompatible grains. | **98,199** commercial orders, **94,983** customers and **R$13.49M** merchandise value reconciled; source hashes, key checks, semantic-layer tests and artifact tests passed. | SQL, DuckDB, Python, Pandas, Parquet | [Notebook](02_SQL_Sales_and_Customer_Analysis.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/02_SQL_Sales_and_Customer_Analysis.ipynb) |
| **Customer Churn Prediction** | Build a decision-ready churn screening system while controlling duplicate-profile leakage, calibration and review capacity. | Holdout PR-AUC **0.955** (95% bootstrap **0.927–0.980**) and ROC-AUC **0.990**; **94.9%** recall at **73.2%** precision while flagging **20.2%** of customers. | Python, scikit-learn, calibration, bootstrap diagnostics | [Notebook](03_Customer_Churn_Prediction.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/03_Customer_Churn_Prediction.ipynb) |
| **Clickstream Analysis with PySpark** | Measure engagement drop-off at Spark scale and identify higher-intent shopping sessions without post-purchase leakage. | **165,474** real click events across **24,026** sessions; separate **12,330-session** conversion dataset; test PR-AUC **0.351** vs **0.155** prevalence and ROC-AUC **0.763**. | PySpark, Spark SQL, Spark ML | [Notebook](06_Clickstream_Analysis_with_PySpark.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/06_Clickstream_Analysis_with_PySpark.ipynb) |
| **Energy Demand Forecasting with TensorFlow** | Forecast a chronological electricity-demand window and test whether the neural model beats honest seasonal baselines. | LSTM test MAE **43.51** vs **53.18** for the 7-day seasonal baseline — **18.2% improvement** — with prediction intervals and saved-model verification. | TensorFlow, Keras, LSTM, time-series validation | [Notebook](05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) |
| **London Air Quality Analysis with R** | Turn messy government files into an auditable monitoring panel and test past-only features for hourly NO₂ estimates. | Official **2025–2026** data; **143,102** observed station-hours; untouched 2026 MAE **10.48**, R² **0.056**; 90% interval coverage **88.2%**. | R, data.table, ggplot2, mgcv | [Notebook](07_London_Air_Quality_Analysis_with_R.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/07_London_Air_Quality_Analysis_with_R.ipynb) |
| **ConsultAI — AI Opportunity Prioritisation Engine** | Select an AI initiative portfolio when value, uncertainty, readiness, risk and budget all matter. | Clean rerun + independent deterministic verifier passed: 6 educational use cases, **10,000 Monte Carlo trials/case**, **17** budget points and **5** stress scenarios. Reproduced 3-project portfolio: **£490k** illustrative spend, **£501,652** expected NPV. Inputs are explicitly synthetic, not real-company forecasts. | Python, NumPy, Monte Carlo, exhaustive optimisation, Gradio | [Notebook](01_ConsultAI_AI_Opportunity_Engine.ipynb) · [Evidence](verified/consultai/consultai_verification.json) · [Verifier](extensions/consultai_verify_v2.py) |
| **Image Classification with CNNs & Transfer Learning** | Classify pet breeds and route uncertain predictions to human review. | **7,349** real images / 37 breeds; test accuracy **58.5%**, macro-F1 **56.3%**, top-3 accuracy **83.1%**; accepted-case accuracy **80.9%** at **49.1%** coverage. | PyTorch, torchvision, MobileNetV3, Grad-CAM | [Notebook](04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) |

## Verified methodology & engineering evidence

These projects have clean retained evidence, but their dataset/scope means they should be read as **engineering or methodology demonstrations**, not current real-world performance claims.

| Project | What is now proven | Evidence |
|---|---|---|
| **Aviation PostgreSQL Optimisation** | Genuine PostgreSQL 17 execution on a deterministic **1,000,000-row** fact table; two `EXPLAIN (ANALYZE, BUFFERS)` plans retained, targeted index observed after creation, semantic financial reconciliation passed. | [Original notebook](Aviation_Strategy_PostgreSQL_Optimization.ipynb) · [SQL v2](extensions/aviation_postgres_v2.sql) · [Evidence](verified/aviation_postgres/verification.json) |
| **PySpark Logistic Regression + Naive Bayes** | **145,585** deduplicated historical KDD rows; train-only Spark pipeline; validation selection and untouched test; serialized pipeline reload matched. Logistic Regression test PR-AUC **0.9966** / ROC-AUC **0.9972**; explicitly historical, not modern cybersecurity evidence. | [Logistic notebook](Logistic_Regression_PySpark.ipynb) · [Naive Bayes notebook](Naive_Bayes_PySpark.ipynb) · [Spark v2](extensions/spark_kdd_classifiers_v2.py) · [Evidence](verified/spark_kdd/verification.json) |
| **Financial Fraud / AML Decision Pipeline** | Chronological synthetic transaction stream, past-only behavioural features, validation-only threshold, review-capacity/cost metrics and model reload all verified. Test PR-AUC **0.102** vs **0.0375** prevalence; explicitly synthetic. | [Original notebook](financial_fraud_aml_detection_system.ipynb) · [v2](extensions/fraud_aml_v2.py) · [Evidence](verified/fraud_aml/verification.json) |
| **Strategic Telecom Churn + Predictive SQL** | Train-only preprocessing, validation-selected model/threshold, untouched test and SQL↔Pandas reconciliation verified. Synthetic test PR-AUC **0.314**, ROC-AUC **0.715**; 14.8% review rate. | [Original notebook](Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb) · [v2](extensions/telecom_churn_decision_v2.py) · [Evidence](verified/telecom_churn/verification.json) |
| **Parkinson's Progression — Grouped Evaluation** | Corrected the original row-level leakage by holding out complete subjects and bootstrapping by subject. The honest result is negative: RF MAE **10.70** vs median baseline **8.13**, R² **-1.09** on 9 unseen subjects. This is retained because it shows the model does not generalise under the stricter design. | [Original notebook](Parkinsons_Progression_ML.ipynb) · [Grouped v2](extensions/parkinsons_grouped_v2.py) · [Evidence](verified/parkinsons_grouped/verification.json) |
| **Clustering Stability Benchmark** | `k` selected from silhouette, Davies–Bouldin, Calinski–Harabasz and resampling stability rather than one score. On the reproducible Wine methodology demo, **k=3** with mean ARI stability **0.974**. | [Original notebook](Clustering_Models.ipynb) · [Stability v2](extensions/clustering_stability_v2.py) · [Evidence](verified/clustering/verification.json) |
| **Pathfinding Benchmark** | BFS, Dijkstra and A* tested on 5 seeded solvable grids; all returned valid optimal paths. A* median expanded nodes **483** vs **1,145** for BFS/Dijkstra; timing retained but explicitly environment-dependent. | [Original notebook](Pathfinding.ipynb) · [Benchmark v2](extensions/pathfinding_benchmark_v2.py) · [Evidence](verified/pathfinding/verification.json) |

## Advanced project laboratory

The remaining notebooks stay visible because their code is useful, but they do **not** receive promoted performance claims until their project-specific evidence contract is satisfied.

| Project | Current completion state | Open |
|---|---|---|
| **Advanced Multi-Modal Health Analytics Suite** | Safety/evidence gate exists; still requires a clean patient/group-safe project run and defensible provenance. | [Notebook](Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb) · [Healthcare gate](extensions/healthcare_evidence_gate.py) |
| **CineIntelligence NoSQL Data Engineering** | Defensive parsing, quarantine, explicit document schema and indexed-query v2 are ready; fresh source/licence evidence is still required. | [Notebook](CineIntelligence_NoSQL_DataEngineering.ipynb) · [v2](extensions/cine_nosql_v2.py) |
| **KDD Cup Analysis** | Separate sklearn historical benchmark v2 exists; the Spark classifier path above is already verified, but this original KDD analysis remains an unpromoted historical experiment. | [Notebook](KDDCup.ipynb) · [v2](extensions/kdd_intrusion_v2.py) |
| **LLM Mastery — Core** | Transformer implementation preserved; frozen evaluation harness exists, but current checkpoint generations/loss evidence still require a fresh model run. | [Notebook](LLM_Mastery_Hands_on_Code.ipynb) · [Evaluation harness](extensions/llm_eval_v2.py) |
| **LLM Mastery — Alignment** | Alignment implementation preserved; base-vs-aligned frozen-prompt evaluation is ready, but fresh model generations are still required. | [Notebook](LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb) · [Evaluation harness](extensions/llm_eval_v2.py) |
| **PyTorch Medical AI — X-ray Diagnosis** | Safety gate blocks promotion without patient/group independence, provenance, uncertainty/abstention and explicit non-clinical guardrails. | [Notebook](PyTorch_medical_AI_xray_diagnosis.ipynb) · [Healthcare gate](extensions/healthcare_evidence_gate.py) |

## Evidence standard

A promoted result must have a clean current run and retained evidence. The repository validator in [`scripts/validate_portfolio.py`](scripts/validate_portfolio.py) checks executed flagship notebooks separately from advanced notebooks and also checks retained verification files.

The rules are:

1. Define the decision and failure cost.
2. Document source, licence/usage constraints, date coverage and freshness.
3. Audit schema, missingness, duplicates, ranges and join grain.
4. Prevent target, temporal and group leakage.
5. Establish a simple baseline before trusting complexity.
6. Reserve genuinely held-out evidence where the task supports it.
7. Add uncertainty, calibration, slices, ranking/decision metrics or human-review policy where appropriate.
8. Save/reload or independently reconstruct important artifacts.
9. Retain negative results when they change the modelling decision.
10. State what the result does **not** prove.

## Data-freshness / scope notes

- **AeroFlow v3** uses official 2026 BTS data with May 2026 untouched test evidence; its earlier regression failure is retained rather than rewritten away.
- **NYC Airbnb v2** uses the 14 June 2026 Inside Airbnb snapshot; listing prices/availability are not transactions or confirmed bookings.
- **Customer Churn** uses the historical UCI Iranian Churn dataset.
- **Energy Demand** uses official Open Power System Data from 2006–2017.
- **Spark KDD** uses KDD Cup 1999 strictly as a historical distributed-ML benchmark.
- **ConsultAI, Fraud/AML and Telecom decision v2** use synthetic data and are labelled accordingly.
- **Parkinson's** is historical, educational and non-clinical; stricter grouped evaluation exposed poor generalisation and that result is retained.

## Skills evidenced

Python · SQL · R · PySpark · Pandas · NumPy · scikit-learn · CatBoost · TensorFlow/Keras · PyTorch · PostgreSQL · DuckDB · Spark SQL · computer vision · recommendation systems · time-series modelling · classification · regression · clustering · calibration · uncertainty · leakage-safe validation · data cleaning/preprocessing · model persistence · CI/evidence gates · decision engineering

## Additional analytics work

The separate [Data Analyst Bootcamp Portfolio](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp) contains executed Python, SQL, data-cleaning, regression, reporting and business-analysis case studies.

## Portfolio status

The main project-content phase is now **evidence-first rather than notebook-count-first**. Strong original code has been preserved; the highest-value projects have been rerun and verified; negative results are retained; and the remaining laboratory items have explicit evidence gates instead of vague "finish later" notes.

For opportunities in data science, machine learning, applied AI or analytics, contact me through my [GitHub profile](https://github.com/Jorgoluka100).
