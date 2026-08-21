# Jorgo Luka — Data & AI Portfolio

**MSc Artificial Intelligence & Data Science (Distinction)**  
Python · SQL · machine learning · applied AI · analytics

I use this repository for the projects I would be comfortable opening in an interview and explaining from the data checks through to the final result. My strongest work is listed first; older coursework and experiments are still kept in the repository, but I do not treat every notebook as portfolio-grade evidence.

[![Portfolio checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)
[![Python project checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml)

## Selected work

### Flight delay prediction and risk analysis

Used official 2026 US flight data to predict flights likely to arrive at least 15 minutes late. An earlier regression version did not beat its baseline, so I reframed the problem as classification and ranking rather than keeping a weak result. I then rebuilt it as a standalone, leakage-safe project with data checks, temporal validation, calibration diagnostics, capacity analysis, tests and reproducible evidence.

- Untouched May test set: **180,000 flights**
- PR-AUC: **0.291** vs **0.215** delay rate
- Highest-risk 10% of flights: **1.58x** the normal delay rate
- Tools: Python, Pandas, CatBoost, scikit-learn, temporal validation

[Project](projects/flight_delay_risk/) · [Model card](projects/flight_delay_risk/MODEL_CARD.md) · [Saved evidence](projects/flight_delay_risk/results/verified_test_metrics.json)

### E-commerce sales and customer analysis

Cleaned and checked an e-commerce dataset before building SQL reporting around orders, customers and merchandise value. I focused on key integrity, join grain and reconciling the headline totals before using them in analysis.

- **98,199** commercial orders
- **94,983** customers
- **R$13.49M** merchandise value after reconciliation checks
- Tools: SQL, DuckDB, Pandas

[Notebook](02_SQL_Sales_and_Customer_Analysis.ipynb)

### Image classification with confidence checks

Fine-tuned EfficientNet-B0 for bean-leaf image classification, then added Grad-CAM, confidence-based rejection and model export checks rather than stopping at test accuracy.

- Test accuracy: **85.9%**
- Macro-F1: **85.8%**
- Accuracy on accepted predictions at the chosen threshold: **90.4%**
- Tools: PyTorch, EfficientNet, Grad-CAM, ONNX

[Notebook](12_VisionForge_PyTorch_Visual_Inspection.ipynb) · [Saved evidence](verified/visionforge/verification_metrics.json)

### Retrieval-augmented support assistant

Built a small local retrieval application over a frozen set of policy and incident documents. It returns sources, abstains on weak matches, exposes a read-only ticket analytics route through FastAPI and includes prompt-injection test cases.

The perfect scores in the small frozen fixture are software/evaluation checks, not a claim that the system would achieve 100% performance on real company data.

- Tools: Python, BM25, TF-IDF/LSA, FastAPI, Docker

[Project](projects/grounded_rag/) · [Saved evidence](verified/grounded_rag/verification.json)

### A/B test analysis

Built a reproducible experiment-analysis project covering treatment-effect estimation, confidence intervals, CUPED, guardrails and power calculations. The dataset is simulated so the implementation can be checked against a known effect.

- **20,000** simulated observations
- CUPED reduced outcome variance by **50.7%** in the retained run
- Tools: Python, statistics, bootstrap, CUPED

[Project](projects/experiment_lab/) · [Saved evidence](verified/experiment_lab/verification.json)

### Model monitoring and drift detection

Built a monitoring demo that compares incoming batches with a reference set and checks feature drift, discrimination and calibration. The shifted batches are deliberate so I can verify that the alert rules move in the expected direction.

- Stable batch remains green
- Deliberate feature and concept shifts trigger red alerts
- Tools: Python, PSI, KS, ROC/PR, Brier score, calibration metrics

[Project](projects/model_watch/) · [Saved evidence](verified/model_watch/verification.json)

## Other checked work

- **UK house prices:** 995,059 modelling transactions with an untouched 2026 test set of 216,564 sales; CatBoost MAE **£81,805**, R² **0.604**.
- **Customer churn:** retained test PR-AUC **0.955**, ROC-AUC **0.990**.
- **NYC Airbnb 2026:** MAE **$68.97** vs **$121.90** median baseline on unseen neighbourhoods.
- **Movie recommendation:** after removing future-interaction leakage, Recall@10 **0.501** vs **0.463** popularity baseline.
- **Energy forecasting:** TensorFlow MAE **43.51** vs **53.18** seasonal baseline.
- **PySpark clickstream:** **165,474** events across **24,026** sessions; PR-AUC **0.351** vs **0.155** positive rate.
- **London air quality:** analysis using official 2025–2026 monitoring data in R.

For smaller data-analysis exercises, see my [Data Analyst Bootcamp repository](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp).

## How I approach a project

I normally start by checking data quality and deciding what a realistic train/test split should look like. I compare against a simple baseline, keep the final test evidence separate, and look at errors and limitations rather than only reporting the best metric. Where a project is packaged as a Python application, I also keep small self-tests or reproducible result files so the claims can be checked later.

## Repository guide

- [`projects/`](projects/) — production-style Python applications and engineering-focused portfolio projects.
- [`verified/`](verified/) — retained result files and verification outputs.
- [`extensions/`](extensions/) — later verification or rerun code for some historical notebook projects.
- [`docs/PROJECT_CATALOG.md`](docs/PROJECT_CATALOG.md) — fuller catalogue of the repository.
- Root `.ipynb` files — a mixture of current portfolio work, university work and older learning notebooks.

Some older medical-imaging, LLM and coursework notebooks remain because they show what I was learning at the time. I do **not** use unverified results from those notebooks as production or clinical claims.

## Main tools

Python, SQL, Pandas, NumPy, scikit-learn, PyTorch, TensorFlow/Keras, CatBoost, DuckDB, PostgreSQL, PySpark, FastAPI, Docker and GitHub Actions.

## Licence

My own code and documentation are MIT-licensed. External datasets and pretrained models retain their original licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
