# Jorgo Luka — Data & AI Portfolio

**MSc Artificial Intelligence & Data Science (Distinction), University of East London**  
Python · SQL · machine learning · data engineering · applied AI

This portfolio combines my **original university projects** with newer **production-style Data & AI work**. The university notebooks show the academic foundation; the newer packages show how I now approach the same problems with stronger validation, testing, reproducibility and engineering discipline.

[![Portfolio checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)
[![Production project checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml)
[![Reliable pipeline checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/reliable-event-pipeline-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/reliable-event-pipeline-ci.yml)

## Start with the role you are hiring for

| Role | Best evidence | What it demonstrates |
| --- | --- | --- |
| **Data Scientist** | [Flight Delay Risk](projects/flight_delay_risk/) · [Customer Churn](projects/customer_churn_prediction/) · [UK House Prices](projects/uk_house_price_prediction/) | supervised ML, temporal/grouped validation, baselines, calibration, thresholding and uncertainty |
| **Data Analyst / Technical Analyst** | [Retail KPI Dashboard](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp/tree/main/dashboard) · [E-commerce SQL Analytics](projects/ecommerce_sql_analytics/) · [ExperimentLab](projects/experiment_lab/) | dashboards, SQL, KPI logic, data cleaning, experimentation and business interpretation |
| **Data Engineer / Analytics Engineer** | [Reliable Event Pipeline](projects/reliable_event_pipeline/) · [PySpark Clickstream](projects/pyspark_clickstream_analytics/) | ingestion, schema validation, deduplication, idempotency, data-quality checks and scalable transformations |
| **ML / AI Engineer** | [Grounded RAG](projects/grounded_rag/) · [Image Classification](projects/image_classification_confidence/) · [ModelWatch](projects/model_watch/) | retrieval/evaluation, PyTorch, APIs, model export, calibration, drift monitoring and deployment-minded engineering |

## Flagship production-style evidence

| Project | Proof point |
| --- | --- |
| **[Reliable Event Pipeline](projects/reliable_event_pipeline/)** | 10 controlled input rows → **7 clean warehouse rows**, with duplicate/null checks, late-event handling, reject logging, batch audits, SQL quality checks and tests. |
| **[Flight Delay Risk](projects/flight_delay_risk/)** | Official 2026 US flight data; untouched **180,000-flight** May test set; PR-AUC **0.291** vs **0.215** prevalence; highest-risk decile at **1.58×** normal delay rate. |
| **[E-commerce SQL Analytics](projects/ecommerce_sql_analytics/)** | **98,199** commercial orders, **94,983** customers and **R$13.49M** merchandise value after cross-grain reconciliation; SQL marts, windows, cohorts and join-safety tests. |
| **[Customer Churn](projects/customer_churn_prediction/)** | Protected holdout, grouped validation, out-of-fold calibration and cost-aware thresholding; PR-AUC **0.955**. |
| **[Grounded RAG](projects/grounded_rag/)** | Hybrid retrieval, evidence thresholds, source attribution, weak-match abstention, prompt-injection checks, allow-listed tool routing, FastAPI and Docker. |
| **[Image Classification](projects/image_classification_confidence/)** | EfficientNet-B0 with bootstrap uncertainty, selective prediction, Grad-CAM and verified TorchScript/ONNX parity; **85.9%** test accuracy. |
| **[ExperimentLab](projects/experiment_lab/)** | Treatment effects, confidence intervals, CUPED, guardrails and power; CUPED reduced retained-run variance by **50.7%**. |
| **[PySpark Clickstream](projects/pyspark_clickstream_analytics/)** | **165,474** real events, Spark transformations, explicit data-quality logic and a separate one-million-row load test. |

## MSc university projects

The original executed university notebooks are kept in the repository rather than hidden or replaced. Where I later rebuilt the same problem to a higher engineering standard, both versions are linked.

| University project | Original work | Follow-on project |
| --- | --- | --- |
| **UK House Price Analysis & Prediction** | [`01_UK_House_Price_Analysis_and_Prediction.ipynb`](01_UK_House_Price_Analysis_and_Prediction.ipynb) | [Production-style house price prediction](projects/uk_house_price_prediction/) |
| **SQL Sales & Customer Analysis** | [`02_SQL_Sales_and_Customer_Analysis.ipynb`](02_SQL_Sales_and_Customer_Analysis.ipynb) | [E-commerce SQL Analytics](projects/ecommerce_sql_analytics/) |
| **Customer Churn Prediction** | [`03_Customer_Churn_Prediction.ipynb`](03_Customer_Churn_Prediction.ipynb) | [Production-style churn modelling](projects/customer_churn_prediction/) |
| **Image Classification with CNNs & Transfer Learning** | [`04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb`](04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) | [Confidence-aware image classification](projects/image_classification_confidence/) |
| **Energy Demand Forecasting with TensorFlow** | [`05_Energy_Demand_Forecasting_with_TensorFlow.ipynb`](05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) | [Production-style energy forecasting](projects/energy_demand_forecasting/) |
| **Clickstream Analysis with PySpark** | [`06_Clickstream_Analysis_with_PySpark.ipynb`](06_Clickstream_Analysis_with_PySpark.ipynb) | [PySpark Clickstream Analytics](projects/pyspark_clickstream_analytics/) |
| **London Air Quality Analysis with R** | [`07_London_Air_Quality_Analysis_with_R.ipynb`](07_London_Air_Quality_Analysis_with_R.ipynb) | Original executed notebook retained as R evidence |

**[View the university-project index →](docs/UNIVERSITY_PROJECTS.md)**

## Engineering standard

Across the strongest work I make these visible rather than leaving them implicit:

- data source, schema and quality checks
- validation or holdout design that matches intended use
- leakage, duplicate and join-grain controls where relevant
- meaningful baselines before complex models
- threshold, calibration, uncertainty or stability checks when useful
- machine-readable retained results
- tests and self-tests for packaged projects
- GitHub Actions for production-style work
- APIs / Docker where deployment adds value
- limitations stated explicitly

## Repository map

```text
projects/                 production-style projects I would discuss in interviews
verified/                 retained machine-readable evidence from earlier project versions
extensions/               hardened rerun / verification code for older notebooks
docs/                     university index, complete catalogue and supporting documentation
portfolio_manifest.json   machine-readable inventory of university + production projects
scripts/                   integrity validators used by GitHub Actions
*.ipynb                   executed university, learning and historical notebooks
```

The portfolio manifest is checked automatically so listed projects cannot silently disappear or point at broken local paths. Use the **[complete project catalogue](docs/PROJECT_CATALOG.md)** for the full inventory.

## Main tools

**Data:** Python, SQL, Pandas, NumPy, PostgreSQL, DuckDB, PySpark  
**ML:** scikit-learn, CatBoost, PyTorch, TensorFlow/Keras  
**Applied AI:** retrieval, NLP, LLM/RAG evaluation, FastAPI  
**Engineering:** Docker, Git, GitHub Actions, testing, model/data validation

## Supporting analyst work

Smaller analyst exercises and the interactive retail dashboard are kept separately in the **[Data Analyst Bootcamp repository](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp)** so this repository stays focused while still providing direct analyst evidence.

## Licence

My own code and documentation are MIT-licensed. External datasets and pretrained models retain their original licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
