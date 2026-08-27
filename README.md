# Jorgo Luka — Data & AI Portfolio

**MSc Artificial Intelligence & Data Science (Distinction), University of East London**  
Python · SQL · machine learning · data engineering · applied AI

This is an **evidence-backed portfolio**, not a notebook dump. The strongest projects make the data checks, validation choices, baselines, tests, retained results and limitations visible alongside the code.

[![Portfolio checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)
[![Production project checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml)
[![Reliable pipeline checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/reliable-event-pipeline-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/reliable-event-pipeline-ci.yml)

## Start with the role you are hiring for

| Role | Best evidence | What it demonstrates |
| --- | --- | --- |
| **Data Scientist** | [Flight Delay Risk](projects/flight_delay_risk/) · [Customer Churn](projects/customer_churn_prediction/) · [UK House Prices](projects/uk_house_price_prediction/) | supervised ML, temporal/grouped validation, baselines, calibration, thresholding and uncertainty |
| **Data Analyst / Technical Analyst** | [E-commerce SQL Analytics](projects/ecommerce_sql_analytics/) · [ExperimentLab](projects/experiment_lab/) · [Retail Segmentation](projects/retail_customer_segmentation/) | SQL, KPI logic, data cleaning, experimentation, segmentation and business interpretation |
| **Data Engineer / Analytics Engineer** | [Reliable Event Pipeline](projects/reliable_event_pipeline/) · [PySpark Clickstream](projects/pyspark_clickstream_analytics/) | ingestion, schema validation, deduplication, idempotency, data-quality checks and scalable transformations |
| **ML / AI Engineer** | [Grounded RAG](projects/grounded_rag/) · [Image Classification](projects/image_classification_confidence/) · [ModelWatch](projects/model_watch/) | retrieval/evaluation, PyTorch, APIs, model export, calibration, drift monitoring and deployment-minded engineering |

## Flagship evidence

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

## Engineering standard

Across the flagship work I try to make these visible rather than leaving them implicit:

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
projects/     production-style projects I would discuss in interviews
verified/     retained machine-readable evidence from earlier project versions
extensions/   hardened rerun / verification code for older notebooks
docs/         complete catalogue and supporting documentation
*.ipynb       executed university, learning and historical notebooks
```

For everything else, use the **[complete project catalogue](docs/PROJECT_CATALOG.md)**. Older notebooks remain available as learning history, but the recruiter path above intentionally leads with the strongest verified work.

## Main tools

**Data:** Python, SQL, Pandas, NumPy, PostgreSQL, DuckDB, PySpark  
**ML:** scikit-learn, CatBoost, PyTorch, TensorFlow/Keras  
**Applied AI:** retrieval, NLP, LLM/RAG evaluation, FastAPI  
**Engineering:** Docker, Git, GitHub Actions, testing, model/data validation

## Supporting analyst work

Smaller analyst exercises from a 220-hour bootcamp are kept separately in the **[Data Analyst Bootcamp repository](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp)** so this repository stays focused.

## Licence

My own code and documentation are MIT-licensed. External datasets and pretrained models retain their original licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
