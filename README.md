# Jorgo Luka — Data, ML & Applied AI Portfolio

**MSc Artificial Intelligence & Data Science (Distinction), University of East London**  
Python · SQL · machine learning · data engineering · applied AI

I use this repository as an evidence-backed portfolio rather than a notebook dump. The projects I lead with are packaged so the data checks, validation choices, baselines, tests and retained results can be inspected alongside the code.

[![Portfolio checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/portfolio-integrity.yml)
[![Production project checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/new-projects-ci.yml)
[![Reliable pipeline checks](https://github.com/Jorgoluka100/uni_projects/actions/workflows/reliable-event-pipeline-ci.yml/badge.svg)](https://github.com/Jorgoluka100/uni_projects/actions/workflows/reliable-event-pipeline-ci.yml)

## Open these first

| If the role needs… | Best evidence | What is demonstrated |
| --- | --- | --- |
| **Data engineering / pipelines** | [Reliable event pipeline](projects/reliable_event_pipeline/) · [PySpark clickstream](projects/pyspark_clickstream_analytics/) | incremental ingestion, validation, deduplication, late data, SQL checks, Spark transformations |
| **Data science / predictive modelling** | [Flight delay risk](projects/flight_delay_risk/) · [Customer churn](projects/customer_churn_prediction/) | temporal/grouped validation, baselines, calibration, thresholding, uncertainty |
| **SQL / analytics** | [E-commerce analytics](projects/ecommerce_sql_analytics/) · [Retail segmentation](projects/retail_customer_segmentation/) | data modelling, join safety, reconciliation, cleaning, customer analysis |
| **Statistics / experimentation** | [ExperimentLab](projects/experiment_lab/) · [Energy forecasting](projects/energy_demand_forecasting/) | hypothesis testing, CUPED, power, time-series validation and forecasting |
| **AI / LLM applications** | [Grounded RAG](projects/grounded_rag/) · [CareerLens AI](projects/careerlens_ai/) | retrieval, evaluation, abstention, tool routing, NLP ranking, FastAPI and Docker |
| **Deep learning / model reliability** | [Image classification](projects/image_classification_confidence/) · [ModelWatch](projects/model_watch/) | PyTorch transfer learning, Grad-CAM, export checks, drift and calibration monitoring |

## Selected evidence

### Reliable event ingestion pipeline

A focused data-engineering project built around the failure modes that matter before modelling begins: schema errors, duplicate IDs, null business keys, late arrivals and re-sent events.

- 10 input rows across two controlled batches → **7 clean warehouse rows**
- **0** duplicate event IDs and **0** null customer IDs in the final table
- late-arriving data is retained and flagged rather than silently discarded
- idempotent primary-key loading, reject logging, batch audit metrics, SQL quality checks and unit tests

[Project](projects/reliable_event_pipeline/) · [Evidence](projects/reliable_event_pipeline/results/verified_run.json)

### Flight delay prediction and risk analysis

Official 2026 US flight data, chronological validation and leakage-safe schedule-time features.

- untouched May test set: **180,000 flights**
- PR-AUC **0.291** vs **0.215** delay prevalence
- highest-risk 10% of flights: **1.58×** the normal delay rate

[Project](projects/flight_delay_risk/) · [Model card](projects/flight_delay_risk/MODEL_CARD.md) · [Evidence](projects/flight_delay_risk/results/verified_test_metrics.json)

### E-commerce SQL analytics

The emphasis is trustworthy reporting grain rather than impressive-looking queries.

- **98,199** commercial orders and **94,983** customers
- **R$13.49M** merchandise value after cross-grain reconciliation
- SQL marts, window functions, cohorts, data-model documentation and join-safety tests

[Project](projects/ecommerce_sql_analytics/) · [SQL](projects/ecommerce_sql_analytics/sql/) · [Data model](projects/ecommerce_sql_analytics/DATA_MODEL.md)

### Retail cleaning and customer segmentation

A full audit-and-cleaning workflow before clustering.

- **541,909** raw transaction rows → **392,692** validated purchases
- explicit duplicate, cancellation, missing-ID, quantity and price rules
- KMeans selection backed by silhouette and stability checks rather than visual choice alone

[Project](projects/retail_customer_segmentation/) · [Project card](projects/retail_customer_segmentation/PROJECT_CARD.md)

### Grounded retrieval application

A local RAG-style support system where retrieval and tool behaviour are measurable.

- hybrid retrieval, evidence thresholds and source attribution
- weak-match abstention and prompt-injection checks
- allow-listed read-only tool routing
- FastAPI and Docker packaging with frozen evaluation fixtures

[Project](projects/grounded_rag/)

### ExperimentLab

A compact experimentation package covering treatment effects, confidence intervals, CUPED, guardrails and power.

- simulated data with a known effect so implementation can be checked
- CUPED reduced retained-run variance by **50.7%**
- machine-readable verification output

[Project](projects/experiment_lab/)

## Additional strong projects

- **[Customer churn prediction](projects/customer_churn_prediction/):** protected holdout, grouped validation, out-of-fold calibration and cost-aware thresholding; PR-AUC **0.955**.
- **[Image classification](projects/image_classification_confidence/):** EfficientNet-B0, bootstrap uncertainty, selective prediction, Grad-CAM and verified TorchScript/ONNX parity; **85.9%** test accuracy.
- **[Energy demand forecasting](projects/energy_demand_forecasting/):** TensorFlow 14-day forecast with MAE **43.51 GWh** vs **53.18 GWh** seasonal baseline.
- **[PySpark clickstream analytics](projects/pyspark_clickstream_analytics/):** 165,474 real events, Spark transformations, a clearly separated one-million-row load test and leakage-aware conversion modelling.
- **[UK house price prediction](projects/uk_house_price_prediction/):** 995,059 modelling transactions and a strict 2026 temporal test set using a strong postcode/property baseline.
- **[ModelWatch](projects/model_watch/):** feature drift, discrimination, calibration and retraining-policy checks using deliberately shifted batches.

## Engineering standard

Before I treat a result as portfolio evidence, I try to make the following visible:

- source and schema checks
- validation or holdout design that matches intended use
- leakage and duplicate controls where relevant
- a meaningful baseline
- threshold selection before final test evaluation
- uncertainty, calibration or stability checks when they add value
- machine-readable result files
- tests / self-tests for packaged projects
- GitHub Actions for the main production-style work
- limitations stated explicitly rather than hidden behind a headline metric

## Repository layout

```text
projects/     production-style projects I would discuss in interviews
verified/     retained machine-readable evidence from earlier project versions
extensions/   hardened rerun / verification code for older notebooks
docs/         complete catalogue and supporting documentation
*.ipynb       executed university, learning and historical laboratory notebooks
```

The complete project inventory is in [`docs/PROJECT_CATALOG.md`](docs/PROJECT_CATALOG.md). Older notebooks remain available as learning history, but I do not use unverified results from them as production or clinical claims.

## Main tools

**Data:** Python, SQL, Pandas, NumPy, PostgreSQL, DuckDB, PySpark  
**ML:** scikit-learn, CatBoost, PyTorch, TensorFlow/Keras  
**Applied AI:** retrieval, NLP, LLM/RAG evaluation, FastAPI  
**Engineering:** Docker, Git, GitHub Actions, testing, model/data validation

## Licence

My own code and documentation are MIT-licensed. External datasets and pretrained models retain their original licences and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
