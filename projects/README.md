# Data & AI Projects

This folder contains the strongest end-to-end projects in my portfolio. They are designed to make the full problem-solving process inspectable: **data quality → feature or data design → modelling / transformation → validation → evaluation → engineering → limitations**.

If you are reviewing this portfolio for a role, start with the evidence most relevant to that role.

## Start by role

| Target role | Best projects | Evidence |
| --- | --- | --- |
| **Data Scientist** | [Flight Delay Risk](flight_delay_risk/) · [Parkinson's Progression](parkinsons_progression/) · [Customer Churn](customer_churn_prediction/) · [UK House Prices](uk_house_price_prediction/) | supervised ML, baselines, feature policy, leakage controls, realistic validation, calibration and thresholding |
| **Data Engineer / Analytics Engineer** | [Reliable Event Pipeline](reliable_event_pipeline/) · [PySpark Clickstream](pyspark_clickstream_analytics/) · [E-commerce SQL Analytics](ecommerce_sql_analytics/) | ingestion, schema validation, deduplication, idempotency, SQL modelling, quality checks and distributed transformations |
| **ML / AI Engineer** | [Grounded RAG](grounded_rag/) · [Image Classification](image_classification_confidence/) · [ModelWatch](model_watch/) | retrieval evaluation, PyTorch, APIs, Docker, model export, calibration, uncertainty and drift monitoring |
| **Data Analyst / Product Analyst** | [E-commerce SQL Analytics](ecommerce_sql_analytics/) · [ExperimentLab](experiment_lab/) · [Retail Customer Segmentation](retail_customer_segmentation/) | SQL, KPI logic, experimentation, customer analysis and business interpretation |

## Flagship projects

### [Reliable Event Pipeline](reliable_event_pipeline/)
A compact data-engineering pipeline covering schema validation, reject handling, duplicate control, late-arriving data, idempotent loads, SQL reconciliation, audit metrics and automated tests.

**Verified demo:** 10 input rows → 7 clean warehouse events, with invalid and duplicate records handled explicitly.

### [Flight Delay Risk](flight_delay_risk/)
A classification project built on official 2026 US flight data with chronological validation and an untouched **180,000-flight** test set.

The retained evaluation focuses on out-of-time generalisation rather than a convenient random split.

### [Grounded RAG](grounded_rag/)
A local retrieval-augmented assistant with hybrid retrieval, source attribution, abstention on weak evidence, prompt-injection checks, read-only tool routing, FastAPI and Docker.

The project includes a frozen evaluation fixture so retrieval, citation and routing behaviour can be tested rather than judged from a demo alone.

### [E-commerce SQL Analytics](ecommerce_sql_analytics/)
An end-to-end SQL analytics project over **98,199 commercial orders**, including relational modelling, reconciliation, cohorts, window functions and join-safety checks.

### [Image Classification with Confidence](image_classification_confidence/)
A PyTorch / EfficientNet-B0 computer-vision project with uncertainty checks, selective prediction, Grad-CAM and verified model export.

**Retained test accuracy:** **85.9%**.

### [Parkinson's Progression](parkinsons_progression/)
A regression project that strengthens my original MSc work with explicit feature policy, meaningful baselines, patient-grouped holdout separation and grouped cross-validation so repeated measurements from one person cannot leak across train and evaluation data.

### [Customer Churn Prediction](customer_churn_prediction/)
A classification project with a protected holdout, grouped validation, out-of-fold calibration and cost-aware threshold selection.

### [PySpark Clickstream Analytics](pyspark_clickstream_analytics/)
Distributed transformations and data-quality logic over **165,474 real events**, plus a separate one-million-row load test.

### [ExperimentLab](experiment_lab/)
An experimentation project covering treatment effects, confidence intervals, CUPED, guardrails and statistical power.

### [ModelWatch](model_watch/)
A model-monitoring project that checks data drift, discrimination and calibration against a reference dataset and verifies that deliberately introduced shifts trigger the expected monitoring behaviour.

## Additional applied projects

- [Energy Demand Forecasting](energy_demand_forecasting/) — TensorFlow / time-series forecasting
- [UK House Price Prediction](uk_house_price_prediction/) — regression and reproducible modelling
- [Retail Customer Segmentation](retail_customer_segmentation/) — data cleaning, RFM features and clustering
- [CareerLens AI](careerlens_ai/) — ranking, retrieval evaluation and explainable skill matching
- [Production Data Pipeline](production_data_pipeline/) — additional pipeline and data-engineering practice

## What I want employers to be able to inspect

Across the strongest projects I make the following explicit rather than hiding them behind a final metric:

- where the data came from and what was wrong with it
- how missing values, duplicates, joins or invalid records were handled
- why the train / validation / test strategy matches the problem
- what leakage or unrealistic evaluation would look like
- which baseline the model or method must beat
- how the result is measured and retained
- what tests or checks protect the implementation
- what the project does **not** claim to prove

The repository also retains my original university notebooks and smaller fundamentals exercises so the progression from learning to stronger end-to-end engineering is visible.