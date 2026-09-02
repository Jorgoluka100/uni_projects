# Apache Spark Retail Intelligence — Customer 360 Pipeline

A standalone Apache Spark / PySpark portfolio application for building a customer-360 feature layer and repeat-purchase risk model from a high-volume retail event stream.

## Decision problem

An e-commerce operations team needs a scalable pipeline that converts raw transaction events into trusted customer features, business KPIs and a model-ready table. The application demonstrates how the same transformations can operate on a small local sample or a one-million-row Spark workload.

## Data

The project generates a deterministic, explicitly synthetic retail event stream inside Spark. Synthetic data is used here to make the large-scale engineering benchmark fully reproducible and licence-free. The generator includes customers, products, channels, event timestamps, quantities, prices, discounts, returns and a repeat-purchase label.

## What this project demonstrates

- SparkSession configuration and reproducibility
- explicit schemas and data contracts
- deterministic million-row data generation
- null, range and uniqueness validation
- deduplication and invalid-record quarantine
- Spark SQL / DataFrame transformations
- window functions
- customer-360 feature engineering
- cohort and channel KPIs
- train/test separation
- Spark ML feature assembly and scaling
- classification with Spark ML
- AUC / accuracy evaluation
- partition-aware Parquet output
- model persistence
- run metrics and evidence
- tests for core transformation contracts

## Run

```bash
python run.py --rows 1000000
```

For a quick local check:

```bash
python run.py --rows 50000
```

## Portfolio files

- `project_notebook.ipynb` — end-to-end recruiter notebook
- `run.py` — Spark application
- `tests/test_spark_pipeline.py` — transformation tests
- `results/` — generated KPIs/model metrics
- `artifacts/` — persisted Spark ML pipeline

## Limitations

The workload is synthetic and is designed to prove distributed engineering patterns rather than estimate a real retailer's commercial behaviour. Production deployment would read governed event tables, use event-time SLAs, include schema-registry enforcement, profile partition skew and run on a managed Spark platform with lineage and orchestration.
