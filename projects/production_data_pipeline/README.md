# Production-Style Data Pipeline

This project is deliberately **not** another modelling notebook. It is a small, runnable data-engineering system built to show the parts of a pipeline that are easy to hide in portfolio work: ingestion state, schema checks, bad-row quarantine, deduplication, incremental updates, SQL transformations, idempotency and automated tests.

## Architecture

`JSONL source → bronze raw ingestion → validation/quarantine → silver current orders → gold daily metrics → run manifest`

The local implementation uses Python and SQLite so the complete pipeline runs without cloud credentials or paid infrastructure. The design maps cleanly to an object-store + warehouse setup: source files in S3/GCS/Azure Blob, bronze/silver/gold tables in a cloud warehouse or lakehouse, orchestration with Airflow/Prefect, and SQL modelling with dbt.

## Engineering behaviour

- **Incremental ingestion:** each source file is fingerprinted with SHA-256 and recorded in an ingestion log.
- **Idempotency:** rerunning an unchanged source file is a no-op rather than duplicating records.
- **Bronze layer:** every raw row is preserved with source hash, line number and ingestion timestamp.
- **Quarantine:** malformed or invalid rows are isolated with an explicit error reason.
- **Silver layer:** order updates are deduplicated on `order_id`, keeping the newest `updated_at` record.
- **Gold layer:** SQL produces daily business metrics from the validated current-state table.
- **Quality gates:** uniqueness, null, negative-value and domain checks run before the batch is committed.
- **Atomicity:** ingestion and transformations are committed as one transaction; failed checks roll back the run.
- **Machine-readable evidence:** every run writes a JSON manifest.

## Run it

```bash
python pipeline.py --seed-fixture
python pipeline.py --source data/orders.jsonl
```

The seeded fixture intentionally includes a duplicate order update and one invalid negative-value row. The pipeline keeps the newest valid order state and quarantines the invalid row.

## Test it

```bash
python -m unittest discover -s tests -v
```

The test checks that:

1. five raw rows are ingested;
2. one invalid row is quarantined;
3. the duplicate order resolves to the latest status;
4. three valid current-state orders remain;
5. the gold table contains two dates; and
6. rerunning the same file is idempotent.

## Why this project exists

The rest of this portfolio already demonstrates analytics, modelling, PySpark, SQL and applied AI. This project adds a different hiring signal: **can I reason about reliable data movement and data contracts, not just train a model once?**

It is intentionally infrastructure-light. I do not claim that SQLite is a production warehouse or that this demo replaces distributed orchestration. The point is to keep the core engineering behaviour inspectable and runnable in a few seconds.

## Structure

```text
projects/production_data_pipeline/
├── README.md
├── pipeline.py
├── Dockerfile
└── tests/
    └── test_pipeline.py
```

The repository-level GitHub Actions workflow runs the unit test on every relevant change.
