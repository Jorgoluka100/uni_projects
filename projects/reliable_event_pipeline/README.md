# Reliable Event Ingestion Pipeline

A small data-engineering project focused on the part that often breaks first in production: getting messy incremental files into a trustworthy warehouse table.

The project deliberately avoids adding another predictive model. It demonstrates a different set of skills: ingestion, schema contracts, data-quality checks, duplicate handling, late-arriving data, idempotent loads, SQL reconciliation and automated tests.

## What happens

`CSV batches → schema/row validation → reject log → event-time watermark → idempotent SQLite load → SQL quality checks → batch audit metrics`

Two fixture batches contain the failure cases the pipeline is expected to handle:

- a duplicate ID inside a batch
- a record with a missing customer ID
- an event already present from an earlier batch
- a legitimate late-arriving event

Late data is **kept and flagged** instead of silently dropped. Existing `event_id` values are skipped by the warehouse primary key, so re-sent events do not create duplicate facts.

## Verified demo

The frozen two-batch demo produces:

- **10** input rows across two batches
- **7** warehouse events after validation and deduplication
- **2** rejected rows from the first batch
- **1** previously loaded event skipped in the second batch
- **1** late-arriving event retained and flagged
- **0** duplicate event IDs in the final table
- **0** null customer IDs in the final table

See [`results/verified_run.json`](results/verified_run.json) for the machine-readable run evidence.

## Structure

```text
projects/reliable_event_pipeline/
├── README.md
├── run.py
├── src/pipeline.py
├── sql/quality_checks.sql
├── fixtures/
│   ├── batch_1.csv
│   └── batch_2.csv
├── tests/test_pipeline.py
└── results/verified_run.json
```

## Run

```bash
python run.py --self-test
python -m unittest discover -s tests -v
python run.py
```

The implementation uses Python's standard library and SQLite so the project is easy to inspect and run locally. The same contracts — source mapping, validation, watermarks, idempotent keys, audit metrics and SQL reconciliation — are the parts I would carry into a managed warehouse or orchestration stack.

## What this demonstrates

- Python ingestion code
- SQL and relational data modelling
- messy-data validation and reject handling
- duplicate and idempotency controls
- late-arriving data / watermark logic
- auditable batch metrics
- unit tests and CI-friendly self-tests
- explicit assumptions rather than hidden cleaning steps

## Scope

This is a compact local pipeline, not a claim of operating a cloud-scale production system. It does not pretend SQLite is BigQuery or that two fixture files represent production volume. The purpose is to make the engineering behaviour inspectable and testable.
