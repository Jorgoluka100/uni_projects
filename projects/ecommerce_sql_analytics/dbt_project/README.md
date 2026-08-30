# dbt Analytics Engineering Extension

This folder adds a **dbt + DuckDB analytics-engineering layer** to the larger Olist SQL project.

The existing project proves full-dataset SQL analysis, reconciliation and business interpretation. This extension focuses on a different hiring signal: **can analytical transformations be organised as documented, testable warehouse models rather than one long SQL script?**

## Model flow

```text
raw source tables
      ↓
staging views
      ↓
intermediate grain-safe aggregations
      ↓
marts.fct_orders
      ↓
schema + relationship + custom data-quality tests
```

The dbt project keeps the same core rule as the main SQL implementation: items, payments and reviews are reduced to the order grain **before** they are joined to orders. This prevents many-to-many multiplication of monetary values.

## What this demonstrates

- dbt project structure and profiles
- source declarations and contracts
- staging, intermediate and mart layers
- explicit semantic grain
- `ref()` and `source()` dependency management
- uniqueness, not-null and relationship tests
- custom SQL data-quality tests
- DuckDB as a local analytical warehouse
- CI-ready `dbt build`

## Fast local smoke test

From the repository root:

```bash
python -m pip install -r projects/ecommerce_sql_analytics/dbt_project/requirements-dbt.txt
python projects/ecommerce_sql_analytics/dbt_project/prepare_fixture.py
dbt debug --project-dir projects/ecommerce_sql_analytics/dbt_project --profiles-dir projects/ecommerce_sql_analytics/dbt_project
dbt build --project-dir projects/ecommerce_sql_analytics/dbt_project --profiles-dir projects/ecommerce_sql_analytics/dbt_project
dbt docs generate --project-dir projects/ecommerce_sql_analytics/dbt_project --profiles-dir projects/ecommerce_sql_analytics/dbt_project
```

`prepare_fixture.py` creates a deterministic warehouse fixture containing the exact join trap the main project discusses: one order has two item rows and two payment rows. The final dbt mart must still keep one order row with the correct merchandise value.

The **full Olist results are still produced and verified by the parent project**. The synthetic fixture here exists only to make the dbt model graph and tests quick to run in CI without repeatedly downloading the external dataset.

## Production mapping

A real team could point the same source layer at Snowflake, BigQuery, Redshift or another supported warehouse, run the models through an orchestrator such as Airflow or Prefect, and publish dbt documentation/lineage as part of deployment. This repository does not claim to operate those managed services; it demonstrates the transformation contracts and model structure locally.
