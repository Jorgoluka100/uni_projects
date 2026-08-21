# E-commerce Sales & Customer Analysis — SQL / DuckDB

A reproducible analytics-engineering project built around the Olist Brazilian e-commerce dataset. The main challenge is not writing a `GROUP BY`: it is getting the **grain, joins and commercial definitions right before trusting any KPI**.

## What I wanted to answer

- How many genuinely commercial orders and customers are in the comparable reporting window?
- Which months and merchandise categories contribute the most value?
- How much repeat purchasing is visible?
- What does cohort retention look like without pretending later cohorts have had equal time to mature?
- How do late deliveries relate to review outcomes?
- Is marketplace merchandise value concentrated among a small number of sellers?

## Verified headline result

| Metric | Retained result |
|---|---:|
| Commercial orders | **98,199** |
| Unique customers | **94,983** |
| Merchandise value | **R$13.49M** |
| Repeat customers | **3.03%** |
| Strongest complete month | **Nov 2017 — R$1.004M** |
| Leading category | **health_beauty — R$1.251M** |
| Top-10 seller share | **13.24%** |

The retained figures come from the pinned Olist dataset version 7 and the complete-month commercial window **1 Jan 2017 to 31 Aug 2018**. See [`results/verified_summary.json`](results/verified_summary.json).

## The join problem I deliberately solve

An Olist order can contain multiple items, multiple payment rows and more than one review record. A naive join such as:

```sql
orders
JOIN order_items USING (order_id)
JOIN payments USING (order_id)
```

can multiply rows and inflate monetary values. If an order has two items and two payment records, the combined join can create four rows.

I therefore build two explicit semantic grains:

```text
analytics.order_mart  -> exactly 1 row per order
analytics.item_mart   -> exactly 1 row per order item
```

Items and payments are aggregated **before** they are joined to the order grain; reviews are ranked and reduced to one record per order. The synthetic self-test includes a deliberate 2-item × 2-payment example: the naive join produces R$300 of duplicated item value while the order mart correctly retains R$150.

See [`DATA_MODEL.md`](DATA_MODEL.md) for the full grain contract.

## SQL work

The project is split into readable SQL modules rather than hiding the analysis inside notebook JSON:

| File | Purpose |
|---|---|
| [`01_build_marts.sql`](sql/01_build_marts.sql) | order/item semantic marts, child-table aggregation, `ROW_NUMBER` review dedupe |
| [`02_commercial_kpis.sql`](sql/02_commercial_kpis.sql) | headline KPIs, monthly trends, MoM growth, repeat-customer rate |
| [`03_customer_retention.sql`](sql/03_customer_retention.sql) | acquisition cohorts, right-censored retention, customer-value segmentation |
| [`04_marketplace_operations.sql`](sql/04_marketplace_operations.sql) | category, delivery/review, seller, payment and state analysis |
| [`05_seller_concentration.sql`](sql/05_seller_concentration.sql) | seller share, cumulative concentration and HHI-style summary |

The queries use CTEs, conditional aggregation, window functions, `ROW_NUMBER`, `RANK`, `NTILE`, `LAG`, `QUALIFY`, date arithmetic and explicit grain control.

## A few findings worth discussing

Only **3.03%** of customers placed at least two commercial orders in the comparable complete-month window. That makes retention a more interesting business question than simply celebrating top-line order growth.

Late deliveries had an average review score of **2.55/5**, compared with **4.28/5** for on-time or early deliveries. Low-review rates were **54.55% vs 9.44%** respectively. I treat this as an **association**, not proof that lateness caused the review outcome.

The ten largest sellers represented **13.24%** of merchandise value in the analysis window, which provides a starting point for supplier-concentration monitoring rather than relying on intuition.

## Reproducibility and data provenance

The runner downloads a specific Kaggle dataset version and checks both the archive hash and each expected CSV hash before analysis. A changed upstream file therefore fails rather than silently producing different results.

```text
Olist Kaggle dataset version: 7
archive SHA-256:
d521eb1d4a8b6dae030aa429380787261d3b04cd95bee0f43f18cb9cb18ffebb
```

The code then loads the source files into a local DuckDB warehouse, runs the numbered SQL modules, executes integrity tests and exports compact Parquet outputs.

## Integrity checks

A full run fails if it finds:

- duplicate core primary keys;
- orphaned required foreign keys;
- more or fewer than one `order_mart` row per source order;
- an item-mart row-count mismatch;
- merchandise-value disagreement between the order and item grains;
- negative item/freight values; or
- review scores outside the expected 1–5 range.

The fast synthetic test separately verifies the classic many-to-many join trap, latest-review selection, cancelled-order exclusion, repeat-customer cohort logic and top-three category ranking.

## Run it

```bash
cd projects/ecommerce_sql_analytics
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# no download required
python run.py --self-test
python -m unittest discover -s tests -v

# downloads the pinned source, builds DuckDB and verifies retained results
python run.py --output-dir artifacts --database artifacts/ecommerce.duckdb
```

## Outputs

A full run writes a DuckDB database, `verification.json` and Parquet exports for the headline KPIs, monthly performance, retention, category performance, delivery/review summary, seller operations, seller concentration, payment behaviour and state/category rankings.

## What I would do next

For a live marketplace I would move the marts into a managed warehouse, schedule data-quality tests, add incremental models and expose the trusted tables to a BI layer. I would also separate **GMV, revenue, margin and profit** rather than using merchandise value as a proxy for business profitability.

## Limitations

This is historical anonymised marketplace data, not a view of Olist today. Merchandise value is not profit. Delivery/review differences are observational. Later cohorts are right-censored and should not be compared as though they have identical follow-up time.
