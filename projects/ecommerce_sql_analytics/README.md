# E-commerce Sales & Customer Analysis — SQL / DuckDB

This project uses the Olist Brazilian e-commerce dataset. The main thing I wanted to get right was not the number of SQL functions I could use; it was the reporting grain. Orders, items, payments and reviews do not all live at the same level, so a careless join can make revenue look larger than it really is.

## Questions I looked at

- How many commercial orders and customers are in a comparable reporting window?
- Which months and categories contribute the most merchandise value?
- How much repeat purchasing is visible?
- What does cohort retention look like once later cohorts are treated fairly?
- How do late deliveries relate to review scores?
- How concentrated is marketplace value among sellers?

## Headline result

| Metric | Result |
|---|---:|
| Commercial orders | **98,199** |
| Unique customers | **94,983** |
| Merchandise value | **R$13.49M** |
| Repeat customers | **3.03%** |
| Strongest complete month | **Nov 2017 — R$1.004M** |
| Leading category | **health_beauty — R$1.251M** |
| Top-10 seller share | **13.24%** |

These figures come from Olist dataset version 7 and the complete-month window from **1 Jan 2017 to 31 Aug 2018**. The saved summary is in [`results/verified_summary.json`](results/verified_summary.json).

## The join problem

One order can contain several items, several payment rows and more than one review record. Joining everything directly can multiply rows and inflate monetary totals.

For example, an order with two items and two payment records can become four rows after a naive join.

I avoid that by keeping two clear grains:

```text
analytics.order_mart  -> 1 row per order
analytics.item_mart   -> 1 row per order item
```

Items and payments are aggregated before joining to the order level. Reviews are ranked and reduced to one record per order. The synthetic self-test includes a 2-item × 2-payment case: the naive join shows R$300 of duplicated item value, while the order mart correctly keeps R$150.

See [`DATA_MODEL.md`](DATA_MODEL.md) for the grain rules.

## SQL files

| File | What it does |
|---|---|
| [`01_build_marts.sql`](sql/01_build_marts.sql) | builds order/item marts and handles review deduplication |
| [`02_commercial_kpis.sql`](sql/02_commercial_kpis.sql) | headline KPIs, monthly trends, MoM growth and repeat-customer rate |
| [`03_customer_retention.sql`](sql/03_customer_retention.sql) | cohorts, right-censored retention and customer-value segments |
| [`04_marketplace_operations.sql`](sql/04_marketplace_operations.sql) | category, delivery/review, seller, payment and state analysis |
| [`05_seller_concentration.sql`](sql/05_seller_concentration.sql) | seller share, cumulative concentration and HHI-style summary |

The queries use CTEs, conditional aggregation, window functions, `ROW_NUMBER`, `RANK`, `NTILE`, `LAG`, `QUALIFY`, date arithmetic and explicit grain control.

## Findings I would discuss in an interview

Only **3.03%** of customers placed at least two commercial orders in the comparable window. That makes repeat behaviour one of the more interesting questions in the dataset.

Late deliveries had an average review score of **2.55/5**, compared with **4.28/5** for orders delivered on time or early. Low-review rates were **54.55% vs 9.44%**. I treat this as an association, not proof that lateness caused the review.

The ten largest sellers accounted for **13.24%** of merchandise value in the analysis window.

## Data checks and reproducibility

The runner downloads a specific Kaggle dataset version and checks the archive hash and the expected CSV hashes before analysis. If the upstream files change, the run fails instead of silently changing the result.

```text
Olist Kaggle dataset version: 7
archive SHA-256:
d521eb1d4a8b6dae030aa429380787261d3b04cd95bee0f43f18cb9cb18ffebb
```

The source files are loaded into a local DuckDB database, the SQL modules run in order, integrity checks are applied and compact Parquet outputs are exported.

## Integrity checks

A full run fails if it finds:

- duplicate core primary keys
- orphaned required foreign keys
- unexpected mart grain
- revenue mismatch after reconciliation
- join multiplication in the synthetic test
- result values outside the saved evidence contract

## Run it

```bash
pip install -r requirements.txt
python run.py --self-test
python run.py --check-evidence
```

The original notebook remains in the repository as the exploration record:
[`02_SQL_Sales_and_Customer_Analysis.ipynb`](../../02_SQL_Sales_and_Customer_Analysis.ipynb)

## Limitations

The data is historical marketplace data and does not represent current Olist performance. Customer identity is based on the identifiers available in the source. Delivery/review findings are observational, and the analysis should not be read as causal proof.