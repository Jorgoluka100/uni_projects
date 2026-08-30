# Executive Commerce Intelligence — Power BI + Tableau

A dual-tool business-intelligence project built on the same verified Olist e-commerce warehouse used in my SQL / dbt work.

The aim is deliberately different from a modelling project: **turn validated commercial data into decisions that an operations, commercial or product team could use.**

![Dashboard portfolio preview](dashboard_preview.svg)

## Business question

**Where is marketplace value being created, where is customer experience breaking down, and what should leadership investigate first?**

The upstream warehouse protects reporting grain before the BI layer sees the data. Orders, items, payments and reviews are reconciled before export so the dashboards do not silently inflate revenue through many-to-many joins.

## Verified scale

| KPI | Retained value |
| --- | ---: |
| Commercial orders | **98,199** |
| Unique customers | **94,983** |
| Merchandise value | **R$13.49M** |
| Repeat customers | **3.03%** |
| Strongest complete month | **Nov 2017 — R$1.004M** |
| Leading category | **health_beauty — R$1.251M** |
| Top-10 seller share | **13.24%** |
| Avg review — on time / early | **4.28 / 5** |
| Avg review — late | **2.55 / 5** |

These are retained results from the verified SQL project rather than values typed into the dashboard manually.

## Dashboard story

### 1. Executive Pulse

- commercial orders
- unique customers
- merchandise value
- average order value
- repeat-customer rate
- monthly GMV and order trend

**Decision:** Is growth being driven by more customers, more orders, or larger baskets?

### 2. Customer & Category

- category value and order volume
- average review by category
- state-level category mix
- repeat behaviour

**Decision:** Which categories and customer markets deserve deeper commercial attention?

### 3. Delivery & Experience

- on-time vs late delivery share
- average review by delivery status
- seller operational review flags

**Decision:** Where does service quality appear to be damaging customer experience?

### 4. Marketplace Health

- seller concentration
- payment behaviour
- seller value and operational quality

**Decision:** Is value overly concentrated, and where are marketplace dependencies emerging?

## Why both Power BI and Tableau?

The point is not to create two unrelated dashboards. Both tools consume the **same governed BI exports and KPI definitions**. That makes cross-tool consistency inspectable and demonstrates that the analysis logic lives in the data model rather than in one dashboard file.

- [`power_bi/`](power_bi/) — source-controlled PBIP / PBIR + TMDL semantic-model project, DAX measures and starter report page.
- [`tableau/`](tableau/) — Tableau workbook source template, calculation definitions and dashboard layout contract.
- [`KPI_DICTIONARY.md`](KPI_DICTIONARY.md) — definitions, grain and caveats.
- [`DASHBOARD_STORY.md`](DASHBOARD_STORY.md) — page-by-page analytical and UX design.
- [`prepare_bi_data.py`](prepare_bi_data.py) — converts verified Parquet marts into BI-friendly CSVs plus a hashed manifest.

## Reproduce the BI data

First build the upstream warehouse:

```bash
cd ../ecommerce_sql_analytics
pip install -r requirements.txt
python run.py --output-dir artifacts --database artifacts/ecommerce.duckdb
```

Then build the BI exports:

```bash
cd ../executive_commerce_bi
pip install pandas pyarrow
python prepare_bi_data.py \
  --source-dir ../ecommerce_sql_analytics/artifacts/tables \
  --output-dir data
```

Outputs include:

- `executive_kpis.csv`
- `monthly_performance.csv`
- `category_performance.csv`
- `delivery_review_summary.csv`
- `payment_behaviour.csv`
- `state_category_mix.csv`
- `tableau_dashboard_long.csv`
- `manifest.json`

## Engineering / quality evidence

- one governed source for both dashboard tools
- explicit KPI dictionary
- data-prep script with required-column contracts
- output hashes and row counts in `manifest.json`
- source-controlled Power BI semantic model and report definitions
- Tableau workbook XML source
- automated JSON/XML/project-contract checks in GitHub Actions
- no claim that historical marketplace revenue equals profit
- no causal claim that late delivery *causes* poor reviews

## What this project demonstrates

**Power BI · Tableau · data visualisation · DAX · TMDL · PBIP/PBIR · KPI design · dashboard storytelling · commercial analytics · data quality · reproducible reporting**

The project intentionally bridges technical data work and business communication: the upstream SQL makes the numbers trustworthy; the BI layer makes the implications usable.
