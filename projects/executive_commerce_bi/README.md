# Executive Commerce Intelligence — Power BI + Tableau

A dual-tool business-intelligence project built on the same verified Olist e-commerce warehouse used in my SQL / dbt work.

The purpose is deliberately different from a modelling project: **turn validated commercial data into decisions that an operations, commercial or product team could use.**

![Dashboard portfolio preview](dashboard_preview.svg)

## Business question

**Where is marketplace value being created, where is customer experience breaking down, and what should leadership investigate first?**

The upstream warehouse protects reporting grain before the BI layer sees the data. Orders, items, payments and reviews are reconciled before export so the dashboards do not silently inflate monetary totals through many-to-many joins.

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

These are retained results from the verified SQL project rather than values typed into dashboard cards.

## Two tools, two jobs

### Power BI — Executive Commerce Command Center

**Use case:** recurring operational and leadership review.

The source-controlled PBIP / PBIR report now has two pages:

1. **Executive Overview** — headline KPIs, monthly GMV, category performance and delivery experience.
2. **Market & Operations** — regional value, regional delivery risk, payment behaviour and seller operational risk.

The TMDL semantic model contains **seven explicit tables** and reusable DAX measures rather than relying on implicit visual calculations.

**[Open Power BI source →](power_bi/)**

### Tableau — Marketplace Story & Explorer

**Use case:** exploratory analysis and presentation-led investigation.

The Tableau workbook now contains **seven analytical worksheets and two dashboards**:

- **Executive Commerce Dashboard** — KPI pulse, monthly trend, category leaders and delivery experience.
- **Marketplace Explorer** — regional performance, payment mix and seller risk.

The `.twb` workbook is committed as inspectable XML and contains filters, shelves and mark definitions rather than being an empty workbook shell.

**[Open Tableau source →](tableau/)**

## One governed data path

Both tools consume data generated from the same pinned, integrity-checked warehouse. The strongest reproducibility path is one command from repository root:

```bash
pip install -r projects/ecommerce_sql_analytics/requirements.txt
pip install pyarrow
python projects/executive_commerce_bi/refresh_verified_data.py
```

That refresh:

1. downloads the pinned Olist dataset version and verifies source hashes;
2. rebuilds the DuckDB warehouse with the existing SQL modules;
3. runs warehouse integrity checks;
4. compares the rebuild with retained executed headline evidence;
5. creates governed Power BI / Tableau extracts;
6. adds regional and seller-operational views for deeper BI analysis;
7. writes a manifest containing row counts, columns, hashes and verification evidence.

Generated dashboard data includes:

- `executive_kpis.csv`
- `monthly_performance.csv`
- `category_performance.csv`
- `delivery_review_summary.csv`
- `payment_behaviour.csv`
- `state_category_mix.csv`
- `state_performance.csv`
- `seller_operational_review.csv`
- `tableau_dashboard_long.csv`
- `manifest.json`

## What a recruiter can inspect quickly

| Evidence | What it demonstrates |
| --- | --- |
| [`dashboard_preview.svg`](dashboard_preview.svg) | clear business communication and dashboard hierarchy |
| [`power_bi/`](power_bi/) | PBIP, PBIR, TMDL, Power Query-style ingestion and DAX measures |
| [`tableau/ExecutiveCommerce.twb`](tableau/ExecutiveCommerce.twb) | Tableau workbook XML, worksheets, filters, shelves, marks and dashboard layouts |
| [`KPI_DICTIONARY.md`](KPI_DICTIONARY.md) | KPI definitions, grain and caveats |
| [`DASHBOARD_STORY.md`](DASHBOARD_STORY.md) | audience, decisions and page-level UX design |
| [`refresh_verified_data.py`](refresh_verified_data.py) | real-data lineage from pinned source to dashboard extracts |
| [`tests/`](tests/) + GitHub Actions | automated BI source and evidence contracts |

## Analytical decisions I would discuss in an interview

- I use **merchandise value / GMV**, not “profit”, because cost and margin data are not available.
- I keep reporting grain controlled upstream instead of trying to repair duplicated revenue inside a dashboard.
- The **3.03% repeat-customer rate** makes retention a more interesting management question than merely celebrating order volume.
- Late deliveries are associated with materially weaker review scores (**2.55 vs 4.28**), but I do not describe that observational relationship as causal proof.
- Seller and regional views are designed as **investigation queues**, not automated business decisions.

## Supporting documentation

- [`KPI_DICTIONARY.md`](KPI_DICTIONARY.md) — definitions, grain and caveats
- [`DASHBOARD_STORY.md`](DASHBOARD_STORY.md) — page-by-page analytical and UX design
- [`prepare_bi_data.py`](prepare_bi_data.py) — reusable governed-export layer and synthetic contract self-test
- [`project_notebook.ipynb`](project_notebook.ipynb) — interview-style walkthrough

## Verification boundary

The **real-data refresh, retained KPI evidence, Power BI/Tableau source contracts and project structure are CI-checked**. The repository does **not** claim that Power BI Desktop or Tableau Desktop/Public is running inside GitHub Actions.

A final interactive open/refresh/publication in the desktop applications is therefore a separate publication checkpoint. That boundary is stated explicitly rather than using screenshots to imply runtime verification that has not occurred.

## Skills demonstrated

**Power BI · Tableau · DAX · TMDL · PBIP/PBIR · dashboard design · KPI definition · commercial analytics · data visualisation · SQL lineage · DuckDB · data quality · reproducible reporting · GitHub Actions**
