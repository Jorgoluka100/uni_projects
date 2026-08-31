# Power BI — Executive Commerce Command Center

This is the operational BI half of **Executive Commerce Intelligence**. It uses the source-control-friendly **Power BI Project (PBIP)** format with a **PBIR report** and **TMDL semantic model**, so the data model, DAX and report definitions are inspectable in Git rather than hidden inside a binary file.

## Report design

### Page 1 — Executive Overview

Designed for a commercial or product lead who needs the answer quickly:

- commercial orders
- merchandise value
- repeat-customer rate
- monthly GMV trend
- category leaders
- delivery / review experience

### Page 2 — Market & Operations

Designed for the next question: **where should the team investigate?**

- late-delivery rate
- delivery review-score gap
- seller-review priority count / rate
- merchandise value by customer state
- late-delivery rate by customer state
- payment value by payment type
- seller operational-status split

### Page 3 — Customer & Service Drivers

Designed to connect the commercial result to the customer and operational drivers behind it:

- **Customer value mix** — GMV share across one-time, one-time high-value, repeat and repeat high-value customers
- **Delivery severity** — average review score across on-time and progressively later delivery buckets
- **Operational priority** — states and categories ranked by merchandise value attached to late orders

This page is fed by the deeper verified analysis tables rather than duplicated dashboard calculations.

## Semantic model

The report is deliberately **model-first rather than visual-first**. The TMDL model contains ten explicit import tables:

1. `ExecutiveKPIs`
2. `MonthlyPerformance`
3. `CategoryPerformance`
4. `DeliveryQuality`
5. `StatePerformance`
6. `PaymentBehaviour`
7. `SellerOperations`
8. `CustomerSegments`
9. `DeliveryImpact`
10. `OperationalPriority`

The final three tables are generated from the deeper verified business analysis:

- **CustomerSegments** — customer count, GMV share and average customer value by one-time/repeat value segment
- **DeliveryImpact** — delay buckets with order volume, GMV, review score, 1-star rate and 5-star rate
- **OperationalPriority** — states/categories ranked transparently by merchandise value attached to late orders

DAX measures include commercial orders, customers, merchandise value, AOV, repeat rate, seller concentration, late-delivery rate, delivery review gap, regional GMV, payment value, priority-seller metrics, segment GMV share, one-star review rate and late-order GMV.

## Reproduce the data

From repository root:

```bash
pip install -r projects/ecommerce_sql_analytics/requirements.txt
pip install pyarrow
python projects/executive_commerce_bi/refresh_verified_data.py
python projects/executive_commerce_bi/enrich_tableau_analysis.py
python projects/executive_commerce_bi/render_dashboard_preview.py
```

The refresh downloads the pinned Olist source, rebuilds the warehouse, runs integrity checks, verifies the retained headline evidence, performs the deeper business analysis and writes the governed dashboard CSVs into `../data/`.

Then edit the `DataRoot` expression in:

`ExecutiveCommerce.SemanticModel/definition/model.tmdl`

so it points at that generated `data/` folder on your machine, and open `ExecutiveCommerce.pbip` in Power BI Desktop.

## What is automatically verified

GitHub Actions checks the real data refresh, retained headline metrics, decision-oriented analysis outputs, generated dashboard preview, Power BI project/report bindings, all three PBIR pages, PBIR JSON, semantic-model table coverage and the shared dashboard evidence.

Power BI Desktop itself is not available in the repository CI environment. I therefore do **not** claim that a `.pbix` binary or published Power BI Service report was runtime-rendered by CI. The source, measures, data lineage, analysis and report definitions are the inspectable evidence here; the final Desktop render is the publication checkpoint.
