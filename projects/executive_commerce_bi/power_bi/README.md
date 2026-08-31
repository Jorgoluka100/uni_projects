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

## Semantic model

The report is deliberately model-first rather than visual-first. The TMDL model contains seven explicit tables:

1. `ExecutiveKPIs`
2. `MonthlyPerformance`
3. `CategoryPerformance`
4. `DeliveryQuality`
5. `StatePerformance`
6. `PaymentBehaviour`
7. `SellerOperations`

DAX measures include commercial orders, customers, merchandise value, AOV, repeat rate, seller concentration, late-delivery rate, delivery review gap, regional GMV, regional delivery risk, payment value and priority-seller metrics.

## Reproduce the data

From repository root:

```bash
pip install -r projects/ecommerce_sql_analytics/requirements.txt
pip install pyarrow
python projects/executive_commerce_bi/refresh_verified_data.py
```

That command downloads the pinned Olist source, rebuilds the warehouse, runs integrity checks, verifies the retained headline evidence and writes the governed dashboard CSVs into `../data/`.

Then edit the `DataRoot` expression in:

`ExecutiveCommerce.SemanticModel/definition/model.tmdl`

so it points at that generated `data/` folder on your machine, and open `ExecutiveCommerce.pbip` in Power BI Desktop.

## What is automatically verified

GitHub Actions checks the real data refresh, retained headline metrics, Power BI project/report bindings, PBIR JSON, semantic-model table coverage and the shared dashboard evidence.

Power BI Desktop itself is not available in the repository CI environment. I therefore do **not** claim that a `.pbix` binary or published Power BI Service report was runtime-rendered by CI. The source, measures, data lineage and report definitions are the inspectable evidence here; the final Desktop render is the publication checkpoint.
