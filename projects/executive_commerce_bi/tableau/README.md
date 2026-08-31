# Tableau — Marketplace Story & Explorer

This is the exploratory / storytelling half of **Executive Commerce Intelligence**. It consumes the governed `../data/tableau_dashboard_long.csv` extract generated from the same verified warehouse as the Power BI report.

The Tableau work is intentionally **not** a clone of Power BI. Power BI acts as the operational command center; Tableau is used to move through the marketplace story and compare commercial opportunity with service quality.

## Workbook structure

### Dashboard 1 — Executive Commerce Dashboard

- **Executive Pulse** — headline commercial KPIs
- **Monthly Trend** — monthly merchandise-value movement
- **Category Leaders** — ranked category value with supporting freight signal
- **Delivery Experience** — on-time vs late review-score comparison

### Dashboard 2 — Marketplace Explorer

- **Regional Performance** — value by customer state, with late-delivery rate available as a risk encoding
- **Payment Mix** — payment value by method with order penetration as supporting context
- **Seller Risk** — seller counts by operational-review status

The workbook therefore answers two different questions:

1. **What is happening?** — executive performance and customer experience.
2. **Where should I investigate?** — geography, payment behaviour and seller operations.

## Inspectable Tableau source

[`ExecutiveCommerce.twb`](ExecutiveCommerce.twb) is committed as plain-text Tableau workbook XML. The worksheets include actual filters, shelves and mark definitions rather than only sheet names, and the two dashboard layouts are version controlled.

The shared data extract is a tidy long table with sections for executive KPIs, trends, categories, delivery, regions, payments and seller risk. See [`calculations.md`](calculations.md) for field semantics and calculated-field logic.

## Reproduce the data

From repository root:

```bash
pip install -r projects/ecommerce_sql_analytics/requirements.txt
pip install pyarrow
python projects/executive_commerce_bi/refresh_verified_data.py
```

Then open `ExecutiveCommerce.twb` in Tableau Desktop / Tableau Public and refresh the text-file connection if your local path differs.

## Verification boundary

CI parses the workbook as XML and verifies the required worksheets, both dashboards, shelves/marks and governed data outputs. Tableau Desktop/Public is not available in CI, so publication and final interactive rendering remain a Desktop/Public checkpoint rather than something this repository falsely claims to have automated.
