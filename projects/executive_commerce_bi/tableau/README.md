# Tableau — Marketplace Story & Explorer

This is the exploratory / storytelling half of **Executive Commerce Intelligence**. It consumes the governed `../data/tableau_dashboard_long.csv` extract generated from the same verified warehouse as the Power BI report.

The Tableau work is intentionally **not** a clone of Power BI. Power BI acts as the operational command center; Tableau is used to move through the marketplace story and compare commercial opportunity with customer and service quality.

## Workbook structure

The workbook contains **11 analytical worksheets across 3 dashboards**.

### Dashboard 1 — Executive Commerce Dashboard

- **Executive Pulse** — headline commercial KPIs
- **Monthly Trend** — monthly merchandise-value movement
- **Category Leaders** — ranked category value with supporting freight signal
- **Delivery Experience** — overall on-time vs late review-score comparison

### Dashboard 2 — Marketplace Explorer

- **Regional Performance** — value by customer state, with late-delivery rate available as a risk encoding
- **Payment Mix** — payment value by method with order penetration as supporting context
- **Seller Risk** — seller counts by operational-review status

### Dashboard 3 — Customer & Service Drivers

- **Customer Value** — GMV share by one-time/repeat and high-value customer segments
- **Cohort Retention** — weighted retention by month since acquisition, with observable cohort count as context
- **Delay Severity** — average review score by delivery-delay bucket, with 1-star share as supporting risk context
- **Operational Priority** — states/categories ranked by merchandise value attached to late orders, with late-delivery rate as context

The cohort view is deliberately right-censored: observed months with zero returning customers are retained, while months that have not yet occurred for newer cohorts are excluded.

The dashboards answer three levels of question:

1. **What is happening?** — executive performance and headline customer experience.
2. **Where should I investigate?** — geography, payment behaviour and seller operations.
3. **What appears to be driving the risk?** — customer value mix, cohort retention, delivery severity and material late-order exposure.

## Inspectable Tableau source

[`ExecutiveCommerce.twb`](ExecutiveCommerce.twb) is committed as plain-text Tableau workbook XML. The worksheets include actual filters, shelves and mark definitions rather than only sheet names, and all three dashboard layouts are version controlled.

The shared tidy extract contains the original executive/trend/category/delivery/region/payment/seller sections plus the executed deeper-analysis sections:

- `Customer Segment`
- `Cohort Retention`
- `Delay Impact`
- `Operational Priority`

Those additional sections are generated from verified analysis outputs rather than manually entered into the workbook.

## Reproduce the data

From repository root:

```bash
pip install -r projects/ecommerce_sql_analytics/requirements.txt
pip install pyarrow
python projects/executive_commerce_bi/refresh_verified_data.py
python projects/executive_commerce_bi/cohort_analysis.py
python projects/executive_commerce_bi/enrich_tableau_analysis.py
```

Then open `ExecutiveCommerce.twb` in Tableau Desktop / Tableau Public and refresh the text-file connection if your local path differs.

## Verification boundary

CI parses the workbook as XML and verifies the required **11 worksheets, 3 dashboards, shelves/marks and governed analysis outputs**. Tableau Desktop/Public is not available in CI, so publication and final interactive rendering remain a Desktop/Public checkpoint rather than something this repository falsely claims to have automated.
