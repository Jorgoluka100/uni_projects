# Tableau — Executive Commerce Intelligence

The Tableau version consumes `../data/tableau_dashboard_long.csv`, a tidy cross-tool export produced by `prepare_bi_data.py`.

## Workbook structure

- **Executive Pulse** — KPI text/table cards
- **Monthly Trend** — monthly GMV / order movement
- **Category Leaders** — ranked category value
- **Delivery Experience** — on-time vs late review-score comparison
- **Executive Commerce Dashboard** — combined dashboard layout

[`ExecutiveCommerce.twb`](ExecutiveCommerce.twb) is committed as XML workbook source so the Tableau work is inspectable in Git rather than existing only as a binary screenshot.

## Calculations

See [`calculations.md`](calculations.md) for the calculated-field logic and KPI semantics.

## Local render checkpoint

Run the data-prep script first and ensure the workbook's text-file connection points at the generated `tableau_dashboard_long.csv`. Tableau Desktop/Public is not available in this execution environment, so the final publish/screenshot step should be performed in Tableau after opening and refreshing the workbook.
