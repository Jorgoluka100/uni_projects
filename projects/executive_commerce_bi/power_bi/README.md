# Power BI — Executive Commerce Intelligence

This folder uses the source-control-friendly **Power BI Project (PBIP)** pattern with a **TMDL semantic model** and PBIR report source.

## What is implemented

- PBIP shortcut file
- local semantic-model binding
- TMDL model with explicit measures
- import partitions for governed CSV exports
- starter executive report page containing:
  - multi-KPI card
  - monthly GMV line chart
  - category GMV bar chart
  - delivery review-score bar chart

## Data connection

Run `../prepare_bi_data.py` first. Then edit the `DataRoot` expression in:

`ExecutiveCommerce.SemanticModel/definition/model.tmdl`

so it points to the generated `data/` folder on your machine.

## Important validation checkpoint

The JSON/TMDL structure and repository contracts are checked in CI. Power BI Desktop itself is not available in this execution environment, so before publishing a `.pbix` or screenshot, open `ExecutiveCommerce.pbip` in a current Power BI Desktop build, refresh, inspect the visuals and save once locally.

That final Desktop-render step is intentionally separate from the Git-controlled source.
