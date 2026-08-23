# Project card — Retail Customer Data Cleaning & Segmentation

## Question

Can a messy retail transaction table be turned into a defensible customer-level dataset and then segmented without hiding data-quality decisions?

## Data

UCI Machine Learning Repository, **Online Retail**. The workbook is downloaded at execution time and is not stored in Git.

## Unit of analysis

- Raw/cleaning stage: transaction line
- Modelling stage: customer

## Cleaning contract

A transaction line is eligible for the segmentation base only when it has a customer ID, invoice number, stock code, invoice timestamp and country; is not a cancellation; and has positive quantity and unit price. Exact duplicate lines are removed first. Every rule is counted in the retained cleaning report.

## Features

Primary clustering features are RFM:

- `recency_days`
- `frequency_orders`
- `monetary_value`

Customer tenure, total items, transaction-line count and average order value are retained for interpretation but are not used to create the KMeans distances.

## Preprocessing

RFM fields are clipped at their 1st and 99th percentiles, transformed with `log1p`, and scaled with `RobustScaler`. This is intended to reduce the influence of extreme retail spend/frequency values while keeping all customers in the segmentation.

## Model selection

KMeans candidates from `k=2` through `k=10` are compared using silhouette, Davies-Bouldin, Calinski-Harabasz, inertia and smallest-cluster share. Silhouette is the primary rule, with a 2% smallest-cluster guard where possible.

The chosen solution is then checked across multiple random seeds using adjusted Rand index.

## What would make the project fail verification?

- the source schema changes unexpectedly;
- cleaning leaves duplicates or invalid required fields;
- RFM features contain missing/non-positive values where prohibited;
- the clustering matrix contains non-finite values;
- a customer does not receive exactly one cluster;
- the final solution collapses to one cluster;
- retained evidence cannot be reproduced by `run.py`.

## Interpretation boundary

This is descriptive segmentation, not causal marketing science. A segment can help prioritize investigation or communication strategy, but the project does not claim that targeting a segment will increase revenue.
