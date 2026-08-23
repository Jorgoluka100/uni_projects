# Retail Customer Data Cleaning & Segmentation

This project is deliberately centred on the part of data science that comes before modelling: checking whether the source can be trusted, making cleaning rules explicit, validating the final analytical table and only then building a customer segmentation.

The source is the **UCI Online Retail** transaction dataset. The workbook is downloaded at run time; the raw data is not committed to this repository.

## Verified result

The full pipeline and unit tests ran successfully in GitHub Actions.

- Raw rows: **541,909**
- Exact duplicate rows identified: **5,268**
- Rows with missing customer ID: **135,080**
- Rows with non-positive quantity: **10,624**
- Rows with non-positive unit price: **2,517**
- Clean valid purchase rows: **392,692**
- Customers retained for segmentation: **4,338**
- Valid invoices: **18,532**
- Selected KMeans solution: **k = 2**
- Silhouette score: **0.4335**
- Initialization stability: adjusted Rand index **0.9963–0.9991** across eight additional seeds

The selected solution separated **1,725 high-value active customers** from **2,613 more inactive/lower-value customers**. The first group represented **39.8% of customers** but **86.5% of retained transaction revenue**. I treat those labels as relative descriptions of this dataset, not universal customer personas.

[Verification](results/verification.json) · [Raw audit](results/raw_data_audit.json) · [Cleaning report](results/cleaning_report.json) · [Cluster diagnostics](results/cluster_diagnostics.csv) · [Cluster summary](results/cluster_summary.csv)

## What this project demonstrates

- schema and data-quality inspection before cleaning
- missing-value, duplicate and invalid-value handling
- cancellation detection
- dtype and text normalization
- auditable row-removal rules
- transaction-level validation checks
- customer-level RFM feature engineering
- outlier treatment without silently deleting customers
- log transformation and robust scaling
- KMeans model selection across several values of `k`
- silhouette, Davies-Bouldin and Calinski-Harabasz diagnostics
- minimum-cluster-size guard
- initialization stability using adjusted Rand index
- customer-segment summaries in original business units
- tests for cleaning, features and clustering behaviour
- machine-readable verification outputs

## Why the cleaning stage matters

Retail transaction data contains rows that should not automatically enter a customer model. Cancelled invoices, non-positive quantities, zero prices, missing customer identifiers and exact duplicates can materially distort frequency and spend. The project therefore records the quality issues first and applies explicit rules instead of calling `dropna()` and moving on.

The final clustering dataset uses only valid customer purchases. Each customer then receives:

- **Recency** — days since the most recent purchase
- **Frequency** — number of distinct valid orders
- **Monetary value** — total valid transaction revenue

Additional customer fields are retained for interpretation, including total items, customer tenure and average order value.

## Pipeline

```text
UCI workbook
   |
   v
raw schema + quality audit
   |
   v
explicit transaction cleaning
   |
   v
validation assertions
   |
   v
customer-level RFM features
   |
   v
1st/99th percentile clipping
   |
   v
log1p transformation + RobustScaler
   |
   v
KMeans candidates (k=2..10)
   |
   v
multi-metric model selection
   |
   v
stability check + segment interpretation
```

## Cleaning rules

The pipeline removes exact duplicate rows first. It then marks and removes rows that have any of the following problems:

1. missing customer ID
2. missing invoice number
3. missing stock code
4. missing invoice timestamp
5. missing country
6. cancellation invoice (`InvoiceNo` beginning with `C`)
7. missing or non-positive quantity
8. missing or non-positive unit price

Text fields are stripped and normalized, date/numeric columns are coerced to explicit types, and `line_revenue = Quantity * UnitPrice` is created only after a row has passed the validity checks.

The code asserts that the cleaned dataset has no exact duplicates, no missing required modelling fields and no non-positive quantity, price or revenue.

## Model selection

The clustering stage does **not** choose a number of clusters because an elbow chart "looks right". Candidate values from 2 to 10 are compared using:

- silhouette score — primary selection metric
- Davies-Bouldin score
- Calinski-Harabasz score
- inertia
- smallest-cluster share

A candidate with a smallest cluster below 2% is not preferred when a better-behaved alternative exists. After the final `k` is selected, the solution is refit with a larger `n_init` and compared across several random seeds using adjusted Rand index.

For this dataset, `k=2` produced the strongest silhouette score among the tested candidates. I keep that result instead of forcing a larger number of clusters merely because more segments would look more impressive.

## Run

From this project directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q
python run.py
```

Generated files include:

```text
results/raw_data_audit.json
results/cleaning_report.json
results/verification.json
results/customer_segments.csv
results/cluster_diagnostics.csv
results/cluster_summary.csv
```

The large downloaded workbook and cleaned transaction CSV remain local and are ignored by Git.

## Repository structure

```text
retail_customer_segmentation/
├── README.md
├── PROJECT_CARD.md
├── requirements.txt
├── pytest.ini
├── run.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   └── evaluation.py
├── tests/
│   ├── test_cleaning.py
│   └── test_features_and_model.py
└── results/
    ├── verification.json
    ├── raw_data_audit.json
    ├── cleaning_report.json
    ├── cluster_diagnostics.csv
    └── cluster_summary.csv
```

## Evidence policy

The numerical claims above come from the retained GitHub Actions verification run. The raw workbook and full customer-assignment table are intentionally not committed; the compact audit, cleaning, diagnostic and summary evidence is retained in `results/`.

## Limitations

- KMeans assumes distance-based partitions and may not reflect natural customer groups.
- RFM behaviour does not include marketing exposure, margins, demographics or causal drivers.
- Outlier clipping changes the geometry of the clustering space and is therefore documented explicitly.
- Segment labels are relative descriptions of the observed clusters, not universal customer personas.
- A descriptive segment is not evidence that a marketing intervention will cause higher revenue.
