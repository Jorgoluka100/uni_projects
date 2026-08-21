# Clickstream Analytics & Conversion Prioritisation with PySpark

This project has three separate parts because I did not want to mix real customer behaviour with a scale test or pretend that an engagement event was a purchase.

## What the project contains

1. **Real clickstream analysis:** 165,474 events across 24,026 sessions.
2. **Spark load test:** 1,000,000 replicated rows, clearly labelled and used only to exercise the Spark pipeline at a larger scale.
3. **Conversion model:** a different UCI dataset with 12,330 sessions and a real `Revenue` label.

## Clickstream findings

The event dataset does not contain cart or purchase events, so the funnel below is about browsing depth only.

| Stage | Sessions | Share of all sessions |
| --- | ---: | ---: |
| All sessions | 24,026 | 100.0% |
| 2+ clicks | 18,984 | 79.0% |
| 5+ clicks | 11,007 | 45.8% |
| 10+ clicks | 5,212 | 21.7% |

Before creating session-level features, the pipeline checks for nulls and duplicate `(session_id, click_order)` keys. Both checks return zero on the source data used here.

## Conversion model

The purchase dataset has **12,330 sessions** and **1,908 conversions**, a positive rate of **15.47%**.

I use a deterministic content hash for the split:

- train: 8,597
- validation: 1,932
- test: 1,801

I exclude `PageValues` because it is calculated using completed transactions. Including it in a model that is supposed to score an active session would create a leakage problem.

The Spark ML pipeline uses:

- categorical indexing and one-hot encoding
- vector assembly
- gradient-boosted trees (`maxIter=50`, `maxDepth=5`)
- threshold selection on validation data only

## Test result

The validation set selected a threshold of **0.25**.

| Metric | Result |
| --- | ---: |
| PR-AUC | **0.351** |
| Conversion prevalence | 0.155 |
| ROC-AUC | 0.763 |
| Precision | 34.3% |
| Recall | 47.7% |
| F1 | 0.399 |
| Alert rate | 22.2% |

PR-AUC is about **2.27×** the raw positive rate. I also saved and reloaded the Spark pipeline; the check scores matched with maximum difference **0.0**.

## About the one-million-row test

The real clickstream has 165,474 events. To test the Spark transformations at a larger row count, the notebook repeats those events until the table reaches one million rows and adds a replica marker.

I do not use those repeated rows for funnel counts, customer numbers or model training.

## Structure

```text
projects/pyspark_clickstream_analytics/
├── README.md
├── MODEL_CARD.md
├── run.py
├── src/
│   ├── analytics.py
│   └── conversion.py
├── results/
│   └── verified_metrics.json
└── requirements.txt
```

Checks:

```bash
pip install -r requirements.txt
python run.py --self-test
python run.py --check-evidence
```

The executed notebook remains available as the full analysis and training record:
[`06_Clickstream_Analysis_with_PySpark.ipynb`](../../06_Clickstream_Analysis_with_PySpark.ipynb)

## Limitations

The two UCI datasets are historical and come from different e-commerce settings. The event source cannot support purchase claims, and the conversion model does not tell us whether an intervention would cause a customer to buy. A production version would need recent consented data, privacy review, monitoring and a controlled experiment.