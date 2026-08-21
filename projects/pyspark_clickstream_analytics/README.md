# Clickstream Analytics & Conversion Prioritisation with PySpark

A Spark project that separates **behavioural analytics, scale testing and purchase modelling** so the claims stay honest.

## Three distinct workloads

1. **Real clickstream analytics:** 165,474 events across 24,026 sessions.
2. **Spark load test:** exactly 1,000,000 clearly labelled replicated rows; never used to inflate business metrics.
3. **Conversion modelling:** a separate UCI dataset with 12,330 sessions and a real `Revenue` label.

## Real clickstream findings

The event source has no cart or purchase event, so the funnel measures engagement depth:

| Stage | Sessions | Share of all sessions |
| --- | ---: | ---: |
| All sessions | 24,026 | 100.0% |
| 2+ clicks | 18,984 | 79.0% |
| 5+ clicks | 11,007 | 45.8% |
| 10+ clicks | 5,212 | 21.7% |

The pipeline validates zero null cells and zero duplicate `(session_id, click_order)` keys before session aggregation.

## Conversion model

The second dataset contains **12,330 sessions**, including **1,908 conversions** (15.47%). A deterministic content hash creates disjoint 70/15/15 train, validation and test sets:

- train: 8,597;
- validation: 1,932;
- test: 1,801.

`PageValues` is excluded because it depends on completed transactions and can leak purchase outcome information into a model intended for active sessions.

Spark ML uses:

- categorical indexing + one-hot encoding;
- vector assembly;
- gradient-boosted trees (`maxIter=50`, `maxDepth=5`);
- validation-only threshold selection.

## Test result

At the validation-selected threshold of **0.25**:

| Metric | Result |
| --- | ---: |
| PR-AUC | **0.351** |
| Conversion prevalence | 0.155 |
| ROC-AUC | 0.763 |
| Precision | 34.3% |
| Recall | 47.7% |
| F1 | 0.399 |
| Alert rate | 22.2% |

The PR-AUC is about **2.27×** the raw positive rate. The saved Spark pipeline reproduced scores with maximum reload delta **0.0**.

## Scale test without fake scale claims

The real clickstream contains 165,474 events. To exercise distributed transformations at a larger row count, the notebook replicates those rows to one million and adds an explicit replica marker.

Those rows are **not** used for funnel counts, customer claims or model training.

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

The executed notebook remains the full analysis/training record:
[`06_Clickstream_Analysis_with_PySpark.ipynb`](../../06_Clickstream_Analysis_with_PySpark.ipynb)

## Limitations

The datasets are historical and come from different e-commerce contexts. The event-level source cannot support purchase claims, and the conversion classifier does not estimate causal treatment uplift. Current production use would require consented recent data, privacy review, monitoring and an experiment.
