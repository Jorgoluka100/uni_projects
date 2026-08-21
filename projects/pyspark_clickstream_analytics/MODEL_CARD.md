# Model card — clickstream conversion prioritisation

## Purpose

Support exploratory merchandising and **human-reviewed** conversion outreach prioritisation while keeping clickstream analytics and purchase modelling logically separate.

## Two datasets, two claims

### Clickstream behaviour

UCI Clickstream Data for Online Shopping:

- 165,474 real click events;
- 24,026 real sessions;
- April–August 2008;
- no purchase outcome.

Because this source has no purchase event, its funnel is an **engagement-depth funnel**, not a sales funnel.

### Conversion model

UCI Online Shoppers Purchasing Intention:

- 12,330 sessions;
- 1,908 revenue-positive sessions;
- deterministic hash split: 8,597 train / 1,932 validation / 1,801 test.

`PageValues` is deliberately excluded because it is calculated using completed transactions and would create leakage for a real-time conversion model.

## Model

Spark ML gradient-boosted trees with indexed/one-hot categorical fields and assembled numeric features. The validation set selects the operating threshold; the test set is used only for final reporting.

## Verified test evidence

At validation-selected threshold **0.25**:

- PR-AUC: **0.351** vs **0.155** conversion prevalence;
- ROC-AUC: **0.763**;
- precision: **34.3%**;
- recall: **47.7%**;
- F1: **0.399**;
- alert rate: **22.2%**.

The saved Spark pipeline reproduced probabilities with maximum reload delta **0.0**.

## Scale claim

A one-million-row table is created by replication **only to exercise Spark transformations**. It is explicitly excluded from business metrics and model training. It is not presented as one million unique shoppers.

## Limitations

Both datasets are historical. The engagement data contains no purchase outcome, while the purchase-intention data is session-level and observational. Feature importance does not establish causal uplift from an intervention.

Production use would require current consented event data, privacy review, identity/session governance, drift monitoring, capacity-aware thresholds and controlled experiments measuring incremental conversion.

## Not for

- automatic pricing or customer exclusion;
- claiming engagement stages are purchases;
- treating replicated load-test rows as unique observations;
- claiming a model score is causal purchase intent.
