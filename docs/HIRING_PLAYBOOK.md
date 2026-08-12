# Hiring Playbook — How to Read This Portfolio

This file turns the repository into a hiring tool. It maps the strongest verified evidence to common junior/graduate roles and makes the interview story explicit.

## 90-second recruiter route

If you only open four projects, use these:

1. **VisionForge** — strongest end-to-end ML engineering / computer-vision example: held-out evaluation, calibration, abstention, Grad-CAM, model export and independent parity checks.
2. **AeroFlow v3** — strongest current-data modelling example: official 2026 BTS data, temporal validation, failed-regression diagnosis, problem reframing and operational ranking lift.
3. **SQL Sales & Customer Analysis** — strongest analytics/SQL example: relational grain control, reconciliation and trustworthy commercial KPIs.
4. **NYC Airbnb 2026** — strongest current market-analysis example: 14-Jun-2026 source, neighbourhood-group holdout and explicit separation of listing-price claims from realised revenue.

## Best projects by target role

| Target role | Lead with | Supporting evidence |
|---|---|---|
| **Junior / Graduate Data Scientist** | AeroFlow, UK House Prices, Customer Churn | NYC Airbnb 2026, Recommender, Parkinson's negative grouped result |
| **Junior Machine Learning Engineer** | VisionForge, Recommender v2 | TensorFlow Energy, Spark KDD pipeline, model reload/export and CI evidence |
| **Data Analyst / Analytics** | SQL Sales, NYC Airbnb 2026, Clickstream | London Air Quality R, PostgreSQL aviation benchmark, bootcamp portfolio |
| **Applied AI / AI Engineer** | VisionForge, ConsultAI | AeroFlow, LLM evaluation harness, evidence gates and reproducibility tooling |
| **Data / ML platform-oriented role** | PySpark Clickstream, Spark KDD, PostgreSQL aviation | SQL Sales, retained artifacts, CI verification workflows |

## Four interview stories worth learning

### 1. I found leakage instead of celebrating a high score

The recommender clean run exposed future interactions in evaluated-user histories. The fix truncated each evaluated user's history strictly before the held-out target, removing **1,756 later interactions** before rerunning ranking metrics.

The Parkinson's project exposed a more serious issue: repeated rows from the same subject had made row-level evaluation too optimistic. A complete-subject holdout produced RF MAE **10.70** versus a median baseline of **8.13** and R² **-1.09**. The weaker result is retained because it is the honest result.

**Interview point:** model validation design matters more than protecting an attractive metric.

### 2. A failed baseline changed the problem formulation

AeroFlow's first 2026 schedule-time regression passed its engineering checks but failed to beat the zero-heavy median baseline on MAE. Rather than hide it, the repository retains the negative result and reframes the operational decision as ranking flights at risk of a **15+ minute arrival delay**.

The resulting classifier reached PR-AUC **0.291** against **0.215** prevalence on an untouched May 2026 test, with **1.58× lift** in the top-risk decile.

**Interview point:** a production-minded data scientist changes the decision formulation when the evidence says the first target is not useful.

### 3. I can distinguish current evidence from methodology demos

Current evidence includes official 2026 BTS flight data and the 14-Jun-2026 Inside Airbnb snapshot. Historical datasets such as KDD Cup 1999, UCI churn and Parkinson's are labelled as historical. Synthetic projects such as ConsultAI, Fraud/AML and Telecom are labelled as methodology demonstrations.

**Interview point:** dataset freshness and external validity are part of model quality.

### 4. I verify artifacts rather than trusting notebook state

VisionForge independently reloads its checkpoint and checks PyTorch, TorchScript and ONNX parity. Spark KDD verifies serialized `PipelineModel` predictions. PostgreSQL runs a real service container and retains two `EXPLAIN (ANALYZE, BUFFERS)` plans. The repository-wide integrity workflow compile-checks extensions and verifies retained evidence records.

**Interview point:** reproducibility is an engineering requirement, not a README claim.

## Safe CV / application metrics

These metrics are backed by retained evidence and can be used when relevant:

- **VisionForge:** 85.9% test accuracy; 85.8% macro-F1; selective accuracy 90.4% at 89.1% coverage.
- **AeroFlow v3:** PR-AUC 0.291 vs 0.215 prevalence; top-10% risk-bucket lift 1.58× on May 2026 BTS test data.
- **UK House Prices:** MAE £81,805; R² 0.604 on 216,564-sale untouched 2026 test; 90% interval coverage 91.6%.
- **NYC Airbnb 2026:** MAE $68.97 vs $121.90 median baseline on 45 unseen neighbourhoods; 43.4% MAE improvement.
- **Customer Churn:** PR-AUC 0.955; ROC-AUC 0.990; 94.9% recall at 73.2% precision.
- **Recommender v2:** Recall@10 0.501 vs 0.463 popularity baseline; NDCG@10 0.336 vs 0.293.
- **TensorFlow Energy:** MAE 43.51 vs 53.18 seasonal baseline; 18.2% improvement.
- **PySpark Clickstream:** PR-AUC 0.351 vs 0.155 prevalence; ROC-AUC 0.763.
- **London Air Quality R:** MAE 10.48; 90% interval coverage 88.2% on untouched 2026 evidence.

## Metrics that need scope attached

These results are valid only with their limitation stated:

- **Spark / standalone KDD:** historical KDD Cup 1999 methodology only; do not present as modern cyber-security performance.
- **ConsultAI:** deterministic synthetic educational cases; NPV values are not real-company forecasts.
- **Fraud/AML + Telecom:** synthetic methodology demonstrations, not bank/telecom production performance.
- **Clustering:** reproducible methodology demo; clusters are descriptive, not ground truth.
- **CineIntelligence harness:** deterministic fixture proves parsing/quarantine/index logic, not TMDB freshness or database-scale performance.
- **LLM evaluation harness:** deterministic fixture proves evaluator behaviour, not the quality of the restored LLM checkpoints.
- **Parkinson's:** historical, educational and non-clinical; the grouped result is deliberately negative.

## What remains intentionally unpromoted

The LLM Core/Alignment notebooks still need fresh checkpoint generations before model-quality claims can be promoted. The multi-modal health and X-ray projects still need strong patient/group independence, defensible provenance and non-clinical safety evidence. These are not gaps hidden from recruiters; they are explicit evidence boundaries.

## Portfolio principle

**Preserve good work. Fix material methodology problems. Execute cleanly. Retain evidence. Quote only what can be defended.**
