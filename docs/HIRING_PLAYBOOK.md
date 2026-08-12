# Hiring Playbook — How to Read This Portfolio

This file maps the strongest verified evidence to common junior/graduate roles and turns the repository into an interview tool rather than a notebook archive.

## 90-second recruiter route

If you only open five things:

1. **VisionForge** — strongest ML-engineering / computer-vision example: held-out evaluation, calibration, abstention, explainability, export and independent parity checks.
2. **AeroFlow v3** — strongest current-data modelling story: official 2026 BTS data, temporal validation, failed-regression diagnosis, problem reframing and operational ranking lift.
3. **SQL Sales & Customer Analysis** — strongest analytics/SQL example: relational grain control, reconciliation and trustworthy commercial KPIs.
4. **ExperimentLab** — strongest statistics/decision-science example: randomized experiment, CUPED, bootstrap uncertainty, guardrail non-inferiority and power/MDE.
5. **ModelWatch** — strongest MLOps example: PSI/KS drift, performance degradation, calibration, subgroup summaries, model reload and explicit retraining policy.

## Best evidence by target role

| Target role | Lead with | Supporting evidence |
|---|---|---|
| **Junior / Graduate Data Scientist** | AeroFlow, UK House Prices, ExperimentLab | Customer Churn, NYC Airbnb 2026, Recommender, Parkinson's negative grouped result |
| **Junior Machine Learning Engineer** | VisionForge, ModelWatch, Recommender v2 | TensorFlow Energy, Spark KDD, model reload/export and CI evidence |
| **Applied AI / AI Engineer** | VisionForge, CareerLens AI, ConsultAI | LLM evaluation harness, AeroFlow, FastAPI/export/reproducibility evidence |
| **Data Analyst / Analytics** | SQL Sales, NYC Airbnb 2026, ExperimentLab | Clickstream, London Air Quality R, bootcamp portfolio |
| **Data / ML platform-oriented** | PySpark Clickstream, Spark KDD, PostgreSQL aviation | ModelWatch, SQL Sales, CI/evidence gates |

## Interview stories worth learning

### 1. I found leakage instead of celebrating a high score

The recommender clean run exposed future interactions in evaluated-user histories. The fix truncated each evaluated user's history before the held-out target, removing **1,756 future interactions** before rerunning ranking metrics.

The Parkinson's project exposed a more serious issue: repeated rows from the same subject had made row-level evaluation too optimistic. Complete-subject holdout produced RF MAE **10.70** versus a median baseline of **8.13** and R² **-1.09**. The weaker result is retained because it is the honest result.

**Interview point:** validation design matters more than protecting a metric.

### 2. A failed baseline changed the problem formulation

AeroFlow's first 2026 schedule-time regression passed engineering checks but failed to beat the zero-heavy median baseline on MAE. Rather than hide it, the project reframed the operational decision as ranking flights at risk of a **15+ minute arrival delay**.

The classifier reached PR-AUC **0.291** against **0.215** prevalence on untouched May 2026 data, with **1.58× lift** in the top-risk decile.

**Interview point:** a production-minded data scientist changes the target when evidence says the original target is not useful.

### 3. I can design experiments, not just predictive models

ExperimentLab simulates a randomized 20,000-row experiment with a known +2.5 treatment effect. CUPED estimates **2.773** with a 95% CI of **2.435–3.112**, reducing outcome variance by **50.7%**, while a pre-declared guardrail and MDE calculation constrain the ship decision.

**Interview point:** prediction is only one part of data science; causal decision quality and uncertainty matter too.

### 4. I think about what happens after deployment

ModelWatch evaluates production-like batches with PSI and KS drift, ROC-AUC/PR-AUC, Brier score, expected calibration error and subgroup summaries. The stable batch remains green; deliberately shifted batches trigger red investigation states, and saved-model reload parity is checked.

**Interview point:** an ML model is not finished when `.fit()` returns.

### 5. I distinguish current evidence from methodology demos

Current evidence includes official 2026 BTS flights and the 14-Jun-2026 Inside Airbnb snapshot. Historical KDD, UCI churn and Parkinson's data are labelled historical. ConsultAI, Fraud/AML, Telecom, CareerLens's built-in corpus, ExperimentLab and ModelWatch are clearly labelled methodology/simulation evidence where appropriate.

**Interview point:** freshness and external validity are part of model quality.

## Safe CV / application metrics

Use these only when relevant to the role:

- **VisionForge:** 85.9% test accuracy; 85.8% macro-F1; selective accuracy 90.4% at 89.1% coverage.
- **AeroFlow v3:** PR-AUC 0.291 vs 0.215 prevalence; top-risk decile lift 1.58× on untouched May 2026 BTS data.
- **UK House Prices:** MAE £81,805; R² 0.604 on 216,564-sale untouched 2026 test; 90% interval coverage 91.6%.
- **NYC Airbnb 2026:** MAE $68.97 vs $121.90 median baseline on 45 unseen neighbourhoods; 43.4% improvement.
- **Customer Churn:** PR-AUC 0.955; ROC-AUC 0.990; 94.9% recall at 73.2% precision.
- **Recommender v2:** Recall@10 0.501 vs 0.463 popularity baseline; NDCG@10 0.336 vs 0.293.
- **TensorFlow Energy:** MAE 43.51 vs 53.18 seasonal baseline; 18.2% improvement.
- **PySpark Clickstream:** PR-AUC 0.351 vs 0.155 prevalence; ROC-AUC 0.763.
- **ExperimentLab:** 50.7% CUPED variance reduction on a deterministic synthetic randomized experiment; attach the synthetic-methodology scope.
- **ModelWatch:** stable max PSI 0.003; deliberate feature/concept shifts 0.264/0.466 trigger red investigation states; attach the simulation scope.

## Metrics that need scope attached

- **CareerLens AI:** deterministic synthetic retrieval benchmark; not real hiring-market accuracy.
- **ExperimentLab:** synthetic randomized-experiment methodology; not real product uplift.
- **ModelWatch:** monitoring simulation; not a live production deployment.
- **Spark / standalone KDD:** KDD Cup 1999 historical methodology only; not modern cybersecurity performance.
- **ConsultAI:** deterministic synthetic educational cases; NPV is not a real-company forecast.
- **Fraud/AML + Telecom:** synthetic methodology demonstrations.
- **Clustering:** descriptive methodology demo, not ground truth.
- **LLM evaluation harness:** proves evaluator behaviour, not restored checkpoint quality.
- **Parkinson's:** historical, educational and non-clinical; grouped result is deliberately negative.

## Portfolio principle

**Preserve good work. Fix material methodology problems. Execute cleanly. Retain evidence. Quote only what can be defended.**
