# Production-Hardening Extensions

The original notebooks stay visible on `main`. This directory contains **targeted completion layers** added only when an original project had a real methodological, evaluation, data-freshness or engineering gap.

The rule is **preserve + harden + verify**. No extension is allowed to manufacture a result. A metric becomes portfolio evidence only after a clean current run and retained verification.

## Current evidence map

| Extension | Project strengthened | What it adds | Current status |
|---|---|---|---|
| [`visionforge_verify_v2.py`](visionforge_verify_v2.py) + [`visionforge_release_gate.py`](visionforge_release_gate.py) | VisionForge | Fresh-process checkpoint recomputation, bootstrap CIs, abstention reproduction, TorchScript/ONNX parity and final evidence gate | **VERIFIED** — [evidence](../verified/visionforge/verification_metrics.json) |
| [`consultai_verify_v2.py`](consultai_verify_v2.py) | ConsultAI | Independent reconstruction of seeded Monte Carlo simulation, exhaustive portfolio optimisation, frontier, stress tests and governance outputs | **VERIFIED** — [evidence](../verified/consultai/consultai_verification.json) |
| [`aeroflow_v2.py`](aeroflow_v2.py) | AeroFlow regression | Official 2026 BTS data, temporal split, baselines, conformal intervals and reload | **VERIFIED NEGATIVE RESULT** — schedule-only delay-minute regression did not beat the global-median MAE baseline; [evidence](../verified/aeroflow/verification.json) |
| [`aeroflow_delay_risk_v3.py`](aeroflow_delay_risk_v3.py) | AeroFlow decision reframing | Reframes zero-inflated delay minutes into ≥15-minute risk classification with April-only threshold selection and untouched May test | **VERIFIED** — [evidence](../verified/aeroflow_delay_risk/verification.json) |
| [`airbnb_nyc_2026_v2.py`](airbnb_nyc_2026_v2.py) | NYC Airbnb | 14-Jun-2026 Inside Airbnb snapshot, grouped unseen-neighbourhood holdout and median baseline | **VERIFIED** — [evidence](../verified/airbnb_nyc_2026/verification.json) |
| [`recommender_v2.py`](recommender_v2.py) + [`recommender_release_gate.py`](recommender_release_gate.py) | Movie Recommender | Past-only per-user holdout, popularity baseline, latent ranking, Recall@K/NDCG@K and evidence gate | **VERIFIED** — clean run exposed and fixed a temporal leakage edge case; [evidence](../verified/recommender/verification.json) |
| [`fraud_aml_v2.py`](fraud_aml_v2.py) + [`fraud_aml_release_gate.py`](fraud_aml_release_gate.py) | Fraud / AML | Chronological synthetic stream, past-only behaviour, validation-only threshold, review/cost metrics and reload | **VERIFIED METHODOLOGY** — synthetic only; [evidence](../verified/fraud_aml/verification.json) |
| [`telecom_churn_decision_v2.py`](telecom_churn_decision_v2.py) | Telecom churn + SQL | Train-only encoding, validation model/threshold selection, untouched test and SQL↔Pandas reconciliation | **VERIFIED METHODOLOGY** — synthetic only; [evidence](../verified/telecom_churn/verification.json) |
| [`parkinsons_grouped_v2.py`](parkinsons_grouped_v2.py) | Parkinson's progression | Complete-subject holdout, removal of `motor_UPDRS` shortcut and subject-bootstrap uncertainty | **VERIFIED NEGATIVE RESULT** — stricter grouped test underperformed the median baseline; [evidence](../verified/parkinsons_grouped/verification.json) |
| [`clustering_stability_v2.py`](clustering_stability_v2.py) | Clustering | Multiple internal indices, resampling ARI stability and interpretable profiles | **VERIFIED METHODOLOGY** — [evidence](../verified/clustering/verification.json) |
| [`pathfinding_benchmark_v2.py`](pathfinding_benchmark_v2.py) | Pathfinding | Seeded BFS/Dijkstra/A* benchmark, route validation, optimality and expanded-node evidence | **VERIFIED** — [evidence](../verified/pathfinding/verification.json) |
| [`aviation_postgres_v2.sql`](aviation_postgres_v2.sql) | Aviation PostgreSQL | Genuine PostgreSQL 17 workload, 1M fact rows, before/after `EXPLAIN (ANALYZE, BUFFERS)`, index and reconciliation | **VERIFIED ENGINEERING** — [evidence](../verified/aviation_postgres/verification.json) |
| [`spark_kdd_classifiers_v2.py`](spark_kdd_classifiers_v2.py) | Logistic Regression + Naive Bayes PySpark | Explicit numeric/categorical schema, deduplication, train-only Spark pipeline, PR-AUC/ROC-AUC, validation selection and PipelineModel reload | **VERIFIED HISTORICAL BENCHMARK** — [evidence](../verified/spark_kdd/verification.json) |
| [`healthcare_evidence_gate.py`](healthcare_evidence_gate.py) | Multi-Modal Health + Medical X-ray | Blocks promotion without patient/group separation, provenance, held-out evidence, uncertainty/abstention and human oversight | **GATE VERIFIED; project evidence still required** |
| [`cine_nosql_v2.py`](cine_nosql_v2.py) | CineIntelligence NoSQL | Defensive parsing, malformed-row quarantine, document contract and indexed-query benchmark | **ENGINEERING PATH READY; source/licence evidence still required** |
| [`kdd_intrusion_v2.py`](kdd_intrusion_v2.py) | KDD Cup Analysis | Maintained loader, historical warning, baseline, held-out diagnostics and reload | **READY; separate historical experiment not promoted** |
| [`llm_eval_v2.py`](llm_eval_v2.py) | LLM Core + Alignment | Frozen prompts, base-vs-aligned comparison and optional blinded human pairwise evaluation | **EVALUATION READY; fresh model generations still required** |

## What the clean runs discovered

The verification work changed projects rather than simply stamping them green:

- **Recommender:** the first clean run found that later low-rated interactions could remain after a user's held-out latest positive. The split was corrected to truncate evaluated-user history strictly before the target.
- **Spark KDD:** the first clean run found the maintained loader exposed numeric fields as object dtype. The schema contract was corrected so only the documented three categoricals remain strings and every other predictor is numeric.
- **AeroFlow:** the regression pipeline was technically sound but failed the global-median MAE baseline. That result was retained, and the operational problem was reframed to delay-risk ranking/classification.
- **Parkinson's:** correcting subject leakage made performance materially worse, which is retained as evidence that the original row split overstated generalisation.

Those outcomes are the point of the evidence-first process.

## Evidence policy

1. Keep useful original notebooks.
2. Fix only material gaps.
3. Execute the current version cleanly.
4. Retain machine-readable evidence and relevant artifacts.
5. Keep negative results when they change the modelling decision.
6. Label historical and synthetic work clearly.
7. Promote only claims that can be defended in a technical interview.
