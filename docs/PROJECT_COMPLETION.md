# Advanced Project Completion Plan

All useful restored notebooks remain on `main`. The portfolio uses a **preserve + harden + verify** approach: keep the original implementation, add a targeted completion layer where something material is missing, then promote only after a clean rerun and evidence review.

## Promotion rule

A project moves into the verified flagship tier only after its current path has been executed cleanly and the retained evidence supports the claims. Promotion requires decision framing, provenance/freshness, data quality, leakage control, a baseline, untouched evaluation, uncertainty/error analysis where appropriate, reproducibility, engineering evidence and honest limitations.

## Current completion map

| Project | Current status | Completion path |
|---|---|---|
| ConsultAI AI Opportunity Engine | **Near-flagship + deterministic verifier** | Original Monte Carlo/optimisation/app work is preserved. [`consultai_verify_v2.py`](../extensions/consultai_verify_v2.py) independently rebuilds the seeded simulations, exhaustive portfolio optimum, budget frontier, stress tests and governance outputs. Clean Colab run + verifier pass remain. |
| VisionForge PyTorch Visual Inspection | **Near-flagship + independent verifier + release gate** | [`visionforge_verify_v2.py`](../extensions/visionforge_verify_v2.py) recomputes held-out evidence from the saved checkpoint and checks TorchScript/ONNX parity; [`visionforge_release_gate.py`](../extensions/visionforge_release_gate.py) blocks promotion without the complete evidence bundle. Clean GPU run remains. |
| Advanced Multi-Modal Health Analytics | **Safety gate added** | [`healthcare_evidence_gate.py`](../extensions/healthcare_evidence_gate.py) requires patient/group separation, provenance, non-intended use, held-out metrics, uncertainty/abstention, limitations and human oversight before promotion. A clean project-specific rerun is still required. |
| AeroFlow AI Engine | **v2 ready** | [`aeroflow_v2.py`](../extensions/aeroflow_v2.py) uses official 2026 BTS data, temporal splits, leakage-safe schedule features, baselines, conformal intervals, operational policy and artifact reload checks. |
| Aviation Strategy PostgreSQL Optimisation | **PostgreSQL v2 ready** | [`aviation_postgres_v2.sql`](../extensions/aviation_postgres_v2.sql) replaces the ambiguous DuckDB/PostgreSQL claim with genuine PostgreSQL DDL, a deterministic 1M-row workload, before/after `EXPLAIN (ANALYZE, BUFFERS)`, a query-aligned index and reconciliation. |
| CineIntelligence NoSQL Data Engineering | **v2 ready** | [`cine_nosql_v2.py`](../extensions/cine_nosql_v2.py) adds defensive parsing, quarantine, document schema, quality checks, indexed-query path and benchmark. Source/licence still need to be supplied on the fresh run. |
| Clustering Models | **Stability v2 ready** | [`clustering_stability_v2.py`](../extensions/clustering_stability_v2.py) adds multiple internal indices, resampling ARI stability and interpretable profiles instead of selecting clusters from one score. |
| KDD Cup Analysis | **v2 ready** | [`kdd_intrusion_v2.py`](../extensions/kdd_intrusion_v2.py) adds maintained loading, explicit historical warning, baseline, imbalance-aware held-out metrics, attack-type errors and reload verification. |
| LLM Mastery — Alignment | **Evaluation v2 ready** | [`llm_eval_v2.py`](../extensions/llm_eval_v2.py) freezes prompts, compares base vs aligned outputs and supports blinded human pairwise labels. Fresh generations are still required. |
| LLM Mastery — Core | **Evaluation v2 ready** | Same frozen evaluation harness; retain current checkpoint/training provenance and held-out loss/perplexity from a fresh model run. |
| Logistic Regression with PySpark | **Spark v2 ready** | [`spark_kdd_classifiers_v2.py`](../extensions/spark_kdd_classifiers_v2.py) removes exact duplicates, fits Spark preprocessing on train only, evaluates PR-AUC/ROC-AUC on holdouts and verifies serialized PipelineModel predictions. |
| Hybrid DL Movie Recommender | **Ranking v2 ready** | [`recommender_v2.py`](../extensions/recommender_v2.py) uses per-user temporal holdout, popularity baseline, latent ranking, Recall@K/NDCG@K and cold-start fallback. |
| NYC Airbnb Market Analysis | **Current-data v2 ready** | [`airbnb_nyc_2026_v2.py`](../extensions/airbnb_nyc_2026_v2.py) uses the 14-Jun-2026 Inside Airbnb NYC snapshot, separates descriptive from predictive claims, caches source data and holds out complete neighbourhoods. |
| Naive Bayes with PySpark | **Spark v2 ready** | The same [`spark_kdd_classifiers_v2.py`](../extensions/spark_kdd_classifiers_v2.py) provides a clean Naive Bayes benchmark alongside Logistic Regression and avoids relying on the old stored interruption. |
| Parkinson's Progression ML | **Patient-grouped v2 ready** | [`parkinsons_grouped_v2.py`](../extensions/parkinsons_grouped_v2.py) fixes row-split leakage using complete-subject holdouts, removes the `motor_UPDRS` shortcut and adds subject-level bootstrap uncertainty. |
| Pathfinding | **Benchmark v2 + CI self-test** | [`pathfinding_benchmark_v2.py`](../extensions/pathfinding_benchmark_v2.py) adds seeded grid families, BFS/Dijkstra/A*, path validity, optimality, expanded nodes, runtime and Manhattan-admissibility checks. |
| PyTorch Medical AI X-ray Diagnosis | **Safety gate added** | [`healthcare_evidence_gate.py`](../extensions/healthcare_evidence_gate.py) prevents promotion without patient/group independence, provenance, held-out metrics, uncertainty/abstention and explicit non-clinical guardrails. Project-specific fresh evidence remains required. |
| Strategic Telecom Churn + Predictive SQL | **Decision v2 ready** | [`telecom_churn_decision_v2.py`](../extensions/telecom_churn_decision_v2.py) adds train-only preprocessing, validation model choice, validation-only review-capacity threshold, untouched test and SQL/Pandas grain reconciliation. Synthetic status remains explicit. |
| Financial Fraud / AML Detection | **Decision v2 ready** | [`fraud_aml_v2.py`](../extensions/fraud_aml_v2.py) uses chronological design, past-only features, validation-only threshold, review-capacity/cost metrics, monthly slices and reload tests. It remains a synthetic methodology demo unless defensible real data is introduced. |

## What is now complete vs what still requires compute

**Engineering/methodology completion paths now exist for every restored advanced project.** The repository is no longer missing the evaluation design for those projects.

What is deliberately not faked is execution evidence. Heavy GPU, Spark, external-data and database projects must still be run in the appropriate runtime, and only those fresh outputs can be promoted into the verified flagship tier.

## Execution priority

1. VisionForge: clean Colab GPU run → independent verifier → release gate.
2. ConsultAI: clean Colab run → deterministic verifier.
3. AeroFlow: official 2026 BTS run.
4. NYC Airbnb: current 14-Jun-2026 data run.
5. Recommender + Fraud/AML + Telecom decision pipelines.
6. Spark KDD classifier benchmark + KDD v2.
7. Parkinson's subject-grouped evaluation.
8. PostgreSQL aviation benchmark and retain before/after plans.
9. Clustering + Pathfinding benchmark evidence.
10. LLM Core/Alignment fresh generations and evaluation.
11. Healthcare projects only after their patient/group and evidence contracts are fully satisfied.

## Evidence discipline

Do not copy an old notebook metric into the README or CV merely because it exists. A metric becomes portfolio evidence only after the **current** completion path has been rerun and the result is retained without errors.

The rule is simple: **keep good code, add what is genuinely missing, and be exact about what has actually been verified.**
