# Advanced Project Completion Plan

The restored notebooks stay on `main` because they contain useful code and demonstrate breadth. They are not all automatically labelled as verified. The portfolio now uses a **preserve + harden + verify** approach rather than deleting good work.

## Promotion rule

A project moves into the verified flagship tier only after a clean **restart / run all** (or equivalent script run) and review of the retained evidence.

Promotion requires:

- clear decision/user framing;
- source provenance, licence/usage constraints and freshness;
- data-quality and leakage checks;
- a simple baseline;
- untouched evaluation where the task supports it;
- uncertainty, calibration, error analysis, stress testing or review policy where relevant;
- reproducible configuration;
- saved artifacts and reload/smoke checks where appropriate;
- honest limitations and next-production-step discussion;
- no stored notebook errors in the promoted path.

## Current completion map

| Project | Current status | Material upgrade |
|---|---|---|
| ConsultAI AI Opportunity Engine | **Strong advanced project** | Original notebook already contains substantial decision framing, Monte Carlo analysis, constrained portfolio selection, governance/testing and app engineering. Priority is clean execution evidence, not rewrite-for-rewrite's-sake. |
| VisionForge PyTorch Visual Inspection | **Near-flagship + release gate** | Original notebook already contains real data, custom + transfer models, validation-only calibration, abstention, Grad-CAM, corruption stress tests, TorchScript/ONNX exports, latency, model-card and release-manifest logic. [`extensions/visionforge_release_gate.py`](../extensions/visionforge_release_gate.py) now blocks promotion when required evidence, hashes, benchmark rows, class metrics, robustness coverage, selective-policy columns or safety documentation are missing. Its self-test runs in CI. Only a clean GPU restart/run-all with retained real artifacts remains before verified promotion. |
| Advanced Multi-Modal Health Analytics | Advanced | Tighten provenance, patient/group leakage controls, safety framing, untouched evaluation and intended/non-intended use before any health-performance claim. |
| AeroFlow AI Engine | **v2 added** | [`extensions/aeroflow_v2.py`](../extensions/aeroflow_v2.py): official 2026 BTS data, temporal split, leakage-safe schedule-time features, baselines, conformal intervals, operational policy, artifact/reload checks. |
| Aviation Strategy PostgreSQL Optimisation | Advanced | Make schema setup reproducible and retain `EXPLAIN (ANALYZE, BUFFERS)` before/after evidence with an index/query-design rationale. |
| CineIntelligence NoSQL Data Engineering | **v2 added** | [`extensions/cine_nosql_v2.py`](../extensions/cine_nosql_v2.py): defensive parsing, quarantine, explicit document model, quality checks, index-style query path and benchmark. |
| Clustering Models | Advanced | Add decision framing, stability/resampling analysis, multiple validation indices and interpretable segment profiles. |
| KDD Cup Analysis | **v2 added** | [`extensions/kdd_intrusion_v2.py`](../extensions/kdd_intrusion_v2.py): maintained loader, historical warning, baseline, held-out imbalance-aware metrics, attack-type errors and reload test. |
| LLM Mastery — Alignment | **evaluation v2 added** | [`extensions/llm_eval_v2.py`](../extensions/llm_eval_v2.py): frozen prompt manifest, base-vs-aligned comparison, transparent checks and optional blinded human pairwise labels. Clean environment/model rerun still required. |
| LLM Mastery — Core | **evaluation v2 added** | Use the same harness for fixed held-out generations; retain training config/checkpoint provenance and held-out loss/perplexity from a clean model run. |
| Logistic Regression with PySpark | Advanced | Add clear source contract, leakage rationale, imbalance-aware test metrics and serialized Spark pipeline reload check. |
| Hybrid DL Movie Recommender | **v2 added** | [`extensions/recommender_v2.py`](../extensions/recommender_v2.py): per-user temporal holdout, popularity baseline, latent-factor ranking, Recall@K/NDCG@K and cold-start fallback. |
| NYC Airbnb Market Analysis | Advanced | Refresh data if feasible; separate descriptive vs predictive claims; add geography/time-aware validation and decision-focused conclusions. |
| Naive Bayes with PySpark | Advanced with stored-error warning | Remove retained `KeyboardInterrupt` via clean rerun; add benchmark, imbalance-aware metrics, feature-pipeline audit and saved Spark pipeline verification. |
| Parkinson's Progression ML | Advanced | Strengthen patient/group leakage protection, non-clinical framing, uncertainty and patient-level untouched evaluation. |
| Pathfinding | Advanced | Add benchmark grid suite, runtime/expanded-node comparison, optimality/admissibility explanation and route diagnostics. |
| PyTorch Medical AI X-ray Diagnosis | Advanced | Make patient/group split and provenance airtight; add uncertainty/abstention, class errors, Grad-CAM limitations and strong non-clinical-use guardrails. |
| Strategic Telecom Churn + Predictive SQL | Advanced | Consolidate data grain, leakage-safe features, capacity-aware thresholding and SQL/model reconciliation. |
| Financial Fraud / AML Detection | **v2 added** | [`extensions/fraud_aml_v2.py`](../extensions/fraud_aml_v2.py): chronological design, past-only behavioural features, train-only pipeline, validation-only threshold, review-capacity/cost metrics, monthly slices and reload test. Synthetic methodology demo remains clearly labelled. |

## Priority from here

1. **Execute VisionForge on a Colab GPU and run the release gate** — the modelling, engineering and promotion checks are now present; real retained run evidence is the final blocker.
2. **Run and verify ConsultAI** — strong applied-AI / decision-engineering signal.
3. **Run Fraud/AML v2** — keep it as a methodology demo unless a defensible real dataset is introduced.
4. **Run Recommender v2** — adds ranking/evaluation signal not covered by the verified seven.
5. **Run LLM Core + Alignment and feed frozen outputs into the evaluation v2 harness.**
6. **Run AeroFlow v2** on official 2026 BTS data and retain May 2026 evidence.
7. Improve healthcare projects only with especially strict provenance/leakage/safety standards.

## Evidence discipline

Do not copy a metric into the README or CV because it appears in an old notebook. A metric becomes portfolio evidence only after the **current** notebook/script version has been rerun and the result is retained without errors.

The goal is no longer “delete anything imperfect.” It is: **keep useful code, make the strongest projects complete, and be exact about what has actually been verified.**
