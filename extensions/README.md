# Production-Hardening Extensions

The original notebooks stay visible on `main`. This directory adds **targeted completion layers** only where a restored project has a real methodological, evaluation, data-freshness or engineering gap.

No extension is allowed to manufacture a performance claim. A result becomes verified portfolio evidence only after the current notebook/script is executed cleanly and the retained evidence is reviewed.

## Completion map

| Extension | Project(s) strengthened | Material addition | Status |
|---|---|---|---|
| [`visionforge_verify_v2.py`](visionforge_verify_v2.py) | VisionForge | Fresh-process checkpoint reload, public test recomputation, 95% bootstrap CIs, abstention reproduction, TorchScript + ONNX parity | **Verifier complete; GPU rerun required** |
| [`visionforge_release_gate.py`](visionforge_release_gate.py) | VisionForge | Evidence gate across model card, metrics, corruption tests, exported models, hashes and independent verification | **Gate self-tested in CI** |
| [`consultai_verify_v2.py`](consultai_verify_v2.py) | ConsultAI | Independently reconstructs seeded Monte Carlo outcomes, exhaustive portfolio optimisation, budget frontier, stress tests and governance outputs | **Verifier self-tested in CI; clean notebook run required** |
| [`aeroflow_v2.py`](aeroflow_v2.py) | AeroFlow | Official 2026 BTS data, temporal split, leakage-safe schedule features, baselines, conformal intervals, operational policy, artifact reload | **Ready for fresh run** |
| [`airbnb_nyc_2026_v2.py`](airbnb_nyc_2026_v2.py) | NYC Airbnb Market Analysis | Current 14-Jun-2026 Inside Airbnb snapshot, data contract, market summary, neighbourhood-group holdout, baseline + model and source guardrails | **Current-data pipeline ready for run** |
| [`parkinsons_grouped_v2.py`](parkinsons_grouped_v2.py) | Parkinson's Progression ML | Complete-subject train/validation/test split, removal of `motor_UPDRS` shortcut, baseline, held-out metrics and subject-level bootstrap CI | **Patient-leakage fix ready for run** |
| [`healthcare_evidence_gate.py`](healthcare_evidence_gate.py) | Multi-Modal Health Analytics + Medical X-ray | Blocks promotion without patient/group separation, provenance, intended/non-intended use, held-out evidence, uncertainty/abstention and human oversight | **Gate self-tested in CI** |
| [`spark_kdd_classifiers_v2.py`](spark_kdd_classifiers_v2.py) | Logistic Regression PySpark + Naive Bayes PySpark | Deduplication, stable holdout, train-only Spark feature pipeline, PR-AUC/ROC-AUC, validation model choice, serialized PipelineModel reload | **Ready for fresh Spark run; historical benchmark only** |
| [`kdd_intrusion_v2.py`](kdd_intrusion_v2.py) | KDD Cup Analysis | Maintained loader, historical warning, baseline, imbalance-aware held-out metrics, attack-type errors and reload verification | **Ready for fresh run; historical benchmark only** |
| [`clustering_stability_v2.py`](clustering_stability_v2.py) | Clustering Models | Silhouette + Davies-Bouldin + Calinski-Harabasz, resampling ARI stability and interpretable cluster profiles | **Ready for fresh run** |
| [`pathfinding_benchmark_v2.py`](pathfinding_benchmark_v2.py) | Pathfinding | Seeded benchmark grids, BFS/Dijkstra/A*, path validation, optimality, expanded nodes, runtime and admissibility sanity checks | **Self-tested in CI** |
| [`aviation_postgres_v2.sql`](aviation_postgres_v2.sql) | Aviation Strategy PostgreSQL | Genuine PostgreSQL DDL/workload, one-million-row deterministic fact table, before/after `EXPLAIN (ANALYZE, BUFFERS)`, covering index and reconciliation | **Ready for PostgreSQL execution** |
| [`cine_nosql_v2.py`](cine_nosql_v2.py) | CineIntelligence NoSQL | Defensive parsing, malformed-row quarantine, explicit document schema, quality checks, indexed-query path and benchmark | **Engineering layer ready** |
| [`recommender_v2.py`](recommender_v2.py) | Hybrid DL Movie Recommender | Per-user temporal holdout, popularity baseline, latent ranking, Recall@K/NDCG@K and cold-start fallback | **Ready for fresh MovieLens run** |
| [`llm_eval_v2.py`](llm_eval_v2.py) | LLM Mastery Core + Alignment | Frozen prompt manifest, base-vs-aligned comparison, transparent automated checks and optional blinded human pairwise labels | **Evaluation layer ready; model rerun required** |
| [`telecom_churn_decision_v2.py`](telecom_churn_decision_v2.py) | Strategic Telecom Churn + Predictive SQL | Train-only encoding, validation model selection, validation-only review-capacity threshold, untouched test and SQL/Pandas grain reconciliation | **Synthetic methodology completion ready for run** |
| [`fraud_aml_v2.py`](fraud_aml_v2.py) | Financial Fraud / AML | Chronological design, past-only features, validation-only threshold, review-capacity/cost metrics, monthly slices and reload | **Methodology-complete synthetic demo** |

## VisionForge: evidence before promotion

After the notebook has been restarted and run end-to-end on a GPU, run:

```bash
pip install torch torchvision datasets scikit-learn onnxruntime numpy
python extensions/visionforge_verify_v2.py --artifact-dir visionforge_artifacts
python extensions/visionforge_release_gate.py --artifact-dir visionforge_artifacts
```

The verifier recomputes the real test evidence from the saved checkpoint in a fresh process. The release gate then requires that the retained claims reproduce, TorchScript and ONNX agree with PyTorch, bootstrap intervals are present, robustness/selective-prediction evidence exists and the model card/manifest are complete.

## ConsultAI: deterministic decision verification

ConsultAI deliberately uses synthetic organisational cases, so its strongest evidence is not a commercial accuracy number. Its independent verifier rebuilds the six source cases, the seeded 10,000-trial Monte Carlo process, the exhaustive constrained portfolio search, the full budget frontier and stress scenarios, then checks the exported decision/governance files against those independently reproduced outputs.

```bash
python extensions/consultai_verify_v2.py --artifact-dir consultai_application_artifacts
```

This proves reproducibility of the **decision engine** while keeping the synthetic-data limitation explicit.

## Current NYC Airbnb refresh

The restored Airbnb notebook now has a current-data route using Inside Airbnb's New York City snapshot dated **14 June 2026**. The script caches the source once, separates descriptive market summaries from predictive evaluation, removes invalid prices, documents extreme-price handling and holds out complete neighbourhoods rather than randomly mixing every neighbourhood across train/test.

```bash
pip install pandas numpy scikit-learn
python extensions/airbnb_nyc_2026_v2.py
```

Availability is never described as confirmed bookings and anonymised listing coordinates are treated as approximate.

## Patient/group-safe healthcare work

Parkinson's now has a concrete subject-grouped modelling path:

```bash
pip install pandas numpy scikit-learn
python extensions/parkinsons_grouped_v2.py
```

For the multi-modal and X-ray projects, promotion requires a machine-readable evidence record that passes:

```bash
python extensions/healthcare_evidence_gate.py evidence.json
```

A gate pass means the portfolio evidence contract is present; it **does not establish clinical validity or safety**.

## Distributed classification

The two restored Spark classifier notebooks are strengthened together because both use KDD Cup 1999. The new benchmark removes exact duplicates before splitting, fits categorical encoding/scaling inside the training Spark Pipeline, chooses between Logistic Regression and Naive Bayes using validation PR-AUC, opens the test split only for final reporting and proves saved `PipelineModel` reload parity.

```bash
pip install pyspark scikit-learn pandas numpy
python extensions/spark_kdd_classifiers_v2.py
```

KDD Cup 1999 remains explicitly labelled as historical methodology evidence, not modern cyber-security performance.

## SQL, clustering and algorithms

Aviation now has a genuine PostgreSQL workload instead of relying on the phrase "PostgreSQL-compatible DuckDB":

```bash
psql "$DATABASE_URL" -f extensions/aviation_postgres_v2.sql
```

The clustering extension chooses `k` from multiple validation views plus bootstrap stability rather than one silhouette score. Pathfinding now measures optimality, expanded nodes and runtime over seeded solvable grids instead of showing algorithms without a benchmark.

## Decision-focused churn, recommendation, fraud and LLM work

- Telecom churn now chooses a model on validation PR-AUC and selects a risk threshold from stated review capacity before opening test data.
- The recommender evaluates ranking quality with Recall@K/NDCG@K instead of relying on rating error alone.
- Fraud/AML uses chronological splits and operational review/cost metrics, while staying explicitly synthetic.
- LLM work uses a frozen base-vs-aligned evaluation harness rather than subjective hand-picked generations.

## Repository rule

The strategy is now **preserve + harden + verify**:

1. Preserve useful original notebooks.
2. Fix material gaps with targeted completion layers.
3. Run the current version cleanly.
4. Retain test/evaluation artifacts.
5. Promote only claims that can be reproduced and defended in an interview.

Good code stays. Unverified claims do not.
