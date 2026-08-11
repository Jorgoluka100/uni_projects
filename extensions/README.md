# Production-Hardening Extensions

This directory contains **material completion upgrades** for restored portfolio projects. The original notebooks remain visible so useful implementation work is preserved. A v2 extension is added only when it fixes a real methodological, evaluation or engineering weakness.

The extensions do **not** manufacture new performance claims. A project is promoted to the verified flagship tier only after the current version has completed a clean end-to-end run and its retained evidence has been reviewed.

## Completion upgrades

| Extension | Restored project strengthened | What the v2 adds | Evidence status |
|---|---|---|---|
| [`aeroflow_v2.py`](aeroflow_v2.py) | AeroFlow AI Engine | Official 2026 BTS data, strict temporal split, leakage-safe schedule-time features, baselines, CatBoost, conformal intervals, operational delay policy, slice analysis, saved artifacts + reload test | **Ready for fresh run** |
| [`fraud_aml_v2.py`](fraud_aml_v2.py) | Financial Fraud / AML Detection | Time-ordered transaction stream, past-only behavioural features, chronological split, train-only preprocessing, PR-AUC/ROC-AUC, validation-only cost threshold, review-capacity metrics, monthly slices, reload test | **Methodology-complete; synthetic demo until real institution data exists** |
| [`recommender_v2.py`](recommender_v2.py) | Hybrid DL Movie Recommender | Latest-positive holdout per user, popularity baseline, latent-factor recommender, sampled-candidate Recall@K/NDCG@K, cold-start fallback and reproducible artifacts | **Ready for fresh MovieLens run** |
| [`llm_eval_v2.py`](llm_eval_v2.py) | LLM Mastery — Core + Alignment | Frozen prompt manifest, deterministic format/content/refusal checks, base-vs-aligned comparison, optional blinded human pairwise results, detail + summary artifacts | **Evaluation layer ready; model generations still need fresh run** |
| [`kdd_intrusion_v2.py`](kdd_intrusion_v2.py) | KDD Cup Analysis | Maintained sklearn loader, explicit historical warning, baseline, held-out imbalance-aware metrics, attack-type error table and reload verification | **Ready for fresh run; historical benchmark only** |
| [`cine_nosql_v2.py`](cine_nosql_v2.py) | CineIntelligence NoSQL | Defensive JSON-ish parsing, row quarantine, explicit document schema, data-quality checks, inverted-index query path, repeatable benchmark and NDJSON export | **Engineering layer ready; source file/licence must be supplied and documented** |

## AeroFlow v2 — 2026 BTS Flight Delay Intelligence

The original notebook's synthetic-data and preprocessing-leakage weaknesses are addressed by a separate current-data pipeline.

**Temporal design**
- Train: January–March 2026
- Validation / conformal calibration: April 2026
- Untouched test: May 2026

```bash
pip install pandas numpy scikit-learn catboost requests joblib
python extensions/aeroflow_v2.py
```

The code uses schedule-time-only features, train-only baselines, CatBoost, split-conformal 90% intervals, 15-minute operational diagnostics, carrier slices, artifact export and a reload smoke test.

## Fraud / AML v2

This remains deliberately **synthetic** by default. That is a feature of the evidence policy, not something hidden: synthetic labels can test the pipeline and decision design, but cannot prove real financial-crime performance.

```bash
pip install pandas numpy scikit-learn joblib
python extensions/fraud_aml_v2.py
```

The important upgrade is the evaluation design: chronological train/validation/test, validation-only threshold selection, review-capacity and false-negative/false-positive costs, untouched test metrics, monthly slices and a serialized decision policy.

## Recommender v2

The old recommender demonstrated breadth, but rating RMSE alone is not enough for a recommendation decision. The v2 evaluates whether the held-out latest positive item is actually ranked near the top.

```bash
pip install pandas numpy scipy scikit-learn requests
python extensions/recommender_v2.py
```

It downloads GroupLens `ml-latest-small`, creates a per-user temporal holdout, compares a popularity baseline against latent SVD, and reports Recall@K / NDCG@K over a reproducible sampled candidate set.

## LLM evaluation v2

The LLM notebooks contain architecture/alignment work, but subjective samples should not be the evaluation. This harness scores the same frozen prompt manifest before and after alignment.

```bash
python extensions/llm_eval_v2.py \
  --base base_generations.jsonl \
  --aligned aligned_generations.jsonl \
  --human-csv blinded_pairwise_labels.csv
```

The rule-based checks are intentionally transparent and limited. Optional human pairwise labels remain separate so automated proxies are not presented as a complete helpfulness/safety score.

## KDD intrusion v2

This project is kept because it demonstrates large-dataset / intrusion-classification methods, but **KDD Cup 1999 is historical** and must never be marketed as a modern cyber-security benchmark.

```bash
pip install pandas numpy scikit-learn joblib
python extensions/kdd_intrusion_v2.py
```

The v2 adds a prior baseline, imbalance-aware metrics, attack-type error diagnostics, a held-out test, artifact saving and reload verification.

## CineIntelligence NoSQL v2

The old notebook's quote-replacement parsing was brittle. The v2 uses `json.loads` with `ast.literal_eval` fallback, quarantines malformed rows, defines a document contract, exports NDJSON and demonstrates an indexed query path.

```bash
python extensions/cine_nosql_v2.py /path/to/tmdb_5000_movies.csv
```

No Kaggle credentials or copied dataset are embedded. Source/licence/freshness remain part of the project evidence.

## Why extensions instead of deleting or rewriting everything?

Because **good code should stay**. The portfolio now follows three rules:

1. Preserve useful original notebooks.
2. Add a v2 only when it fixes a material weakness.
3. Do not promote a metric until the current code has been rerun and the evidence is retained.

That gives the repository breadth without pretending every older notebook is equally verified.
