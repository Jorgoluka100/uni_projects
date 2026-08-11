# Production-Hardening Extensions

This directory contains focused upgrades to restored portfolio projects. The original notebooks remain visible so the underlying work is preserved; extensions are used when a project needs a material methodological or engineering correction rather than a cosmetic rewrite.

## AeroFlow v2 — 2026 BTS Flight Delay Intelligence

[`aeroflow_v2.py`](aeroflow_v2.py) is the first completion upgrade.

It replaces the original synthetic-only evaluation path with official 2026 U.S. Bureau of Transportation Statistics on-time-performance data and fixes the original preprocessing leakage by using a strictly temporal design:

- **Train:** January–March 2026
- **Validation / conformal calibration:** April 2026
- **Untouched test:** May 2026

The v2 pipeline adds:

- explicit source/data contract and schema checks;
- schedule-time-only features to prevent post-outcome leakage;
- train-only global-median and route/carrier baselines;
- CatBoost regression with April early stopping;
- split-conformal 90% prediction intervals;
- 15-minute operational delay precision/recall diagnostics;
- carrier-level error slices;
- decision-oriented planning-buffer output;
- model + metadata + slice artifact export;
- model reload smoke test and hard temporal-overlap assertions.

### Run

```bash
pip install pandas numpy scikit-learn catboost requests joblib
python extensions/aeroflow_v2.py
```

The monthly BTS ZIP files are downloaded directly from the official `transtats.bts.gov/PREZIP` directory. The code limits row counts by default to stay practical on a normal development machine; those limits are explicit in `RunConfig` and can be changed for a full-scale run.

### Evidence rule

The extension intentionally contains **no fabricated pre-computed performance claims**. After a clean end-to-end run, review the saved `aeroflow_metadata.json`, confirm the May 2026 test result and uncertainty coverage, and only then promote the measured result to the main README/CV.

## Why extensions instead of deleting old notebooks?

Because the goal is to preserve useful implementation work while fixing real weaknesses. We add a v2 when there is a material issue — leakage, stale/synthetic data, weak evaluation, missing production checks — and leave strong code alone when the only missing item is fresh execution evidence.
