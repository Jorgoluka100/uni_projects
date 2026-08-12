# ModelWatch — Drift & Retraining Monitor

**Decision:** when should a deployed classifier trigger investigation or retraining rather than continuing silently under distribution shift?

ModelWatch trains a reference classifier and evaluates sequential production-like batches with **PSI + KS feature drift**, ROC-AUC / PR-AUC degradation, Brier score, expected calibration error, subgroup summaries and an explicit green/amber/red policy. Saved-model reload parity is also checked.

```bash
pip install numpy pandas scipy scikit-learn joblib
python projects/model_watch/run.py --self-test
python projects/model_watch/run.py --output-dir modelwatch_artifacts
```

Outputs: `verification.json`, `batch_summary.csv` and `model.joblib`.

**Evidence boundary:** this is a deterministic monitoring simulation. A red state recommends investigation/retraining; it deliberately does not auto-deploy a replacement model.
