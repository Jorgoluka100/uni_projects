# Restored Advanced Projects — Audit Notes

These notes complement the restored notebooks where the original file contains little or no Markdown narrative. The original code is preserved because it contains useful implementation work; this document records what is already valuable and what must be strengthened before a project is promoted to the verified flagship tier.

A project is **not** considered verified merely because old cells have execution counts or old outputs. Promotion requires a clean rerun of the current version and a review of the resulting evidence.

## AeroFlow AI Engine

### What is worth keeping

- End-to-end aviation modelling workflow.
- TensorFlow / LSTM experimentation alongside classical preprocessing.
- A useful foundation for an operational flight-delay decision-support project.

### Current audit findings

- The current notebook generates synthetic flight data. Any result must therefore be described as a methodology demonstration, not evidence about real airline operations.
- The visible pipeline label-encodes categorical variables and fits `StandardScaler` before `train_test_split`. That allows holdout information to influence preprocessing and must be corrected before promotion.
- The notebook needs an explicit baseline, temporal evaluation design where timestamps are used, uncertainty/error analysis, and a clear operational decision policy.

### Completion target

Create a leakage-safe v2 that splits first, fits preprocessing only on training data, compares against simple baselines, evaluates on an untouched holdout, states the synthetic-data limitation prominently, and reports what action a prediction would actually support.

---

## CineIntelligence NoSQL Data Engineering

### What is worth keeping

- TinyDB/document-store implementation.
- Kaggle TMDB data ingestion and relational-to-document transformation.
- A useful NoSQL/data-engineering signal distinct from the modelling-heavy projects.

### Current audit findings

- The notebook has little narrative documentation.
- JSON-like fields are parsed using quote replacement plus `json.loads`, which is brittle for malformed or escaped content.
- It needs a documented document schema, robust parsing/quarantine behaviour, explicit indexes/query patterns where the chosen engine supports them, data-quality checks, and reproducible query benchmarks.

### Completion target

Turn the notebook into a small data-engineering case study: define the source contract, validate and quarantine malformed rows, build a documented movie document model, demonstrate representative analytical queries, and compare the document workflow against a simple Pandas/relational baseline where meaningful.

---

## KDD Cup Analysis

### What is worth keeping

- PySpark setup and large-dataset processing workflow.
- UCI KDD Cup 1999 intrusion data ingestion.
- Distributed-ML learning evidence.

### Current audit findings

- The dataset is historical and must never be presented as current cybersecurity behaviour.
- The notebook needs stronger provenance/version documentation and an explicit feature/label schema.
- Evaluation must address the severe class imbalance and the known limitations of KDD Cup 1999 as a modern intrusion benchmark.

### Completion target

Add a reproducible Spark pipeline with schema validation, a defensible split strategy, simple baseline, imbalance-aware metrics, per-class error analysis, saved-pipeline reload check, and a prominent historical-dataset limitation section.

---

## LLM Mastery — Alignment

### What is worth keeping

- Custom LLaMA-style implementation work.
- RoPE, RMSNorm, SwiGLU, attention/KV-cache concepts, and alignment experimentation.
- Strong evidence that the project goes beyond merely calling a hosted LLM API.

### Current audit findings

- The notebook currently retains an old environment-install error involving `numpy==1.26.4`; this is a rerun warning and prevents promotion of old outputs as verified evidence.
- Environment mutation is too brittle and should be replaced by a documented compatible dependency set.
- The source/checkpoint provenance and licence should be made explicit.
- Alignment needs a fixed evaluation set and a before-versus-after comparison rather than subjective sample inspection alone.

### Completion target

Pin a reproducible environment, document model/checkpoint provenance, run a clean base-versus-aligned evaluation on a fixed prompt set, score helpfulness/safety/format adherence using a transparent rubric, retain representative failures, and state the limits of the evaluation.

---

## LLM Mastery — Core

### What is worth keeping

- Transformer implementation from low-level components rather than a thin wrapper.
- Multi-head attention, embeddings, transformer blocks, training loop and checkpoint logic.
- Useful evidence of architecture-level understanding.

### Current audit findings

- The notebook needs stronger dataset provenance and data-quality documentation.
- Training/evaluation configuration and checkpoint selection should be easier to reproduce.
- A simple language-modelling baseline and a fixed validation metric such as loss/perplexity should be reported consistently.

### Completion target

Document the corpus and tokenizer, separate train/validation data deterministically, save the full training configuration, compare against a simple baseline or smaller ablation, report held-out loss/perplexity, and include fixed-seed generation examples plus failure analysis.

---

## PyTorch Medical AI — X-ray Diagnosis

### What is worth keeping

- Medical-imaging/deep-learning implementation signal.
- A useful foundation for explainability and uncertainty work.

### Current audit findings

- The notebook lacks sufficient narrative documentation for a health-related project.
- Before any metric is promoted, the split must be demonstrably patient-safe/group-safe where patient identifiers are available.
- Dataset provenance, intended/non-intended use, uncertainty, Grad-CAM limitations, and the distinction between a research demonstration and a clinical device must be explicit.

### Completion target

Add patient/group leakage protection, dataset/provenance documentation, a simple baseline, untouched evaluation, uncertainty or abstention analysis, class-level error analysis, explainability limitations, and a strong non-clinical-use statement. Only then rerun and promote measured results.

---

## Stored-output warnings found by CI

The integrity check currently identifies two old stored errors in advanced notebooks:

- **LLM Mastery — Alignment:** an environment/package-install `SystemExit` involving NumPy.
- **Naive Bayes with PySpark:** a retained `KeyboardInterrupt` output.

These are intentionally treated as **warnings for the advanced tier**, not as verified evidence. If either project is promoted to the verified tier, the current notebook must first complete a clean restart/run-all with no stored error output.

## Why this document exists

Large working notebooks should not be rewritten purely to satisfy a formatting rule. The portfolio therefore accepts this companion audit as narrative documentation for advanced projects while CI continues to enforce hard integrity requirements: valid notebook JSON, at least one code cell, README discoverability, and stricter execution evidence for the verified flagship tier.
