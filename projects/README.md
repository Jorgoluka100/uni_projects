# Production-Style Projects

This directory contains compact, package-like projects added after the original notebook portfolio. Each project has a clear decision, reproducible command, machine-readable evidence and a self-test; API/container packaging is added where it strengthens the engineering case rather than as decoration.

## Projects

1. **GroundedRAG v2** — hybrid sparse+dense retrieval, deterministic LSA embeddings, dense vector indexing, grounded citations, abstention, allow-listed read-only tool routing, prompt-injection blocking, FastAPI and Docker packaging.
2. **CareerLens AI** — hybrid job matching and skill-gap ranking for NLP / information retrieval.
3. **ExperimentLab** — experiment decision engine with CUPED variance reduction, bootstrap uncertainty, guardrails and power diagnostics.
4. **ModelWatch** — production ML monitoring with drift, performance, calibration, subgroup and retraining-policy checks.

Together these projects cover gaps that the original notebook portfolio did not demonstrate strongly: **applied-AI/RAG engineering, structured tool orchestration, NLP retrieval, experimentation/causal decision science and MLOps monitoring**.

All demo datasets and fixtures are explicitly labelled. Metrics are promoted to the root README only after the GitHub Actions verification run succeeds and retained evidence is committed. Perfect scores on small deterministic fixtures are treated as test-contract evidence, not as generalisation claims.

GroundedRAG v2 intentionally keeps its dense embedding layer local and deterministic (TF-IDF + TruncatedSVD) rather than claiming transformer embeddings, a managed vector database or hosted-LLM quality that has not been evaluated.
