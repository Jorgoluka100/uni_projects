# Advanced Project Completion Record

The portfolio now follows **preserve + harden + verify**: useful original notebooks stay on `main`, targeted completion layers fix real gaps, and only clean current evidence is promoted.

## Verified / executed completion paths

| Project | Current status | Evidence outcome |
|---|---|---|
| ConsultAI | **Verified flagship** | Clean notebook run and independent deterministic reconstruction passed. Synthetic organisational inputs remain explicit. |
| VisionForge | **Verified flagship** | Clean notebook run, independent public-test recomputation, bootstrap intervals, abstention reproduction and TorchScript/ONNX parity passed. |
| AeroFlow | **Verified current-data iteration** | Official 2026 BTS regression v2 is retained as a negative result; delay-risk v3 reframing passed on untouched May 2026 evidence. |
| NYC Airbnb | **Verified current-data iteration** | 14-Jun-2026 Inside Airbnb snapshot with complete-neighbourhood holdout and retained baseline comparison passed. |
| Hybrid Movie Recommender | **Verified ranking iteration** | First clean run exposed a per-user temporal leakage edge case; corrected past-only split then passed ranking evidence gate. |
| Aviation PostgreSQL | **Verified engineering evidence** | PostgreSQL 17, 1M-row workload, two `EXPLAIN (ANALYZE, BUFFERS)` plans, targeted index and semantic reconciliation retained. |
| Logistic Regression + Naive Bayes PySpark | **Verified historical Spark benchmark** | First clean run exposed a schema bug; corrected numeric/categorical contract, validation selection, untouched test and serialized pipeline reload passed. |
| Fraud / AML | **Verified methodology** | Chronological synthetic pipeline, validation-only threshold, review/cost evidence and reload passed. Not real AML performance. |
| Strategic Telecom Churn + SQL | **Verified methodology** | Synthetic train/validation/test decision pipeline and SQL↔Pandas reconciliation passed. |
| Parkinson's Progression | **Verified negative result** | Complete-subject split and subject-bootstrap uncertainty passed; stricter test underperformed the median baseline, so no positive performance claim is promoted. |
| Clustering | **Verified methodology** | Multiple internal indices plus resampling stability completed and retained. |
| Pathfinding | **Verified benchmark** | BFS/Dijkstra/A* returned valid optimal paths across all seeded benchmark cases; expanded-node evidence retained. |

## Original verified notebooks retained

The seven original verified flagships remain unchanged unless a genuine bug or materially better rerun is introduced:

- `01_UK_House_Price_Analysis_and_Prediction.ipynb`
- `02_SQL_Sales_and_Customer_Analysis.ipynb`
- `03_Customer_Churn_Prediction.ipynb`
- `04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb`
- `05_Energy_Demand_Forecasting_with_TensorFlow.ipynb`
- `06_Clickstream_Analysis_with_PySpark.ipynb`
- `07_London_Air_Quality_Analysis_with_R.ipynb`

ConsultAI and VisionForge now also have clean executed notebook evidence and are enforced by the repository validator.

## Advanced work that is intentionally not promoted yet

| Project | Why promotion is still blocked |
|---|---|
| Advanced Multi-Modal Health Analytics | Requires a clean patient/group-safe run with defensible source provenance and project-specific safety evidence. |
| PyTorch Medical AI X-ray Diagnosis | Requires patient/group independence, source provenance, uncertainty/abstention and explicit non-clinical evidence. |
| LLM Mastery — Core | Current checkpoint/training provenance and fresh held-out generations/loss evidence still required. |
| LLM Mastery — Alignment | Fresh base-vs-aligned generations are required before the frozen evaluation harness can support a promoted claim. |
| CineIntelligence NoSQL | v2 ingestion/query engineering is ready, but fresh source/licence evidence must accompany the run. |
| Standalone KDD Cup Analysis | Historical v2 exists, but the separately verified Spark KDD benchmark already covers the strongest distributed-classification signal; this notebook remains an historical experiment. |

## What verification changed

Clean execution was not ceremonial. It found real problems:

1. **Recommender leakage:** the initial latest-positive split could leave later lower-rated interactions in training. Evaluated-user histories are now truncated strictly before the target.
2. **Spark schema:** numeric KDD predictors arrived with object dtype and were being decoded as strings. Only the three documented categorical fields now stay categorical.
3. **AeroFlow formulation:** schedule-only delay-minute regression failed the global-median MAE baseline. The failure is retained and the decision was reframed to 15-minute delay risk.
4. **Parkinson's leakage/generalisation:** complete-subject holdout removed optimistic row-level leakage and exposed poor unseen-subject performance.

## Evidence discipline

A result is promoted only when the **current** path has run cleanly and the relevant evidence is retained. Historical datasets, synthetic inputs and negative results are labelled rather than disguised. The repository validator requires the retained verification files to remain present and passing.

The remaining work is no longer broad portfolio reconstruction. It is limited to the few source/GPU/safety-dependent laboratory projects above and normal future maintenance.
