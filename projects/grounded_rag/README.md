# GroundedRAG — Retrieval, Citations & Safety Evaluation

A compact, testable RAG engineering project focused on the parts that should be measurable before an LLM is trusted: **hybrid retrieval, reranking, citations, abstention, prompt-injection handling, API integration and automated evidence gates**.

## Why this project exists

Many AI roles now expect more than prompt demos. They expect retrieval quality to be measured, unsupported questions to be handled safely, sources to be traceable, and AI behaviour to be testable in CI. GroundedRAG demonstrates that engineering layer without requiring a paid model API.

## Architecture

`query → transparent query expansion → BM25 + word/character TF-IDF retrieval → hybrid reranking → evidence threshold → extractive grounded response → source citation`

Safety path:

`user input → prompt-injection pattern check → block privileged instruction attempts → abstain when evidence is weak`

The project deliberately keeps the answer layer extractive and deterministic. A hosted or open-weight LLM can be placed behind the same retrieval contract later, but this repository does **not** claim LLM-generation quality that has not been evaluated.

## Verified evidence

The built-in evaluation uses a **small synthetic enterprise-policy knowledge base**, designed for deterministic methodology testing rather than real-world accuracy claims.

- 12 golden queries: 10 answerable + 2 deliberately unanswerable
- Recall@3: **1.000**
- MRR@3: **1.000**
- NDCG@3: **1.000**
- citation-or-correct-abstention accuracy: **1.000**
- abstention decision accuracy: **1.000**
- prompt-injection block rate: **1.000** across the deterministic attack fixture

These perfect scores are intentionally **not presented as generalisation evidence**. They show that the implementation satisfies its frozen test fixture and that the evaluation contract is executable in CI.

## Run locally

```bash
python -m pip install -r requirements.txt
python run.py --self-test
python run.py --output-dir artifacts
python run.py --query "What should a RAG system do when it cannot find evidence?"
```

## API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /query` with `{"query": "...", "top_k": 3}`

## Docker

```bash
docker build -t grounded-rag .
docker run -p 8000:8000 grounded-rag
```

## What this proves

- retrieval and ranking fundamentals rather than opaque prompt-only demos
- source-grounded answers and explicit citations
- abstention when retrieval evidence is insufficient
- prompt-injection handling before tool execution
- deterministic golden-set evaluation
- API-ready packaging and Docker deployment pattern
- CI-friendly machine-readable evidence

## What this does **not** prove

- production deployment or production traffic
- real enterprise-document generalisation
- semantic-embedding model quality
- hosted-LLM answer quality
- complete prompt-injection defence against adversarial attacks

Those limitations are intentional. The project is designed so stronger embedding models, a vector database, an LLM provider and production observability can be introduced later without changing the evidence-first evaluation contract.
