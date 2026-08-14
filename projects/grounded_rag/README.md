# GroundedRAG v2 — Retrieval, Vector Search, Tool Routing & Safety

A compact, testable applied-AI project focused on the parts of a RAG/agent system that can be measured before a hosted LLM is trusted: **hybrid retrieval, dense vector indexing, citations, abstention, safe read-only tool routing, prompt-injection blocking, API packaging and automated evidence gates**.

## Why this project exists

Modern AI-engineering roles increasingly expect more than prompt demos. They expect retrieval quality to be measured, unsupported questions to be handled safely, sources to be traceable, tool use to be constrained, and AI behaviour to be testable in CI.

GroundedRAG v2 demonstrates that engineering contract without requiring a paid model API.

## Architecture

Retrieval path:

`query → transparent expansion → BM25 + word TF-IDF + char TF-IDF + dense LSA vector index → hybrid reranking → evidence threshold → grounded extractive response → citation`

Tool path:

`query → injection check → deterministic router → allow-listed read-only ticket analytics tool → structured result + tool audit payload`

Safety path:

`user input → injection check → block suspicious instruction override attempts before any tool execution → abstain when retrieval evidence is weak`

The dense store uses deterministic **Latent Semantic Analysis (TF-IDF + TruncatedSVD)** embeddings so the project remains local and reproducible. It is a real dense vector retrieval layer, but it is **not** presented as a production vector database or transformer embedding service.

## Verified evidence

The evaluation uses a deliberately small synthetic enterprise-policy corpus and frozen incident dataset. It is a methodology/evaluation fixture, not a claim about real enterprise traffic.

Retrieval fixture:

- 12 golden queries: 10 answerable + 2 deliberately unanswerable
- Recall@3: **1.000**
- MRR@3: **1.000**
- NDCG@3: **1.000**
- citation-or-correct-abstention accuracy: **1.000**
- abstention decision accuracy: **1.000**

Tool/safety fixture:

- 4 frozen ticket-analytics questions
- tool-route accuracy: **1.000**
- tool-result accuracy: **1.000**
- prompt-injection block rate: **1.000**
- tool execution on attack fixture: **0.000**

Perfect fixture scores mean the deterministic implementation satisfies its frozen test contract. They **do not** prove production generalisation.

## Run locally

```bash
python -m pip install -r requirements.txt
python run.py --self-test
python run.py --output-dir artifacts
python run.py --query "What should a RAG system do when it cannot find evidence?"
python run.py --query "How many open Severity 1 tickets are there?"
```

## API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /query` with `{"query": "...", "top_k": 3}`

`/health` exposes the document count, dense embedding dimensionality and available read-only tool names.

## Docker

```bash
docker build -t grounded-rag .
docker run -p 8000:8000 grounded-rag
```

## What this proves

- sparse + dense retrieval fundamentals
- deterministic dense vector indexing with LSA embeddings
- source-grounded responses and explicit citations
- abstention when evidence is insufficient
- constrained tool routing with an allow-listed read-only analytics tool
- prompt-injection blocking before tool execution
- golden-set retrieval, tool and safety evaluation
- FastAPI and Docker packaging
- CI-friendly machine-readable evidence

## What this does **not** prove

- production deployment or real production traffic
- transformer-embedding quality
- a managed vector database such as Qdrant/Pinecone/Weaviate
- hosted-LLM answer quality
- autonomous multi-step agent reasoning
- complete prompt-injection defence against adaptive adversaries

Those limitations stay explicit. The important portfolio signal is that retrieval, tool use and safety have measurable contracts rather than being hidden inside a prompt demo.
