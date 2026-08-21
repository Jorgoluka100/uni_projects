# Retrieval-Augmented Support Assistant

I built this as a small local RAG project where the retrieval and tool behaviour can be tested without depending on a paid LLM API.

The main things I wanted to check were simple: can the system find the right source, cite it, refuse weak matches, route a read-only analytics question correctly and stop obvious prompt-injection attempts before a tool runs?

## Architecture

Retrieval path:

`query → query expansion → BM25 + word TF-IDF + char TF-IDF + LSA vectors → hybrid reranking → evidence threshold → answer or abstain → citation`

Tool path:

`query → injection check → deterministic router → allow-listed read-only ticket analytics tool → structured result + audit payload`

The dense retrieval layer uses TF-IDF followed by TruncatedSVD (LSA). That keeps the whole project local and reproducible. I do not present it as a replacement for a production vector database or transformer embedding service.

## Evaluation fixture

The evaluation data is deliberately small and synthetic. It is there to test the implementation, not to make a claim about production enterprise traffic.

Retrieval checks:

- 12 golden queries: 10 answerable and 2 deliberately unanswerable
- Recall@3: **1.000**
- MRR@3: **1.000**
- NDCG@3: **1.000**
- citation-or-correct-abstention accuracy: **1.000**
- abstention decision accuracy: **1.000**

Tool and safety checks:

- 4 frozen ticket-analytics questions
- tool-route accuracy: **1.000**
- tool-result accuracy: **1.000**
- prompt-injection block rate: **1.000**
- tool execution on attack fixture: **0.000**

Those perfect scores only mean the deterministic code passes this frozen test set. They do **not** mean the system would be perfect on real company documents or adversarial traffic.

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

`/health` reports the document count, dense-vector dimensionality and the available read-only tools.

## Docker

```bash
docker build -t grounded-rag .
docker run -p 8000:8000 grounded-rag
```

## What is demonstrated here

- sparse and dense retrieval
- deterministic LSA vector indexing
- source citations
- abstention when evidence is weak
- allow-listed read-only tool routing
- prompt-injection checks before tool execution
- golden-set evaluation
- FastAPI and Docker packaging
- machine-readable CI evidence

## What is outside the scope

- real production traffic
- transformer embedding quality
- a managed vector database such as Qdrant, Pinecone or Weaviate
- hosted-LLM answer quality
- autonomous multi-step agents
- complete defence against adaptive prompt-injection attacks

The point of the project is to keep the parts that can be measured visible rather than hiding everything inside a prompt demo.