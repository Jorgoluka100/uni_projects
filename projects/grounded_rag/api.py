from fastapi import FastAPI
from pydantic import BaseModel, Field

from run import GroundedRAG

app = FastAPI(
    title="GroundedRAG",
    version="2.0.0",
    description="Evidence-first hybrid retrieval with dense vector indexing, citations, abstention and allow-listed read-only tool routing.",
)
system = GroundedRAG()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=5)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": len(system.docs),
        "dense_embedding_dimensions": system.vector_store.dimensions,
        "tools": ["ticket_analytics"],
    }


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return system.answer(request.query, top_k=request.top_k)
