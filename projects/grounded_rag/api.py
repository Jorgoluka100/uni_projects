from fastapi import FastAPI
from pydantic import BaseModel, Field

from run import GroundedRAG

app = FastAPI(title="GroundedRAG", version="1.0.0")
system = GroundedRAG()

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=5)

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(system.docs)}

@app.post("/query")
def query(request: QueryRequest) -> dict:
    return system.answer(request.query, top_k=request.top_k)
