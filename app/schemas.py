from typing import List, Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    source: str
    chunks_added: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    latency_ms: float
    tokens_used: Optional[int] = None
