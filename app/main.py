import os
import tempfile
import time

from fastapi import FastAPI, File, UploadFile

from app.ingest import ingest_file
from app.rag_chain import generate_answer
from app.schemas import IngestResponse, QueryRequest, QueryResponse, SourceChunk
from app.search import hybrid_search

app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        num_chunks = ingest_file(tmp_path, source_name=file.filename)
    finally:
        os.remove(tmp_path)
    return IngestResponse(source=file.filename, chunks_added=num_chunks)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start = time.time()
    chunks = hybrid_search(request.question, top_k=request.top_k)
    answer, tokens_used = generate_answer(request.question, chunks)
    latency_ms = (time.time() - start) * 1000
    sources = [
        SourceChunk(
            chunk_id=c["chunk_id"], source=c["source"], text=c["text"][:300], score=c["score"]
        )
        for c in chunks
    ]
    return QueryResponse(answer=answer, sources=sources, latency_ms=latency_ms, tokens_used=tokens_used)
