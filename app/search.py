import json
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi

from app import config
from app.ingest import embed_texts, qdrant_client

_bm25_cache = {"mtime": None, "store": None, "index": None}


def _load_store_and_index():
    path = Path(config.CHUNKS_STORE_PATH)
    if not path.exists():
        return [], None

    mtime = path.stat().st_mtime
    if _bm25_cache["mtime"] == mtime:
        return _bm25_cache["store"], _bm25_cache["index"]

    store = json.loads(path.read_text())
    corpus = [record["text"].split() for record in store]
    index = BM25Okapi(corpus) if corpus else None

    _bm25_cache["mtime"] = mtime
    _bm25_cache["store"] = store
    _bm25_cache["index"] = index
    return store, index


def vector_search(query: str, top_k: int = 10) -> List[Dict]:
    vector = embed_texts([query])[0]
    hits = qdrant_client.search(
        collection_name=config.COLLECTION_NAME, query_vector=vector, limit=top_k
    )
    return [
        {
            "chunk_id": str(h.id),
            "text": h.payload["text"],
            "source": h.payload["source"],
            "score": h.score,
        }
        for h in hits
    ]


def bm25_search(query: str, top_k: int = 10) -> List[Dict]:
    store, index = _load_store_and_index()
    if not store or index is None:
        return []
    scores = index.get_scores(query.split())
    ranked = sorted(zip(store, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [
        {"chunk_id": r["chunk_id"], "text": r["text"], "source": r["source"], "score": float(s)}
        for r, s in ranked
    ]


def reciprocal_rank_fusion(
    vector_results: List[Dict], bm25_results: List[Dict], k: int = 60, top_k: int = 5
) -> List[Dict]:
    scores: Dict[str, float] = {}
    lookup: Dict[str, Dict] = {}
    for result_list in (vector_results, bm25_results):
        for rank, item in enumerate(result_list):
            scores[item["chunk_id"]] = scores.get(item["chunk_id"], 0.0) + 1 / (k + rank + 1)
            lookup[item["chunk_id"]] = item
    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{**lookup[cid], "score": scores[cid]} for cid in ranked_ids]


def hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
    vector_results = vector_search(query, top_k=10)
    bm25_results = bm25_search(query, top_k=10)
    return reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)
