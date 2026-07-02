from app.search import reciprocal_rank_fusion


def test_rrf_merges_results_from_both_lists():
    vector_results = [
        {"chunk_id": "a", "text": "A", "source": "doc1", "score": 0.9},
        {"chunk_id": "b", "text": "B", "source": "doc1", "score": 0.5},
    ]
    bm25_results = [
        {"chunk_id": "b", "text": "B", "source": "doc1", "score": 5.0},
        {"chunk_id": "a", "text": "A", "source": "doc1", "score": 3.0},
    ]
    fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=2)
    ids = [f["chunk_id"] for f in fused]
    assert set(ids) == {"a", "b"}


def test_rrf_dedupes_same_chunk_seen_in_both_lists():
    vector_results = [{"chunk_id": "a", "text": "A", "source": "doc1", "score": 0.9}]
    bm25_results = [{"chunk_id": "a", "text": "A", "source": "doc1", "score": 5.0}]
    fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=5)
    assert len(fused) == 1


def test_rrf_respects_top_k_limit():
    vector_results = [
        {"chunk_id": str(i), "text": str(i), "source": "d", "score": 1.0} for i in range(10)
    ]
    fused = reciprocal_rank_fusion(vector_results, [], top_k=3)
    assert len(fused) == 3
