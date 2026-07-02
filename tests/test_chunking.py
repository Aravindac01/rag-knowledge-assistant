from app.ingest import chunk_text


def test_chunk_text_splits_long_input():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("") == []


def test_chunk_text_short_input_single_chunk():
    text = "short text well under the chunk size"
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text
