import json
import uuid
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import config

openai_client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
qdrant_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def ensure_collection() -> None:
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if config.COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        )


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(errors="ignore")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    response = openai_client.embeddings.create(model=config.OPENAI_EMBEDDING_MODEL, input=texts, dimensions=config.EMBEDDING_DIM)
    return [item.embedding for item in response.data]


def _load_chunk_store() -> List[Dict]:
    path = Path(config.CHUNKS_STORE_PATH)
    if path.exists():
        return json.loads(path.read_text())
    return []


def _save_chunk_store(records: List[Dict]) -> None:
    path = Path(config.CHUNKS_STORE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def ingest_file(file_path: str, source_name: str) -> int:
    ensure_collection()
    text = extract_text(file_path)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    vectors = embed_texts(chunks)
    store = _load_chunk_store()
    points = []
    for chunk, vector in zip(chunks, vectors):
        chunk_id = str(uuid.uuid4())
        points.append(
            PointStruct(id=chunk_id, vector=vector, payload={"text": chunk, "source": source_name})
        )
        store.append({"chunk_id": chunk_id, "text": chunk, "source": source_name})

    qdrant_client.upsert(collection_name=config.COLLECTION_NAME, points=points)
    _save_chunk_store(store)
    return len(chunks)
