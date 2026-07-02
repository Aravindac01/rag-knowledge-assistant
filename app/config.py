import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gemini-2.5-flash")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "gemini-embedding-001")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")
CHUNKS_STORE_PATH = os.getenv("CHUNKS_STORE_PATH", "./data/chunks.json")

EMBEDDING_DIM = 768
