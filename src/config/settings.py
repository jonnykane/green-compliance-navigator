import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Project root is two levels up from this file (src/config/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORPUS_REAL_DIR = Path(os.getenv("CORPUS_REAL_DIR", str(PROJECT_ROOT / "corpus" / "real")))
CORPUS_MOCK_DIR = Path(os.getenv("CORPUS_MOCK_DIR", str(PROJECT_ROOT / "corpus" / "mock")))
CORPUS_METADATA_DIR = Path(os.getenv("CORPUS_METADATA_DIR", str(PROJECT_ROOT / "corpus" / "metadata")))

CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db")))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "green_compliance")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")
VOYAGE_BATCH_SIZE = int(os.getenv("VOYAGE_BATCH_SIZE", "128"))
VOYAGE_REQUEST_INTERVAL = float(os.getenv("VOYAGE_REQUEST_INTERVAL", "0.0"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "green-compliance-navigator")

CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))

RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "8"))

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# Canonical source injection — maps trigger phrases (lowercase) to the source
# filename that must be present in retrieved chunks when that phrase appears in
# the user's question.  Loaded from config so query.py stays data-free.
CANONICAL_SOURCE_MAP: dict[str, str] = {
    "secr": "secr_guidelines_summary.md",
    "streamlined energy": "secr_guidelines_summary.md",
    "streamlined carbon": "secr_guidelines_summary.md",
}
