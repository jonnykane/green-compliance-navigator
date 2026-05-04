"""
Tests for src/wiring.py — verifies that _build_vector_store selects the correct
implementation based on whether PINECONE_API_KEY is set.

No real API calls are made; external clients are mocked.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.vector_store.vector_store_client import VectorStoreClient
from src.vector_store.pinecone_vector_store import PineconeVectorStore


# ---------------------------------------------------------------------------
# _build_vector_store — ChromaDB path (no PINECONE_API_KEY)
# ---------------------------------------------------------------------------

def test_build_vector_store_returns_chroma_when_no_pinecone_key():
    """No PINECONE_API_KEY → VectorStoreClient (ChromaDB-backed)."""
    with patch("src.wiring.settings") as mock_settings, \
         patch("src.wiring.chromadb") as mock_chromadb:
        mock_settings.PINECONE_API_KEY = ""
        mock_settings.CHROMA_PERSIST_DIR = Path("/tmp/test_chroma")
        mock_settings.CHROMA_COLLECTION_NAME = "test_col"

        from src.wiring import _build_vector_store
        store = _build_vector_store()

    assert isinstance(store, VectorStoreClient)


def test_build_vector_store_chroma_path_uses_correct_collection():
    """ChromaDB path passes CHROMA_COLLECTION_NAME to VectorStoreClient."""
    with patch("src.wiring.settings") as mock_settings, \
         patch("src.wiring.chromadb") as mock_chromadb:
        mock_settings.PINECONE_API_KEY = ""
        mock_settings.CHROMA_PERSIST_DIR = Path("/tmp/test_chroma")
        mock_settings.CHROMA_COLLECTION_NAME = "my_collection"

        from src.wiring import _build_vector_store
        _build_vector_store()

    mock_chromadb.PersistentClient.return_value.get_or_create_collection.assert_called_once_with(
        "my_collection"
    )


# ---------------------------------------------------------------------------
# _build_vector_store — Pinecone path (PINECONE_API_KEY set)
# ---------------------------------------------------------------------------

def test_build_vector_store_returns_pinecone_when_key_set():
    """PINECONE_API_KEY present → PineconeVectorStore."""
    with patch("src.wiring.settings") as mock_settings, \
         patch("src.wiring.Pinecone") as mock_pinecone_cls:
        mock_settings.PINECONE_API_KEY = "pc-test-key"
        mock_settings.PINECONE_INDEX_NAME = "green-compliance-navigator"

        from src.wiring import _build_vector_store
        store = _build_vector_store()

    assert isinstance(store, PineconeVectorStore)


def test_build_vector_store_pinecone_passes_api_key():
    """Pinecone client is constructed with the correct API key."""
    with patch("src.wiring.settings") as mock_settings, \
         patch("src.wiring.Pinecone") as mock_pinecone_cls:
        mock_settings.PINECONE_API_KEY = "pc-real-key"
        mock_settings.PINECONE_INDEX_NAME = "green-compliance-navigator"

        from src.wiring import _build_vector_store
        _build_vector_store()

    mock_pinecone_cls.assert_called_once_with(api_key="pc-real-key")


def test_build_vector_store_pinecone_uses_index_name():
    """PineconeVectorStore is pointed at the configured index name."""
    with patch("src.wiring.settings") as mock_settings, \
         patch("src.wiring.Pinecone") as mock_pinecone_cls:
        mock_settings.PINECONE_API_KEY = "pc-test-key"
        mock_settings.PINECONE_INDEX_NAME = "my-custom-index"

        from src.wiring import _build_vector_store
        _build_vector_store()

    mock_pinecone_cls.return_value.Index.assert_called_once_with("my-custom-index")


# ---------------------------------------------------------------------------
# build_query_engine — RETRIEVER_TOP_K wired from settings
# ---------------------------------------------------------------------------

def test_build_query_engine_passes_top_k_from_settings():
    """Retriever must be constructed with top_k from settings.RETRIEVER_TOP_K."""
    with patch("src.wiring.settings") as mock_settings, \
         patch("src.wiring.Pinecone"), \
         patch("src.wiring.voyageai"), \
         patch("src.wiring.anthropic"), \
         patch("src.wiring.Retriever") as mock_retriever_cls:
        mock_settings.PINECONE_API_KEY = "pc-test-key"
        mock_settings.PINECONE_INDEX_NAME = "test-index"
        mock_settings.VOYAGE_API_KEY = "va-key"
        mock_settings.VOYAGE_MODEL = "voyage-3"
        mock_settings.VOYAGE_BATCH_SIZE = 128
        mock_settings.ANTHROPIC_API_KEY = "sk-test"
        mock_settings.ANTHROPIC_MODEL = "claude-sonnet-4-5"
        mock_settings.RETRIEVER_TOP_K = 8

        from src.wiring import build_query_engine
        build_query_engine()

    _, kwargs = mock_retriever_cls.call_args
    assert kwargs["top_k"] == 8


def test_build_query_engine_top_k_respects_custom_setting():
    """A non-default RETRIEVER_TOP_K value must be passed through unchanged."""
    with patch("src.wiring.settings") as mock_settings, \
         patch("src.wiring.Pinecone"), \
         patch("src.wiring.voyageai"), \
         patch("src.wiring.anthropic"), \
         patch("src.wiring.Retriever") as mock_retriever_cls:
        mock_settings.PINECONE_API_KEY = "pc-test-key"
        mock_settings.PINECONE_INDEX_NAME = "test-index"
        mock_settings.VOYAGE_API_KEY = "va-key"
        mock_settings.VOYAGE_MODEL = "voyage-3"
        mock_settings.VOYAGE_BATCH_SIZE = 128
        mock_settings.ANTHROPIC_API_KEY = "sk-test"
        mock_settings.ANTHROPIC_MODEL = "claude-sonnet-4-5"
        mock_settings.RETRIEVER_TOP_K = 12

        from src.wiring import build_query_engine
        build_query_engine()

    _, kwargs = mock_retriever_cls.call_args
    assert kwargs["top_k"] == 12
