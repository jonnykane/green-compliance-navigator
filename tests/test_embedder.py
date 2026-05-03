import pytest
from unittest.mock import MagicMock, patch

from src.embedder.embedding_client import EmbeddingClient, EmbeddingResult


# ---------------------------------------------------------------------------
# Fake Voyage client used across tests
# ---------------------------------------------------------------------------

class FakeVoyageClient:
    """Deterministic fake that returns fixed-length vectors without network calls."""

    DIM = 8

    def embed(self, texts: list[str], model: str, input_type: str) -> MagicMock:
        result = MagicMock()
        result.embeddings = [[float(i) / 10] * self.DIM for i in range(len(texts))]
        return result


# ---------------------------------------------------------------------------
# EmbeddingResult
# ---------------------------------------------------------------------------

def test_embedding_result_stores_vector_and_text():
    er = EmbeddingResult(text="hello", vector=[0.1, 0.2, 0.3])
    assert er.text == "hello"
    assert er.vector == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# EmbeddingClient
# ---------------------------------------------------------------------------

def test_embed_texts_returns_correct_count():
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3")
    texts = ["text one", "text two", "text three"]
    results = client.embed_texts(texts)
    assert len(results) == 3


def test_embed_texts_returns_embedding_results():
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3")
    results = client.embed_texts(["hello"])
    assert isinstance(results[0], EmbeddingResult)
    assert results[0].text == "hello"
    assert len(results[0].vector) == FakeVoyageClient.DIM


def test_embed_texts_preserves_order():
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3")
    texts = ["alpha", "beta", "gamma"]
    results = client.embed_texts(texts)
    assert [r.text for r in results] == texts


def test_embed_texts_empty_list_returns_empty():
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3")
    assert client.embed_texts([]) == []


def test_embed_texts_batches_large_input():
    """Client should split into batches and still return all results."""
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3", batch_size=3)
    texts = [f"doc {i}" for i in range(10)]
    results = client.embed_texts(texts)
    assert len(results) == 10


def test_embed_texts_uses_document_input_type():
    """embed_texts must pass input_type='document' to the underlying client."""
    fake = FakeVoyageClient()
    spy = MagicMock(wraps=fake.embed)
    fake.embed = spy
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    client.embed_texts(["some text"])
    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs.get("input_type") == "document" or spy.call_args.args[2] == "document"


def test_embed_query_uses_query_input_type():
    """embed_query must pass input_type='query' to the underlying client."""
    fake = FakeVoyageClient()
    spy = MagicMock(wraps=fake.embed)
    fake.embed = spy
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    client.embed_query("what is ESOS?")
    spy.assert_called_once()
    call_args = spy.call_args
    passed_input_type = call_args.kwargs.get("input_type") or call_args.args[2]
    assert passed_input_type == "query"


def test_embed_query_returns_single_vector():
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3")
    vector = client.embed_query("ESOS threshold?")
    assert isinstance(vector, list)
    assert len(vector) == FakeVoyageClient.DIM
