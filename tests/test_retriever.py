import pytest
from src.retriever.retriever import Retriever, RetrievedChunk
from src.vector_store.vector_store_client import SearchResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class FakeStore:
    def __init__(self, results: list[SearchResult]):
        self._results = results
        self.last_vector: list[float] = []
        self.last_n: int = 0

    def query(self, vector: list[float], n_results: int) -> list[SearchResult]:
        self.last_vector = vector
        self.last_n = n_results
        return self._results[:n_results]


def _make_results(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            text=f"Chunk {i} text about ESOS compliance.",
            metadata={"source": f"doc{i}.md", "section": f"Section {i}", "doc_type": "mock"},
            distance=0.1 * i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------------

def test_retrieved_chunk_fields():
    rc = RetrievedChunk(
        text="some text",
        source="esos.md",
        section="Requirements",
        doc_type="mock",
        score=0.95,
    )
    assert rc.text == "some text"
    assert rc.source == "esos.md"
    assert rc.section == "Requirements"
    assert rc.doc_type == "mock"
    assert rc.score == 0.95


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def test_retrieve_returns_retrieved_chunks():
    store = FakeStore(_make_results(5))
    retriever = Retriever(embedder=FakeEmbedder(), store=store)
    results = retriever.retrieve("What is ESOS?")
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_retrieve_respects_top_k():
    store = FakeStore(_make_results(10))
    retriever = Retriever(embedder=FakeEmbedder(), store=store, top_k=3)
    results = retriever.retrieve("ESOS threshold")
    assert len(results) <= 3
    assert store.last_n == 3


def test_retrieve_default_top_k_is_five():
    store = FakeStore(_make_results(10))
    retriever = Retriever(embedder=FakeEmbedder(), store=store)
    retriever.retrieve("question")
    assert store.last_n == 5


def test_retrieve_embeds_query_and_passes_to_store():
    store = FakeStore(_make_results(3))
    retriever = Retriever(embedder=FakeEmbedder(), store=store)
    retriever.retrieve("some query")
    assert store.last_vector == [0.1, 0.2, 0.3, 0.4]


def test_retrieve_maps_distance_to_score():
    """Lower distance → higher score. score = 1 - distance (clamped to [0,1])."""
    store = FakeStore([
        SearchResult(text="t", metadata={"source": "a", "section": "s", "doc_type": "mock"}, distance=0.2),
        SearchResult(text="t", metadata={"source": "b", "section": "s", "doc_type": "mock"}, distance=0.8),
    ])
    retriever = Retriever(embedder=FakeEmbedder(), store=store)
    results = retriever.retrieve("q")
    assert results[0].score > results[1].score


def test_retrieve_extracts_metadata_fields():
    store = FakeStore([
        SearchResult(
            text="ESOS requires audits",
            metadata={"source": "esos.md", "section": "Requirements", "doc_type": "mock"},
            distance=0.1,
        )
    ])
    retriever = Retriever(embedder=FakeEmbedder(), store=store)
    results = retriever.retrieve("ESOS")
    assert results[0].source == "esos.md"
    assert results[0].section == "Requirements"
    assert results[0].doc_type == "mock"


def test_retrieve_empty_store_returns_empty():
    store = FakeStore([])
    retriever = Retriever(embedder=FakeEmbedder(), store=store)
    assert retriever.retrieve("anything") == []


def test_retrieve_text_preserved():
    store = FakeStore([
        SearchResult(
            text="The specific text content",
            metadata={"source": "x.pdf", "section": "A", "doc_type": "real"},
            distance=0.05,
        )
    ])
    retriever = Retriever(embedder=FakeEmbedder(), store=store)
    results = retriever.retrieve("q")
    assert results[0].text == "The specific text content"
