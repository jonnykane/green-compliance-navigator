import pytest
from src.vector_store.vector_store_client import VectorStoreClient, SearchResult


# ---------------------------------------------------------------------------
# In-memory fake ChromaDB collection used across tests
# ---------------------------------------------------------------------------

class FakeCollection:
    """Minimal in-memory imitation of a ChromaDB collection."""

    def __init__(self):
        self._ids: list[str] = []
        self._embeddings: list[list[float]] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._ids.extend(ids)
        self._embeddings.extend(embeddings)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict:
        # Return the first n_results stored documents (ignores actual similarity)
        n = min(n_results, len(self._documents))
        return {
            "ids": [self._ids[:n]],
            "documents": [self._documents[:n]],
            "metadatas": [self._metadatas[:n]],
            "distances": [[0.1 * i for i in range(n)]],
        }

    def count(self) -> int:
        return len(self._ids)


class FakeChromaClient:
    def __init__(self):
        self._collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name: str) -> FakeCollection:
        if name not in self._collections:
            self._collections[name] = FakeCollection()
        return self._collections[name]


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

def test_search_result_fields():
    sr = SearchResult(text="content", metadata={"source": "a.md"}, distance=0.05)
    assert sr.text == "content"
    assert sr.metadata["source"] == "a.md"
    assert sr.distance == 0.05


# ---------------------------------------------------------------------------
# VectorStoreClient
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    chroma = FakeChromaClient()
    return VectorStoreClient(chroma_client=chroma, collection_name="test_col")


def test_add_chunks_stores_documents(store):
    texts = ["chunk one", "chunk two"]
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    metadatas = [{"source": "a.md"}, {"source": "b.md"}]
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    assert store.count() == 2


def test_add_chunks_generates_unique_ids(store):
    texts = ["a", "b", "c"]
    vectors = [[0.0], [0.1], [0.2]]
    metadatas = [{}, {}, {}]
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    # Calling again should not collide (IDs include chunk index offset)
    store.add_chunks(texts=["d"], vectors=[[0.3]], metadatas=[{}])
    assert store.count() == 4


def test_add_chunks_empty_is_noop(store):
    store.add_chunks(texts=[], vectors=[], metadatas=[])
    assert store.count() == 0


def test_add_chunks_mismatched_lengths_raises(store):
    with pytest.raises(ValueError):
        store.add_chunks(texts=["a", "b"], vectors=[[0.1]], metadatas=[{}])


def test_query_returns_search_results(store):
    store.add_chunks(
        texts=["ESOS requires audits", "SECR requires reports"],
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"source": "esos.md"}, {"source": "secr.pdf"}],
    )
    results = store.query(vector=[0.1, 0.2], n_results=2)
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)


def test_query_returns_at_most_n_results(store):
    for i in range(5):
        store.add_chunks(
            texts=[f"doc {i}"],
            vectors=[[float(i)]],
            metadatas=[{"source": f"doc{i}.md"}],
        )
    results = store.query(vector=[0.0], n_results=3)
    assert len(results) <= 3


def test_query_result_has_distance(store):
    store.add_chunks(texts=["hello"], vectors=[[0.5]], metadatas=[{"source": "x.md"}])
    results = store.query(vector=[0.5], n_results=1)
    assert results[0].distance is not None


def test_count_reflects_additions(store):
    assert store.count() == 0
    store.add_chunks(texts=["x"], vectors=[[0.1]], metadatas=[{}])
    assert store.count() == 1
    store.add_chunks(texts=["y", "z"], vectors=[[0.2], [0.3]], metadatas=[{}, {}])
    assert store.count() == 3
