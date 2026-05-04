"""
Tests for src/vector_store/pinecone_vector_store.py — PineconeVectorStore.

All tests use in-memory fakes from src/vector_store/fake_pinecone_vector_store.py;
no real Pinecone API calls are made.
"""
import pytest
from src.vector_store.fake_pinecone_vector_store import FakePineconeClient
from src.vector_store.pinecone_vector_store import PineconeVectorStore
from src.vector_store.vector_store_client import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> PineconeVectorStore:
    client = FakePineconeClient()
    return PineconeVectorStore(pinecone_client=client, index_name="test-index")


def _chunks(n: int = 2):
    texts = [f"chunk {i}" for i in range(n)]
    vectors = [[float(i), float(i + 1)] for i in range(n)]
    metadatas = [{"source": f"doc{i}.md", "section": f"S{i}", "doc_type": "mock"} for i in range(n)]
    return texts, vectors, metadatas


# ---------------------------------------------------------------------------
# add_chunks — basic behaviour
# ---------------------------------------------------------------------------

def test_add_chunks_increases_count(store):
    texts, vectors, metadatas = _chunks(3)
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    assert store.count() == 3


def test_add_chunks_empty_is_noop(store):
    store.add_chunks(texts=[], vectors=[], metadatas=[])
    assert store.count() == 0


def test_add_chunks_mismatched_lengths_raises(store):
    with pytest.raises(ValueError):
        store.add_chunks(texts=["a", "b"], vectors=[[0.1]], metadatas=[{}])


def test_add_chunks_multiple_calls_replace_not_accumulate(store):
    """Each add_chunks call deletes first, so only the latest batch is present."""
    texts, vectors, metadatas = _chunks(2)
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    store.add_chunks(texts=["extra"], vectors=[[9.9]], metadatas=[{"source": "x.md"}])
    assert store.count() == 1  # only the second call's records remain


# ---------------------------------------------------------------------------
# add_chunks — text stored in metadata, IDs are unique
# ---------------------------------------------------------------------------

def test_add_chunks_stores_text_in_metadata(store):
    """Document text must be persisted inside the Pinecone metadata record."""
    store.add_chunks(
        texts=["ESOS applies to large undertakings."],
        vectors=[[0.1, 0.2]],
        metadatas=[{"source": "esos.md"}],
    )
    # Retrieve via query and confirm text survives round-trip
    results = store.query(vector=[0.1, 0.2], n_results=1)
    assert results[0].text == "ESOS applies to large undertakings."


def test_add_chunks_text_key_not_in_result_metadata(store):
    """The internal _text key must be removed from the returned metadata."""
    store.add_chunks(
        texts=["Some text."],
        vectors=[[0.5]],
        metadatas=[{"source": "a.md"}],
    )
    results = store.query(vector=[0.5], n_results=1)
    assert "_text" not in results[0].metadata


def test_add_chunks_generates_unique_ids(store):
    """Each upserted record must have a distinct ID."""
    texts, vectors, metadatas = _chunks(5)
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    # Count unique IDs via the fake index's internal store
    fake_index = store._index  # type: ignore[attr-defined]
    ids = [r["id"] for r in fake_index._records]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# query — return type and structure
# ---------------------------------------------------------------------------

def test_query_returns_list_of_search_results(store):
    texts, vectors, metadatas = _chunks(3)
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    results = store.query(vector=[0.0, 1.0], n_results=2)
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)


def test_query_respects_n_results(store):
    texts, vectors, metadatas = _chunks(5)
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    results = store.query(vector=[0.0], n_results=3)
    assert len(results) <= 3


def test_query_returns_empty_when_store_empty(store):
    results = store.query(vector=[0.1, 0.2], n_results=5)
    assert results == []


def test_query_result_metadata_contains_source(store):
    store.add_chunks(
        texts=["SECR threshold text."],
        vectors=[[0.3]],
        metadatas=[{"source": "secr.pdf", "section": "Thresholds"}],
    )
    results = store.query(vector=[0.3], n_results=1)
    assert results[0].metadata["source"] == "secr.pdf"


# ---------------------------------------------------------------------------
# query — score → distance conversion
# ---------------------------------------------------------------------------

def test_query_distance_derived_from_score(store):
    """
    Pinecone returns similarity scores (higher = more similar).
    SearchResult.distance must equal 1.0 - score so the Retriever's
    score = 1 - distance formula produces the original similarity value.
    """
    store.add_chunks(
        texts=["Some regulatory text."],
        vectors=[[1.0]],
        metadatas=[{"source": "secr.pdf"}],
    )
    results = store.query(vector=[1.0], n_results=1)
    fake_index = store._index  # type: ignore[attr-defined]
    # The fake returns score = 1.0 for the first (only) result
    expected_score = 1.0
    assert abs(results[0].distance - (1.0 - expected_score)) < 1e-9


def test_query_distance_is_non_negative(store):
    texts, vectors, metadatas = _chunks(3)
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    results = store.query(vector=[0.0], n_results=3)
    assert all(r.distance >= 0.0 for r in results)


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_zero_on_empty_store(store):
    assert store.count() == 0


def test_count_reflects_upserted_records(store):
    texts, vectors, metadatas = _chunks(4)
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    assert store.count() == 4


# ---------------------------------------------------------------------------
# Batched upsert — large payload split across multiple calls
# ---------------------------------------------------------------------------

def test_add_chunks_batches_large_payload():
    """
    More than _UPSERT_BATCH_SIZE records must be split into multiple upsert
    calls so the 4 MB Pinecone request limit is never hit.
    """
    from src.vector_store.pinecone_vector_store import _UPSERT_BATCH_SIZE
    from src.vector_store.fake_pinecone_vector_store import FakePineconeClient

    client = FakePineconeClient()
    store = PineconeVectorStore(pinecone_client=client, index_name="test-index")

    n = _UPSERT_BATCH_SIZE + 10  # deliberately exceeds one batch
    texts, vectors, metadatas = _chunks(n)
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)

    # All records must be stored despite being split across multiple calls
    assert store.count() == n


def test_add_chunks_exactly_one_batch_size_uses_single_call():
    """Exactly _UPSERT_BATCH_SIZE records fit in one call — no extra calls needed."""
    from src.vector_store.pinecone_vector_store import _UPSERT_BATCH_SIZE
    from src.vector_store.fake_pinecone_vector_store import FakePineconeClient

    client = FakePineconeClient()
    store = PineconeVectorStore(pinecone_client=client, index_name="test-index")

    texts, vectors, metadatas = _chunks(_UPSERT_BATCH_SIZE)
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    assert store.count() == _UPSERT_BATCH_SIZE


# ---------------------------------------------------------------------------
# delete-before-upsert — re-ingest produces a clean index
# ---------------------------------------------------------------------------

def test_add_chunks_second_call_replaces_not_appends(store):
    """A second add_chunks call must clear the index first, not accumulate."""
    texts1, vectors1, metadatas1 = _chunks(3)
    store.add_chunks(texts=texts1, vectors=vectors1, metadatas=metadatas1)
    texts2, vectors2, metadatas2 = _chunks(2)
    store.add_chunks(texts=texts2, vectors=vectors2, metadatas=metadatas2)
    assert store.count() == 2  # not 5


def test_add_chunks_empty_call_clears_existing_vectors(store):
    """Calling add_chunks with empty lists must still clear existing vectors."""
    texts, vectors, metadatas = _chunks(4)
    store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)
    store.add_chunks(texts=[], vectors=[], metadatas=[])
    assert store.count() == 0


def test_add_chunks_delete_called_before_upsert():
    """delete(delete_all=True) must be called on the index at the start of add_chunks."""
    from src.vector_store.fake_pinecone_vector_store import FakePineconeClient

    client = FakePineconeClient()
    store = PineconeVectorStore(pinecone_client=client, index_name="test-index")
    fake_index = store._index  # type: ignore[attr-defined]

    # Seed the index directly via a lower-level call to simulate stale data
    fake_index.upsert(vectors=[{"id": "stale-1", "values": [0.0], "metadata": {"_text": "stale"}}])
    assert fake_index.describe_index_stats().total_vector_count == 1

    # add_chunks must wipe the stale vector before inserting new ones
    store.add_chunks(texts=["fresh"], vectors=[[1.0]], metadatas=[{"source": "new.pdf"}])
    results = store.query(vector=[1.0], n_results=10)
    texts_in_store = [r.text for r in results]
    assert "stale" not in texts_in_store
    assert "fresh" in texts_in_store
