"""
Pinecone-backed vector store implementing the same protocol as VectorStoreClient.

The PineconeVectorStore satisfies both:
  - ingestion_pipeline.StoreProtocol  (add_chunks, count)
  - retriever.StoreProtocol           (query → list[SearchResult])

so it can be swapped in wherever VectorStoreClient is used without
modifying QueryEngine, Retriever, or IngestionPipeline.

Document text is stored inside Pinecone metadata under the reserved key
``_text``, because Pinecone vectors do not have a native document field.
The key is stripped from the metadata dict before results are returned,
keeping the SearchResult.metadata shape identical to the ChromaDB path.

Pinecone returns similarity *scores* (higher = better); the Retriever
expects *distances* (lower = better).  We convert:  distance = 1 − score.
"""
import uuid
from typing import Any, Protocol

from src.vector_store.vector_store_client import SearchResult

_TEXT_KEY = "_text"
_UPSERT_BATCH_SIZE = 100  # Pinecone's recommended max vectors per request (keeps payload < 4 MB)


# ---------------------------------------------------------------------------
# Protocols — minimal surface of the Pinecone SDK we depend on
# ---------------------------------------------------------------------------

class _PineconeQueryResponseProtocol(Protocol):
    @property
    def matches(self) -> list[Any]: ...


class _PineconeIndexProtocol(Protocol):
    def upsert(self, vectors: list[dict[str, Any]]) -> None: ...

    def delete(self, delete_all: bool = False) -> None: ...

    def query(
        self,
        vector: list[float],
        top_k: int,
        include_metadata: bool,
    ) -> _PineconeQueryResponseProtocol: ...

    def describe_index_stats(self) -> Any: ...


class PineconeClientProtocol(Protocol):
    def Index(self, name: str) -> _PineconeIndexProtocol: ...


# ---------------------------------------------------------------------------
# PineconeVectorStore
# ---------------------------------------------------------------------------

class PineconeVectorStore:
    """Pinecone-backed vector store compatible with VectorStoreClient's protocol."""

    def __init__(self, pinecone_client: PineconeClientProtocol, index_name: str) -> None:
        self._index = pinecone_client.Index(index_name)

    # ------------------------------------------------------------------
    # ingestion_pipeline.StoreProtocol
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        if not (len(texts) == len(vectors) == len(metadatas)):
            raise ValueError("texts, vectors, and metadatas must have equal length")

        self._index.delete(delete_all=True)

        if not texts:
            return

        records = [
            {
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {_TEXT_KEY: text, **meta},
            }
            for text, vector, meta in zip(texts, vectors, metadatas)
        ]
        # Upsert in batches to stay within Pinecone's 4 MB per-request limit.
        for i in range(0, len(records), _UPSERT_BATCH_SIZE):
            self._index.upsert(vectors=records[i : i + _UPSERT_BATCH_SIZE])

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.total_vector_count

    # ------------------------------------------------------------------
    # retriever.StoreProtocol
    # ------------------------------------------------------------------

    def query(self, vector: list[float], n_results: int = 5) -> list[SearchResult]:
        response = self._index.query(
            vector=vector,
            top_k=n_results,
            include_metadata=True,
        )
        results: list[SearchResult] = []
        for match in response.matches:
            meta = dict(match.metadata)
            text = meta.pop(_TEXT_KEY, "")
            # Pinecone score is cosine similarity (0–1, higher = closer).
            # Retriever converts: final_score = 1 − distance, so we invert here.
            distance = 1.0 - match.score
            results.append(SearchResult(text=text, metadata=meta, distance=distance))
        return results
