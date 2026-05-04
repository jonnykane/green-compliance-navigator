from dataclasses import dataclass
from typing import Any, Protocol

from src.vector_store.vector_store_client import SearchResult


class EmbedderProtocol(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class StoreProtocol(Protocol):
    def query(self, vector: list[float], n_results: int) -> list[SearchResult]: ...


class FilterableStoreProtocol(Protocol):
    def query_filtered(
        self, vector: list[float], n_results: int, filter: dict[str, Any]
    ) -> list[SearchResult]: ...


@dataclass
class RetrievedChunk:
    text: str
    source: str
    section: str
    doc_type: str
    score: float
    parent_section_text: str = ""
    retrieval_reason: str = ""
    canonical_trigger: str = ""


class Retriever:
    def __init__(self, embedder: EmbedderProtocol, store: StoreProtocol, top_k: int = 5):
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        vector = self._embedder.embed_query(query)
        raw = self._store.query(vector=vector, n_results=self._top_k)
        return [self._to_chunk(r) for r in raw]

    def retrieve_by_source(self, query: str, source: str, n: int = 5) -> list[RetrievedChunk]:
        vector = self._embedder.embed_query(query)
        if hasattr(self._store, "query_filtered"):
            raw = self._store.query_filtered(  # type: ignore[union-attr]
                vector=vector,
                n_results=n,
                filter={"source": {"$eq": source}},
            )
        else:
            # Chroma fallback: retrieve a large window and filter client-side.
            raw = self._store.query(vector=vector, n_results=200)
            raw = [r for r in raw if r.metadata.get("source") == source][:n]
        return [self._to_chunk(r) for r in raw]

    def _to_chunk(self, result: SearchResult) -> RetrievedChunk:
        score = max(0.0, min(1.0, 1.0 - result.distance))
        meta = result.metadata
        return RetrievedChunk(
            text=result.text,
            source=meta.get("source", ""),
            section=meta.get("section", ""),
            doc_type=meta.get("doc_type", ""),
            score=score,
            parent_section_text=meta.get("parent_section_text", ""),
        )
