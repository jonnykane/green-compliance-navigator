from dataclasses import dataclass
from typing import Protocol

from src.vector_store.vector_store_client import SearchResult


class EmbedderProtocol(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class StoreProtocol(Protocol):
    def query(self, vector: list[float], n_results: int) -> list[SearchResult]: ...


@dataclass
class RetrievedChunk:
    text: str
    source: str
    section: str
    doc_type: str
    score: float


class Retriever:
    def __init__(self, embedder: EmbedderProtocol, store: StoreProtocol, top_k: int = 5):
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        vector = self._embedder.embed_query(query)
        raw = self._store.query(vector=vector, n_results=self._top_k)
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
        )
