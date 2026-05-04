from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Protocol


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 512
    overlap: int = 64
    embed_batch_size: int = 32


# ------------------------------------------------------------------ #
# Result
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class RechunkResult:
    framework_id: str
    chunk_count: int
    vector_ids: list[str]


# ------------------------------------------------------------------ #
# Protocols (injectable)
# ------------------------------------------------------------------ #

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class PineconeIndex(Protocol):
    def upsert(self, vectors: list[dict], namespace: str) -> None:
        ...

    def delete(self, ids: list[str], namespace: str) -> None:
        ...


# ------------------------------------------------------------------ #
# Rechunker
# ------------------------------------------------------------------ #

def _chunk_id(framework_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"{framework_id}_{chunk_index}_{digest}"


class Rechunker:
    def __init__(
        self,
        embedder: Embedder,
        index: PineconeIndex,
        config: ChunkingConfig,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._config = config

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def rechunk(
        self,
        content: str,
        *,
        framework_id: str,
        previous_vector_ids: Optional[list[str]] = None,
    ) -> RechunkResult:
        if not content.strip():
            raise ValueError("content must not be empty")

        chunks = self._split(content)
        embeddings = self._embed_batched(chunks)

        vectors = [
            {
                "id": _chunk_id(framework_id, i, chunk),
                "values": embeddings[i],
                "metadata": {
                    "framework_id": framework_id,
                    "chunk_index": i,
                    "text": chunk,
                },
            }
            for i, chunk in enumerate(chunks)
        ]

        if previous_vector_ids:
            self._index.delete(ids=previous_vector_ids, namespace=framework_id)

        self._index.upsert(vectors=vectors, namespace=framework_id)

        return RechunkResult(
            framework_id=framework_id,
            chunk_count=len(vectors),
            vector_ids=[v["id"] for v in vectors],
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _split(self, text: str) -> list[str]:
        size = self._config.chunk_size
        overlap = self._config.overlap
        step = size - overlap
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + size])
            start += step
        return chunks

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        batch_size = self._config.embed_batch_size
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self._embedder.embed(batch))
        return results
