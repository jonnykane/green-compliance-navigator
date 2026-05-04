"""
In-memory fake for the Pinecone index and client, for use in tests.

Mirrors the duck-typed surface of the Pinecone Python SDK (v3+):
  - FakePineconeClient  ≈  pinecone.Pinecone
  - FakePineconeIndex   ≈  the object returned by Pinecone().Index(name)

No real network calls are made; all state lives in Python memory.
"""
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Return-value types that mirror the Pinecone SDK response objects
# ---------------------------------------------------------------------------

@dataclass
class _FakeMatch:
    id: str
    score: float
    metadata: dict[str, Any]


@dataclass
class _FakeQueryResponse:
    matches: list[_FakeMatch]


@dataclass
class _FakeIndexStats:
    total_vector_count: int


# ---------------------------------------------------------------------------
# FakePineconeIndex — mirrors pinecone.Index
# ---------------------------------------------------------------------------

class FakePineconeIndex:
    """
    In-memory imitation of a Pinecone index.

    upsert() stores records keyed by ID (overwrites on collision, matching
    real Pinecone behaviour).  query() returns the first top_k stored
    records with fake descending scores — similarity ordering is not
    simulated; tests should assert on structure, not ranking.
    """

    def __init__(self) -> None:
        # Ordered list so iteration is deterministic in tests.
        self._records: list[dict[str, Any]] = []
        self._ids: set[str] = set()

    def upsert(self, vectors: list[dict[str, Any]]) -> None:
        for record in vectors:
            record_id = record["id"]
            if record_id in self._ids:
                # Overwrite existing record (Pinecone upsert semantics)
                self._records = [r for r in self._records if r["id"] != record_id]
                self._ids.discard(record_id)
            self._records.append(record)
            self._ids.add(record_id)

    def query(
        self,
        vector: list[float],
        top_k: int,
        include_metadata: bool,
    ) -> _FakeQueryResponse:
        n = min(top_k, len(self._records))
        matches = [
            _FakeMatch(
                id=r["id"],
                # Fake scores: first result gets 1.0, each subsequent drops by 0.1
                score=max(0.0, 1.0 - 0.1 * i),
                metadata=dict(r.get("metadata", {})),
            )
            for i, r in enumerate(self._records[:n])
        ]
        return _FakeQueryResponse(matches=matches)

    def describe_index_stats(self) -> _FakeIndexStats:
        return _FakeIndexStats(total_vector_count=len(self._records))


# ---------------------------------------------------------------------------
# FakePineconeClient — mirrors pinecone.Pinecone
# ---------------------------------------------------------------------------

class FakePineconeClient:
    """
    In-memory imitation of the Pinecone client.

    Calling .Index(name) returns (or creates) a FakePineconeIndex for that
    name, so multiple indexes can coexist independently in the same test.
    """

    def __init__(self) -> None:
        self._indexes: dict[str, FakePineconeIndex] = {}

    def Index(self, name: str) -> FakePineconeIndex:
        if name not in self._indexes:
            self._indexes[name] = FakePineconeIndex()
        return self._indexes[name]
