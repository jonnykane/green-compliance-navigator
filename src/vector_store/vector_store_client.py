import uuid
from dataclasses import dataclass
from typing import Any, Protocol


class ChromaCollectionProtocol(Protocol):
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None: ...

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict: ...

    def count(self) -> int: ...


class ChromaClientProtocol(Protocol):
    def get_or_create_collection(self, name: str) -> ChromaCollectionProtocol: ...


@dataclass
class SearchResult:
    text: str
    metadata: dict[str, Any]
    distance: float


class VectorStoreClient:
    def __init__(self, chroma_client: ChromaClientProtocol, collection_name: str):
        self._collection = chroma_client.get_or_create_collection(collection_name)

    def add_chunks(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        if not texts:
            return
        if not (len(texts) == len(vectors) == len(metadatas)):
            raise ValueError("texts, vectors, and metadatas must have equal length")
        ids = [str(uuid.uuid4()) for _ in texts]
        self._collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, vector: list[float], n_results: int = 5) -> list[SearchResult]:
        response = self._collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        results: list[SearchResult] = []
        docs = response["documents"][0]
        metas = response["metadatas"][0]
        dists = response["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            results.append(SearchResult(text=doc, metadata=meta, distance=dist))
        return results

    def count(self) -> int:
        return self._collection.count()
