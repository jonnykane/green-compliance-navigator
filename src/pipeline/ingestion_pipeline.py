from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.chunker.chunker import Chunk, Chunker
from src.embedder.embedding_client import EmbeddingClient, EmbeddingResult
from src.loader.document_loader import Document, DocumentLoader
from src.vector_store.vector_store_client import VectorStoreClient


class LoaderProtocol(Protocol):
    def load_directory(self, directory: Path, file_type: str) -> list[Document]: ...


class ChunkerProtocol(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


class EmbedderProtocol(Protocol):
    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]: ...


class StoreProtocol(Protocol):
    def add_chunks(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None: ...

    def count(self) -> int: ...


@dataclass
class IngestionResult:
    docs_loaded: int
    chunks_created: int
    vectors_stored: int


class IngestionPipeline:
    def __init__(
        self,
        loader: LoaderProtocol,
        chunker: ChunkerProtocol,
        embedder: EmbedderProtocol,
        store: StoreProtocol,
    ):
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    @staticmethod
    def _enrich_for_embedding(chunk: Chunk) -> str:
        context = chunk.metadata.get("doc_context", "")
        return f"{context}\n\n{chunk.text}" if context else chunk.text

    def run(self, directories: list[tuple[Path, str]]) -> IngestionResult:
        all_docs: list[Document] = []
        for directory, file_type in directories:
            all_docs.extend(self._loader.load_directory(directory, file_type))

        all_chunks: list[Chunk] = []
        for doc in all_docs:
            all_chunks.extend(self._chunker.chunk(doc))

        if not all_chunks:
            return IngestionResult(
                docs_loaded=len(all_docs),
                chunks_created=0,
                vectors_stored=0,
            )

        texts = [c.text for c in all_chunks]
        texts_for_embedding = [self._enrich_for_embedding(c) for c in all_chunks]
        embedding_results = self._embedder.embed_texts(texts_for_embedding)
        vectors = [er.vector for er in embedding_results]
        metadatas = [
            {
                "source": c.source,
                "doc_type": c.doc_type,
                "section": c.section,
                "chunk_index": c.chunk_index,
                **c.metadata,
            }
            for c in all_chunks
        ]

        self._store.add_chunks(texts=texts, vectors=vectors, metadatas=metadatas)

        return IngestionResult(
            docs_loaded=len(all_docs),
            chunks_created=len(all_chunks),
            vectors_stored=len(all_chunks),
        )
