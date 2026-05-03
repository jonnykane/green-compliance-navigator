"""
Factory functions that wire concrete implementations together using settings.
Nothing here should be imported by tests.
"""
import anthropic
import chromadb
import voyageai

from src.chunker.chunker import Chunker
from src.config import settings
from src.embedder.embedding_client import EmbeddingClient
from src.generator.generator import Generator
from src.loader.document_loader import DocumentLoader
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.query import QueryEngine
from src.retriever.retriever import Retriever
from src.vector_store.vector_store_client import VectorStoreClient


def build_ingestion_pipeline() -> IngestionPipeline:
    voyage = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
    embedder = EmbeddingClient(
        voyage_client=voyage,
        model=settings.VOYAGE_MODEL,
        batch_size=settings.VOYAGE_BATCH_SIZE,
        request_interval=settings.VOYAGE_REQUEST_INTERVAL,
    )
    chroma = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
    store = VectorStoreClient(
        chroma_client=chroma,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )
    return IngestionPipeline(
        loader=DocumentLoader(manifest_path=settings.CORPUS_METADATA_DIR / "real_corpus_manifest.json"),
        chunker=Chunker(
            max_tokens=settings.CHUNK_MAX_TOKENS,
            overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
        ),
        embedder=embedder,
        store=store,
    )


def build_query_engine() -> QueryEngine:
    voyage = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
    embedder = EmbeddingClient(
        voyage_client=voyage,
        model=settings.VOYAGE_MODEL,
        batch_size=settings.VOYAGE_BATCH_SIZE,
    )
    chroma = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
    store = VectorStoreClient(
        chroma_client=chroma,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )
    retriever = Retriever(embedder=embedder, store=store, top_k=5)
    generator = Generator(
        anthropic_client=anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY),
        model=settings.ANTHROPIC_MODEL,
    )
    return QueryEngine(retriever=retriever, generator=generator)
