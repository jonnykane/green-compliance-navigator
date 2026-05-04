"""
Factory functions that wire concrete implementations together using settings.
Nothing here should be imported by tests — except _build_vector_store,
which is tested directly to verify Pinecone/ChromaDB branching logic.
"""
import anthropic
import chromadb
import voyageai
from pinecone import Pinecone

from src.chunker.chunker import Chunker
from src.classifier.classifier import Classifier, CLASSIFIER_SYSTEM_PROMPT
from src.config import settings
from src.embedder.embedding_client import EmbeddingClient
from src.generator.generator import Generator
from src.loader.document_loader import DocumentLoader
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.query import QueryEngine
from src.retriever.retriever import Retriever
from src.vector_store.pinecone_vector_store import PineconeVectorStore
from src.vector_store.vector_store_client import VectorStoreClient


def _build_vector_store() -> VectorStoreClient | PineconeVectorStore:
    """
    Return PineconeVectorStore when PINECONE_API_KEY is set, otherwise
    return VectorStoreClient (ChromaDB-backed).
    """
    if settings.PINECONE_API_KEY:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        return PineconeVectorStore(pinecone_client=pc, index_name=settings.PINECONE_INDEX_NAME)
    chroma = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
    return VectorStoreClient(
        chroma_client=chroma,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )


class _AnthropicClassifierClient:
    """Adapts anthropic.Anthropic to ClassifierClientProtocol for the Classifier."""

    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=CLASSIFIER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


def build_ingestion_pipeline() -> IngestionPipeline:
    voyage = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
    embedder = EmbeddingClient(
        voyage_client=voyage,
        model=settings.VOYAGE_MODEL,
        batch_size=settings.VOYAGE_BATCH_SIZE,
        request_interval=settings.VOYAGE_REQUEST_INTERVAL,
    )
    store = _build_vector_store()
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
    store = _build_vector_store()
    retriever = Retriever(embedder=embedder, store=store, top_k=settings.RETRIEVER_TOP_K)
    anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    generator = Generator(
        anthropic_client=anthropic_client,
        model=settings.ANTHROPIC_MODEL,
    )
    classifier = Classifier(
        client=_AnthropicClassifierClient(
            client=anthropic_client,
            model=settings.ANTHROPIC_MODEL,
        )
    )
    return QueryEngine(
        retriever=retriever,
        generator=generator,
        classifier=classifier,
        canonical_retriever=retriever,
        top_k=settings.RETRIEVER_TOP_K,
    )
