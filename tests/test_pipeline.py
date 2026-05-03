import pytest
from pathlib import Path
from unittest.mock import MagicMock, call

from src.loader.document_loader import Document
from src.chunker.chunker import Chunk
from src.embedder.embedding_client import EmbeddingResult
from src.pipeline.ingestion_pipeline import IngestionPipeline, IngestionResult


# ---------------------------------------------------------------------------
# Fakes / stubs
# ---------------------------------------------------------------------------

class FakeLoader:
    def __init__(self, docs: list[Document]):
        self._docs = docs

    def load_directory(self, directory: Path, file_type: str) -> list[Document]:
        return self._docs


class FakeChunker:
    def __init__(self, chunks_per_doc: int = 2):
        self._n = chunks_per_doc

    def chunk(self, doc: Document) -> list[Chunk]:
        return [
            Chunk(
                text=f"{doc.source} chunk {i}",
                source=doc.source,
                doc_type=doc.doc_type,
                section=f"Section {i}",
                chunk_index=i,
                metadata=dict(doc.metadata),
            )
            for i in range(self._n)
        ]


class FakeEmbedder:
    DIM = 4

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        return [EmbeddingResult(text=t, vector=[0.1] * self.DIM) for t in texts]


class FakeVectorStore:
    def __init__(self):
        self.added_texts: list[str] = []
        self.added_vectors: list[list[float]] = []
        self.added_metadatas: list[dict] = []

    def add_chunks(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self.added_texts.extend(texts)
        self.added_vectors.extend(vectors)
        self.added_metadatas.extend(metadatas)

    def count(self) -> int:
        return len(self.added_texts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_docs(n: int = 2) -> list[Document]:
    return [
        Document(
            content=f"# Section\n\nContent for doc {i}.",
            source=f"doc{i}.md",
            doc_type="mock",
            metadata={"issuing_body": f"Body {i}"},
        )
        for i in range(n)
    ]


@pytest.fixture
def pipeline_components():
    docs = _make_docs(2)
    loader = FakeLoader(docs)
    chunker = FakeChunker(chunks_per_doc=2)
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    return loader, chunker, embedder, store


# ---------------------------------------------------------------------------
# IngestionResult
# ---------------------------------------------------------------------------

def test_ingestion_result_fields():
    result = IngestionResult(docs_loaded=4, chunks_created=20, vectors_stored=20)
    assert result.docs_loaded == 4
    assert result.chunks_created == 20
    assert result.vectors_stored == 20


# ---------------------------------------------------------------------------
# IngestionPipeline
# ---------------------------------------------------------------------------

def test_pipeline_runs_without_error(pipeline_components, tmp_path):
    loader, chunker, embedder, store = pipeline_components
    pipeline = IngestionPipeline(loader=loader, chunker=chunker, embedder=embedder, store=store)
    result = pipeline.run(directories=[(tmp_path, "md")])
    assert isinstance(result, IngestionResult)


def test_pipeline_reports_correct_doc_count(pipeline_components, tmp_path):
    loader, chunker, embedder, store = pipeline_components
    pipeline = IngestionPipeline(loader=loader, chunker=chunker, embedder=embedder, store=store)
    result = pipeline.run(directories=[(tmp_path, "md")])
    assert result.docs_loaded == 2


def test_pipeline_reports_correct_chunk_count(pipeline_components, tmp_path):
    loader, chunker, embedder, store = pipeline_components
    pipeline = IngestionPipeline(loader=loader, chunker=chunker, embedder=embedder, store=store)
    result = pipeline.run(directories=[(tmp_path, "md")])
    # 2 docs * 2 chunks each
    assert result.chunks_created == 4


def test_pipeline_vectors_stored_equals_chunks(pipeline_components, tmp_path):
    loader, chunker, embedder, store = pipeline_components
    pipeline = IngestionPipeline(loader=loader, chunker=chunker, embedder=embedder, store=store)
    result = pipeline.run(directories=[(tmp_path, "md")])
    assert result.vectors_stored == result.chunks_created


def test_pipeline_stores_chunk_text(pipeline_components, tmp_path):
    loader, chunker, embedder, store = pipeline_components
    pipeline = IngestionPipeline(loader=loader, chunker=chunker, embedder=embedder, store=store)
    pipeline.run(directories=[(tmp_path, "md")])
    assert len(store.added_texts) == 4


def test_pipeline_metadata_includes_source_and_section(pipeline_components, tmp_path):
    loader, chunker, embedder, store = pipeline_components
    pipeline = IngestionPipeline(loader=loader, chunker=chunker, embedder=embedder, store=store)
    pipeline.run(directories=[(tmp_path, "md")])
    for meta in store.added_metadatas:
        assert "source" in meta
        assert "section" in meta


def test_pipeline_metadata_carries_doc_metadata(pipeline_components, tmp_path):
    loader, chunker, embedder, store = pipeline_components
    pipeline = IngestionPipeline(loader=loader, chunker=chunker, embedder=embedder, store=store)
    pipeline.run(directories=[(tmp_path, "md")])
    bodies = {m.get("issuing_body") for m in store.added_metadatas}
    assert "Body 0" in bodies
    assert "Body 1" in bodies


def test_pipeline_handles_multiple_directories(tmp_path):
    docs_a = [Document(content="# A\nDoc A", source="a.md", doc_type="mock")]
    docs_b = [Document(content="Doc B text", source="b.pdf", doc_type="real")]

    class MultiLoader:
        def load_directory(self, directory: Path, file_type: str) -> list[Document]:
            if file_type == "md":
                return docs_a
            return docs_b

    store = FakeVectorStore()
    pipeline = IngestionPipeline(
        loader=MultiLoader(),
        chunker=FakeChunker(chunks_per_doc=1),
        embedder=FakeEmbedder(),
        store=store,
    )
    result = pipeline.run(directories=[(tmp_path, "md"), (tmp_path, "pdf")])
    assert result.docs_loaded == 2


def test_pipeline_empty_corpus_returns_zero_counts(tmp_path):
    loader = FakeLoader([])
    pipeline = IngestionPipeline(
        loader=loader,
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        store=FakeVectorStore(),
    )
    result = pipeline.run(directories=[(tmp_path, "md")])
    assert result.docs_loaded == 0
    assert result.chunks_created == 0
    assert result.vectors_stored == 0


# ---------------------------------------------------------------------------
# Contextual enrichment: embed enriched text, store original
# ---------------------------------------------------------------------------

class CapturingEmbedder:
    """Records the exact texts passed to embed_texts for assertion."""
    DIM = 4

    def __init__(self):
        self.embedded_texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        self.embedded_texts.extend(texts)
        return [EmbeddingResult(text=t, vector=[0.1] * self.DIM) for t in texts]


def _doc_with_context(context: str) -> Document:
    return Document(
        content="# Section\n\nContent.",
        source="secr.pdf",
        doc_type="real",
        metadata={"doc_context": context},
    )


def test_pipeline_embeds_enriched_text_when_doc_context_present(tmp_path):
    context = "SECR mandatory UK sustainability reporting for large companies"
    loader = FakeLoader([_doc_with_context(context)])
    embedder = CapturingEmbedder()
    store = FakeVectorStore()
    pipeline = IngestionPipeline(
        loader=loader,
        chunker=FakeChunker(chunks_per_doc=1),
        embedder=embedder,
        store=store,
    )
    pipeline.run(directories=[(tmp_path, "pdf")])
    # Every embedded text must start with the context prefix
    assert all(t.startswith(context) for t in embedder.embedded_texts)


def test_pipeline_stores_original_chunk_text_not_enriched(tmp_path):
    context = "SECR mandatory UK sustainability reporting"
    loader = FakeLoader([_doc_with_context(context)])
    embedder = CapturingEmbedder()
    store = FakeVectorStore()
    pipeline = IngestionPipeline(
        loader=loader,
        chunker=FakeChunker(chunks_per_doc=1),
        embedder=embedder,
        store=store,
    )
    pipeline.run(directories=[(tmp_path, "pdf")])
    # Stored texts must NOT contain the context prefix
    assert all(not t.startswith(context) for t in store.added_texts)


def test_pipeline_embeds_original_text_when_no_doc_context(tmp_path):
    docs = _make_docs(1)  # no doc_context in metadata
    loader = FakeLoader(docs)
    embedder = CapturingEmbedder()
    store = FakeVectorStore()
    pipeline = IngestionPipeline(
        loader=loader,
        chunker=FakeChunker(chunks_per_doc=1),
        embedder=embedder,
        store=store,
    )
    pipeline.run(directories=[(tmp_path, "md")])
    # Without context, embedded text equals stored text
    assert embedder.embedded_texts == store.added_texts


def test_pipeline_doc_with_no_chunks_is_skipped(tmp_path):
    """Documents that produce zero chunks don't cause errors."""
    docs = [Document(content="", source="empty.md", doc_type="mock")]
    loader = FakeLoader(docs)
    store = FakeVectorStore()
    pipeline = IngestionPipeline(
        loader=loader,
        chunker=FakeChunker(chunks_per_doc=0),
        embedder=FakeEmbedder(),
        store=store,
    )
    result = pipeline.run(directories=[(tmp_path, "md")])
    assert result.chunks_created == 0
    assert store.count() == 0
