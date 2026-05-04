import pytest
from src.agent.rechunker import Rechunker, RechunkResult, ChunkingConfig


# ------------------------------------------------------------------ #
# Fakes
# ------------------------------------------------------------------ #

class FakeEmbedder:
    def __init__(self, dimension: int = 4):
        self._dim = dimension
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(i)] * self._dim for i in range(len(texts))]


class FakePineconeIndex:
    def __init__(self):
        self.upserted: list[dict] = []
        self.deleted_ids: list[str] = []

    def upsert(self, vectors: list[dict], namespace: str) -> None:
        for v in vectors:
            v["namespace"] = namespace
        self.upserted.extend(vectors)

    def delete(self, ids: list[str], namespace: str) -> None:
        self.deleted_ids.extend(ids)


# ------------------------------------------------------------------ #
# Chunking
# ------------------------------------------------------------------ #

class TestChunking:
    def _rechunker(self, config=None):
        return Rechunker(
            embedder=FakeEmbedder(),
            index=FakePineconeIndex(),
            config=config or ChunkingConfig(),
        )

    def test_short_content_produces_single_chunk(self):
        rechunker = self._rechunker()
        result = rechunker.rechunk("Short text.", framework_id="SECR")
        assert result.chunk_count == 1

    def test_long_content_produces_multiple_chunks(self):
        config = ChunkingConfig(chunk_size=50, overlap=10)
        rechunker = self._rechunker(config=config)
        content = "word " * 100  # 500 chars
        result = rechunker.rechunk(content, framework_id="SECR")
        assert result.chunk_count > 1

    def test_overlap_means_chunks_share_content(self):
        config = ChunkingConfig(chunk_size=40, overlap=10)
        rechunker = Rechunker(
            embedder=FakeEmbedder(),
            index=FakePineconeIndex(),
            config=config,
        )
        content = "A" * 40 + "B" * 10 + "C" * 40
        chunks = rechunker._split(content)
        assert len(chunks) >= 2
        # Last chars of chunk 0 should appear at start of chunk 1
        assert chunks[0][-config.overlap:] == chunks[1][:config.overlap]

    def test_empty_content_raises(self):
        rechunker = self._rechunker()
        with pytest.raises(ValueError, match="empty"):
            rechunker.rechunk("", framework_id="SECR")


# ------------------------------------------------------------------ #
# Embedding
# ------------------------------------------------------------------ #

class TestEmbedding:
    def test_embedder_called_with_all_chunks(self):
        embedder = FakeEmbedder()
        config = ChunkingConfig(chunk_size=20, overlap=0)
        rechunker = Rechunker(embedder=embedder, index=FakePineconeIndex(), config=config)
        rechunker.rechunk("word " * 20, framework_id="SECR")
        total_chunks = sum(len(call) for call in embedder.calls)
        assert total_chunks > 0

    def test_embedding_batch_size_respected(self):
        embedder = FakeEmbedder()
        config = ChunkingConfig(chunk_size=10, overlap=0, embed_batch_size=2)
        rechunker = Rechunker(embedder=embedder, index=FakePineconeIndex(), config=config)
        rechunker.rechunk("word " * 30, framework_id="SECR")
        # Each batch call should have at most 2 items
        for call in embedder.calls:
            assert len(call) <= 2


# ------------------------------------------------------------------ #
# Pinecone upsert
# ------------------------------------------------------------------ #

class TestPineconeUpsert:
    def _run(self, content="Some content about SECR reporting obligations.", framework_id="SECR"):
        index = FakePineconeIndex()
        rechunker = Rechunker(
            embedder=FakeEmbedder(),
            index=index,
            config=ChunkingConfig(),
        )
        result = rechunker.rechunk(content, framework_id=framework_id)
        return result, index

    def test_vectors_upserted_to_correct_namespace(self):
        _, index = self._run(framework_id="ESOS")
        namespaces = {v["namespace"] for v in index.upserted}
        assert namespaces == {"ESOS"}

    def test_upserted_vector_count_matches_chunk_count(self):
        result, index = self._run()
        assert len(index.upserted) == result.chunk_count

    def test_vector_ids_include_framework_id(self):
        result, index = self._run(framework_id="TCFD")
        for v in index.upserted:
            assert "TCFD" in v["id"]

    def test_vectors_have_metadata(self):
        _, index = self._run(framework_id="SECR")
        for v in index.upserted:
            assert "framework_id" in v["metadata"]
            assert "chunk_index" in v["metadata"]

    def test_result_contains_vector_ids(self):
        result, _ = self._run()
        assert len(result.vector_ids) == result.chunk_count

    def test_previous_vectors_deleted_before_upsert(self):
        index = FakePineconeIndex()
        config = ChunkingConfig()
        rechunker = Rechunker(embedder=FakeEmbedder(), index=index, config=config)
        existing_ids = ["SECR_0", "SECR_1"]
        rechunker.rechunk(
            "New content.",
            framework_id="SECR",
            previous_vector_ids=existing_ids,
        )
        assert set(existing_ids).issubset(set(index.deleted_ids))


# ------------------------------------------------------------------ #
# Result type
# ------------------------------------------------------------------ #

class TestRechunkResult:
    def test_result_carries_framework_id(self):
        result = RechunkResult(framework_id="SECR", chunk_count=3, vector_ids=["a", "b", "c"])
        assert result.framework_id == "SECR"

    def test_chunk_count_matches_vector_ids_length(self):
        result = RechunkResult(framework_id="SECR", chunk_count=3, vector_ids=["a", "b", "c"])
        assert result.chunk_count == len(result.vector_ids)
