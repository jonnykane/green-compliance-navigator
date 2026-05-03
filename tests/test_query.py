import pytest
from src.query import QueryEngine, QueryResponse
from src.retriever.retriever import RetrievedChunk
from src.generator.generator import Answer


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]):
        self._chunks = chunks
        self.last_query: str = ""

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.last_query = query
        return self._chunks


class FakeGenerator:
    def __init__(self, answer: Answer):
        self._answer = answer
        self.last_query: str = ""
        self.last_chunks: list[RetrievedChunk] = []

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        self.last_query = query
        self.last_chunks = chunks
        return self._answer


def _chunks(n: int = 3) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text=f"chunk {i}", source=f"doc{i}.md",
            section="S", doc_type="mock", score=0.9 - 0.1 * i,
        )
        for i in range(n)
    ]


def _answer() -> Answer:
    return Answer(text="ESOS applies to large organisations.", sources=["esos.md"], chunks_used=3)


# ---------------------------------------------------------------------------
# QueryResponse
# ---------------------------------------------------------------------------

def test_query_response_fields():
    qr = QueryResponse(answer="text", sources=["a.md"], chunks_used=2)
    assert qr.answer == "text"
    assert qr.sources == ["a.md"]
    assert qr.chunks_used == 2


# ---------------------------------------------------------------------------
# QueryEngine
# ---------------------------------------------------------------------------

def test_ask_returns_query_response():
    engine = QueryEngine(retriever=FakeRetriever(_chunks()), generator=FakeGenerator(_answer()))
    result = engine.ask("What is ESOS?")
    assert isinstance(result, QueryResponse)


def test_ask_passes_query_to_retriever():
    retriever = FakeRetriever(_chunks())
    engine = QueryEngine(retriever=retriever, generator=FakeGenerator(_answer()))
    engine.ask("ESOS compliance threshold")
    assert retriever.last_query == "ESOS compliance threshold"


def test_ask_passes_chunks_to_generator():
    chunks = _chunks(4)
    retriever = FakeRetriever(chunks)
    generator = FakeGenerator(_answer())
    engine = QueryEngine(retriever=retriever, generator=generator)
    engine.ask("q")
    assert generator.last_chunks == chunks


def test_ask_passes_query_to_generator():
    generator = FakeGenerator(_answer())
    engine = QueryEngine(retriever=FakeRetriever(_chunks()), generator=generator)
    engine.ask("My question")
    assert generator.last_query == "My question"


def test_ask_answer_text_from_generator():
    engine = QueryEngine(retriever=FakeRetriever(_chunks()), generator=FakeGenerator(_answer()))
    result = engine.ask("q")
    assert result.answer == "ESOS applies to large organisations."


def test_ask_sources_from_generator():
    engine = QueryEngine(retriever=FakeRetriever(_chunks()), generator=FakeGenerator(_answer()))
    result = engine.ask("q")
    assert result.sources == ["esos.md"]


def test_ask_chunks_used_from_generator():
    engine = QueryEngine(retriever=FakeRetriever(_chunks(3)), generator=FakeGenerator(_answer()))
    result = engine.ask("q")
    assert result.chunks_used == 3
