import pytest
from src.query import QueryEngine, QueryResponse
from src.models import ClassificationResult, DetectedContext, QueryResult
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
        self.last_company_context: str = ""

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        company_context: str = "",
    ) -> Answer:
        self.last_query = query
        self.last_chunks = chunks
        self.last_company_context = company_context
        return self._answer


class FakeClassifier:
    def __init__(self, result: ClassificationResult):
        self._result = result
        self.last_question: str = ""

    def classify(self, question: str) -> ClassificationResult:
        self.last_question = question
        return self._result


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


def _clear_classifier() -> FakeClassifier:
    return FakeClassifier(ClassificationResult(state="clear", reason="Has context."))


def _needs_clarification_classifier() -> FakeClassifier:
    return FakeClassifier(
        ClassificationResult(
            state="needs_clarification",
            reason="Missing company size.",
            missing_fields=["company_size"],
        )
    )


def _out_of_scope_classifier() -> FakeClassifier:
    return FakeClassifier(
        ClassificationResult(state="out_of_scope", reason="Unrelated question.")
    )


# ---------------------------------------------------------------------------
# QueryResponse (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def test_query_response_fields():
    qr = QueryResponse(answer="text", sources=["a.md"], chunks_used=2)
    assert qr.answer == "text"
    assert qr.sources == ["a.md"]
    assert qr.chunks_used == 2


# ---------------------------------------------------------------------------
# QueryEngine — original behaviour (clear path)
# ---------------------------------------------------------------------------

def test_ask_returns_query_result():
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    result = engine.ask("What is ESOS?")
    assert isinstance(result, QueryResult)


def test_ask_passes_query_to_retriever():
    retriever = FakeRetriever(_chunks())
    engine = QueryEngine(
        retriever=retriever,
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    engine.ask("ESOS compliance threshold")
    assert retriever.last_query == "ESOS compliance threshold"


def test_ask_passes_chunks_to_generator():
    chunks = _chunks(4)
    retriever = FakeRetriever(chunks)
    generator = FakeGenerator(_answer())
    engine = QueryEngine(
        retriever=retriever,
        generator=generator,
        classifier=_clear_classifier(),
    )
    engine.ask("q")
    assert generator.last_chunks == chunks


def test_ask_passes_query_to_generator():
    generator = FakeGenerator(_answer())
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=generator,
        classifier=_clear_classifier(),
    )
    engine.ask("My question")
    assert generator.last_query == "My question"


def test_ask_answer_text_from_generator():
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    result = engine.ask("q")
    assert result.answer == "ESOS applies to large organisations."


def test_ask_sources_from_generator():
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    result = engine.ask("q")
    assert result.sources == ["esos.md"]


# ---------------------------------------------------------------------------
# QueryEngine — four flow paths
# ---------------------------------------------------------------------------

def test_ask_clear_question_returns_answer_kind():
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    result = engine.ask("We have 300 employees — what reporting do we need?")
    assert result.kind == "answer"
    assert result.answer is not None


def test_ask_needs_clarification_without_context_returns_clarification_kind():
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=_needs_clarification_classifier(),
    )
    result = engine.ask("What sustainability reporting do we need to do?")
    assert result.kind == "clarification_needed"
    assert result.clarification_question is not None
    assert result.options is not None
    assert len(result.options) == 6


def test_ask_needs_clarification_with_company_context_returns_answer():
    """When needs_clarification but company_context is provided, proceed to answer."""
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=_needs_clarification_classifier(),
    )
    result = engine.ask(
        "What sustainability reporting do we need to do?",
        company_context="The user has confirmed they are a large UK company with 250+ employees or £36m+ turnover or £18m+ balance sheet.",
    )
    assert result.kind == "answer"


def test_ask_out_of_scope_returns_out_of_scope_kind():
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=_out_of_scope_classifier(),
    )
    result = engine.ask("How do I make pasta?")
    assert result.kind == "out_of_scope"
    assert result.answer is not None  # contains the out-of-scope message


def test_ask_out_of_scope_skips_retriever_and_generator():
    """Retriever and generator must not be called for out-of-scope questions."""
    retriever = FakeRetriever(_chunks())
    generator = FakeGenerator(_answer())
    engine = QueryEngine(
        retriever=retriever,
        generator=generator,
        classifier=_out_of_scope_classifier(),
    )
    engine.ask("How do I make pasta?")
    assert retriever.last_query == ""
    assert generator.last_query == ""


def test_ask_clarification_needed_skips_retriever_and_generator():
    """Retriever and generator must not be called when clarification is needed."""
    retriever = FakeRetriever(_chunks())
    generator = FakeGenerator(_answer())
    engine = QueryEngine(
        retriever=retriever,
        generator=generator,
        classifier=_needs_clarification_classifier(),
    )
    engine.ask("What do I need to do?")
    assert retriever.last_query == ""
    assert generator.last_query == ""


def test_ask_company_context_passed_to_generator():
    """Company context string must reach the generator when provided."""
    generator = FakeGenerator(_answer())
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=generator,
        classifier=_clear_classifier(),
    )
    engine.ask("q", company_context="The user is a large UK company.")
    assert generator.last_company_context == "The user is a large UK company."


def test_ask_classifier_receives_the_question():
    classifier = _clear_classifier()
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=classifier,
    )
    engine.ask("specific question about SECR")
    assert classifier.last_question == "specific question about SECR"


def test_ask_retriever_receives_augmented_query_when_company_context_provided():
    """Retriever must see both the question and company_context so embeddings match the right corpus."""
    retriever = FakeRetriever(_chunks())
    engine = QueryEngine(
        retriever=retriever,
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    engine.ask(
        "What sustainability reporting do we need to do?",
        company_context="The user has confirmed they are a large UK company with 250+ employees or £36m+ turnover or £18m+ balance sheet.",
    )
    assert "What sustainability reporting do we need to do?" in retriever.last_query
    assert "The user has confirmed they are a large UK company with 250+ employees or £36m+ turnover or £18m+ balance sheet." in retriever.last_query


def test_ask_retriever_receives_bare_question_when_no_company_context():
    """Without company_context the retrieval query must be exactly the original question."""
    retriever = FakeRetriever(_chunks())
    engine = QueryEngine(
        retriever=retriever,
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    engine.ask("We have 300 employees — what must we report?")
    assert retriever.last_query == "We have 300 employees — what must we report?"
