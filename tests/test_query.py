import pytest
from src.query import QueryEngine, _enrich_retrieval_query
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


# ---------------------------------------------------------------------------
# QueryEngine — partial_answer_needs_clarification routing
# ---------------------------------------------------------------------------

def _partial_classifier(missing_fields: list[str] | None = None) -> FakeClassifier:
    if missing_fields is None:
        missing_fields = ["quoted_or_listed"]
    return FakeClassifier(
        ClassificationResult(
            state="partial_answer_needs_clarification",
            reason="Enough for SECR/ESOS but listing status unknown.",
            missing_fields=missing_fields,
        )
    )


def test_partial_state_returns_answer_kind():
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=_partial_classifier(),
    )
    result = engine.ask("We have 300 employees and £45m turnover.")
    assert result.kind == "answer"


def test_partial_state_reaches_retriever():
    retriever = FakeRetriever(_chunks())
    engine = QueryEngine(
        retriever=retriever,
        generator=FakeGenerator(_answer()),
        classifier=_partial_classifier(),
    )
    engine.ask("We have 300 employees and £45m turnover.")
    assert retriever.last_query != ""


def test_partial_state_reaches_generator():
    generator = FakeGenerator(_answer())
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=generator,
        classifier=_partial_classifier(),
    )
    engine.ask("We have 300 employees and £45m turnover.")
    assert generator.last_query == "We have 300 employees and £45m turnover."


def test_partial_state_missing_fields_note_in_generator_context():
    """When missing_fields is non-empty the generator receives the augmented context note."""
    generator = FakeGenerator(_answer())
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=generator,
        classifier=_partial_classifier(missing_fields=["quoted_or_listed", "eu_operations"]),
    )
    engine.ask("We have 300 employees and £45m turnover.")
    assert "Note: the following context is missing" in generator.last_company_context
    assert "quoted_or_listed" in generator.last_company_context
    assert "eu_operations" in generator.last_company_context


def test_partial_state_missing_fields_note_appended_to_existing_context():
    """Missing-fields note is appended after any existing company_context string."""
    generator = FakeGenerator(_answer())
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=generator,
        classifier=_partial_classifier(missing_fields=["quoted_or_listed"]),
    )
    engine.ask(
        "We have 300 employees and £45m turnover.",
        company_context="The user has confirmed they are a large UK company.",
    )
    ctx = generator.last_company_context
    assert "The user has confirmed they are a large UK company." in ctx
    assert "Note: the following context is missing" in ctx
    assert ctx.index("The user has confirmed") < ctx.index("Note:")


def test_partial_state_no_missing_fields_note_when_missing_fields_empty():
    """When missing_fields is empty no note is appended to company_context."""
    generator = FakeGenerator(_answer())
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=generator,
        classifier=_partial_classifier(missing_fields=[]),
    )
    engine.ask("We have 300 employees and £45m turnover.")
    assert "Note: the following context is missing" not in generator.last_company_context


# ---------------------------------------------------------------------------
# _enrich_retrieval_query — standalone function tests
# ---------------------------------------------------------------------------

def test_enrich_retrieval_query_no_keyword_returns_unchanged():
    """Questions with no regulation keyword must pass through unmodified."""
    q = "We have 300 employees — what must we report?"
    assert _enrich_retrieval_query(q) == q


def test_enrich_retrieval_query_esos_only_returns_unchanged():
    """'ESOS' alone (no 'secr' / 'streamlined energy') must not be enriched."""
    q = "ESOS compliance threshold"
    assert _enrich_retrieval_query(q) == q


def test_enrich_retrieval_query_secr_lowercase_triggers_enrichment():
    """Lower-case 'secr' in the question must trigger vocabulary expansion."""
    result = _enrich_retrieval_query("what is secr and who must comply?")
    assert len(result) > len("what is secr and who must comply?")


def test_enrich_retrieval_query_secr_uppercase_triggers_enrichment():
    """Upper-case 'SECR' must also trigger vocabulary expansion (case-insensitive)."""
    result = _enrich_retrieval_query("What is SECR and who does it apply to?")
    assert len(result) > len("What is SECR and who does it apply to?")


def test_enrich_retrieval_query_secr_result_contains_llp_terms():
    """Enriched SECR query must contain qualifying-threshold vocabulary."""
    result = _enrich_retrieval_query("What is SECR?")
    lower = result.lower()
    assert "llp" in lower or "250 employees" in lower or "£36m" in lower or "directors report" in lower


def test_enrich_retrieval_query_secr_result_contains_original_question():
    """Original question text must be preserved at the start of the enriched query."""
    q = "What is SECR and who does it apply to?"
    result = _enrich_retrieval_query(q)
    assert result.startswith(q)


def test_enrich_retrieval_query_streamlined_energy_triggers_enrichment():
    """'streamlined energy' keyword must trigger vocabulary expansion."""
    q = "What is streamlined energy reporting?"
    result = _enrich_retrieval_query(q)
    assert len(result) > len(q)


def test_enrich_retrieval_query_streamlined_energy_result_contains_original():
    """Original question preserved when enriching via 'streamlined energy' keyword."""
    q = "What is streamlined energy reporting?"
    result = _enrich_retrieval_query(q)
    assert result.startswith(q)


# ---------------------------------------------------------------------------
# QueryEngine integration — enriched retrieval query
# ---------------------------------------------------------------------------

def test_ask_retriever_receives_enriched_query_for_secr_question():
    """Retriever must see the expanded vocabulary when the question mentions SECR."""
    retriever = FakeRetriever(_chunks())
    engine = QueryEngine(
        retriever=retriever,
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    engine.ask("What is SECR and who does it apply to?")
    lower = retriever.last_query.lower()
    assert "llp" in lower or "250 employees" in lower or "£36m" in lower or "directors report" in lower


def test_ask_retriever_query_still_contains_original_secr_question():
    """Original question must remain present in the (enriched) retrieval query."""
    retriever = FakeRetriever(_chunks())
    engine = QueryEngine(
        retriever=retriever,
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    q = "What is SECR and who does it apply to?"
    engine.ask(q)
    assert q in retriever.last_query


# ---------------------------------------------------------------------------
# QueryEngine — retrieved_chunks in QueryResult
# ---------------------------------------------------------------------------

def test_ask_answer_result_has_retrieved_chunks():
    """QueryResult for an answer must carry the retrieved chunks."""
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks(3)),
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    result = engine.ask("What is ESOS?")
    assert isinstance(result.retrieved_chunks, list)
    assert len(result.retrieved_chunks) == 3


def test_ask_retrieved_chunks_have_source_text_score_keys():
    """Each retrieved chunk dict must contain source, text, and score."""
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks(2)),
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    result = engine.ask("What is ESOS?")
    for chunk in result.retrieved_chunks:
        assert "source" in chunk
        assert "text" in chunk
        assert "score" in chunk


def test_ask_retrieved_chunks_carry_correct_values():
    """Chunk dict values must match the RetrievedChunk fields from the retriever."""
    chunks = _chunks(1)
    engine = QueryEngine(
        retriever=FakeRetriever(chunks),
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    result = engine.ask("What is ESOS?")
    assert result.retrieved_chunks[0]["source"] == chunks[0].source
    assert result.retrieved_chunks[0]["text"] == chunks[0].text
    assert result.retrieved_chunks[0]["score"] == chunks[0].score


def test_ask_out_of_scope_has_empty_retrieved_chunks():
    """out_of_scope responses never retrieve — retrieved_chunks must be empty."""
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=_out_of_scope_classifier(),
    )
    result = engine.ask("How do I make pasta?")
    assert result.retrieved_chunks == []


def test_ask_clarification_needed_has_empty_retrieved_chunks():
    """clarification_needed responses never retrieve — retrieved_chunks must be empty."""
    engine = QueryEngine(
        retriever=FakeRetriever(_chunks()),
        generator=FakeGenerator(_answer()),
        classifier=_needs_clarification_classifier(),
    )
    result = engine.ask("What sustainability reporting do we need to do?")
    assert result.retrieved_chunks == []


# ---------------------------------------------------------------------------
# QueryEngine — parent-section expansion for generation
# ---------------------------------------------------------------------------

def _chunk_with_parent(parent_text: str) -> RetrievedChunk:
    return RetrievedChunk(
        text="250 employees",
        source="secr.pdf",
        section="Thresholds",
        doc_type="real",
        score=0.9,
        parent_section_text=parent_text,
    )


def test_ask_generator_receives_expanded_text_when_parent_section_text_set():
    """Generator must see the full parent section text, not just the matched chunk text."""
    full_section = "2. Thresholds\n\n250 employees or £36m turnover or £18m balance sheet."
    chunk = _chunk_with_parent(full_section)
    generator = FakeGenerator(_answer())
    engine = QueryEngine(
        retriever=FakeRetriever([chunk]),
        generator=generator,
        classifier=_clear_classifier(),
    )
    engine.ask("SECR thresholds")
    assert len(generator.last_chunks) == 1
    assert generator.last_chunks[0].text == full_section


def test_ask_retrieved_chunks_in_result_keep_original_text():
    """QueryResult.retrieved_chunks must carry the original (pre-expansion) chunk text."""
    full_section = "2. Thresholds\n\n250 employees or £36m turnover or £18m balance sheet."
    chunk = _chunk_with_parent(full_section)
    engine = QueryEngine(
        retriever=FakeRetriever([chunk]),
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    result = engine.ask("SECR thresholds")
    assert result.retrieved_chunks[0]["text"] == "250 employees"


def test_ask_generator_receives_original_text_when_no_parent_section_text():
    """Chunks without parent_section_text must pass through to the generator unchanged."""
    generator = FakeGenerator(_answer())
    chunks = _chunks(2)
    engine = QueryEngine(
        retriever=FakeRetriever(chunks),
        generator=generator,
        classifier=_clear_classifier(),
    )
    engine.ask("q")
    assert generator.last_chunks[0].text == chunks[0].text
    assert generator.last_chunks[1].text == chunks[1].text
