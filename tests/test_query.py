import pytest
from src.query import QueryEngine, _enrich_retrieval_query, _expand_parent_sections, _inject_canonical_sources
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


# ---------------------------------------------------------------------------
# _expand_parent_sections — standalone function tests
# ---------------------------------------------------------------------------

def test_expand_parent_sections_output_length_equals_input_length_with_parents():
    """Expansion must replace each chunk — never append — so len(out) == len(in)."""
    full_section = "Heading\n\nFull body text with all details."
    chunks = [
        RetrievedChunk(
            text="Full body text", source="doc.pdf", section="Heading",
            doc_type="real", score=0.9, parent_section_text=full_section,
        ),
        RetrievedChunk(
            text="other snippet", source="doc2.pdf", section="S2",
            doc_type="real", score=0.8, parent_section_text=full_section,
        ),
    ]
    result = _expand_parent_sections(chunks)
    assert len(result) == len(chunks)


def test_expand_parent_sections_output_length_equals_input_length_without_parents():
    """Pass-through chunks (no parent_section_text) must not be duplicated."""
    chunks = _chunks(5)
    result = _expand_parent_sections(chunks)
    assert len(result) == len(chunks)


def test_expand_parent_sections_replaces_text_with_parent_text():
    """Each chunk with parent_section_text must have its text replaced by the parent text."""
    full_section = "2. Thresholds\n\n250 employees or £36m turnover."
    chunk = RetrievedChunk(
        text="250 employees", source="secr.pdf", section="Thresholds",
        doc_type="real", score=0.9, parent_section_text=full_section,
    )
    result = _expand_parent_sections([chunk])
    assert result[0].text == full_section


def test_expand_parent_sections_passthrough_chunk_is_same_object():
    """Chunks without parent_section_text must be passed through as-is."""
    chunk = RetrievedChunk(
        text="no parent here", source="doc.pdf", section="S",
        doc_type="real", score=0.7,
    )
    result = _expand_parent_sections([chunk])
    assert result[0] is chunk


def test_expand_parent_sections_mixed_chunks():
    """Mixed input: expanded chunks replace text; pass-through chunks are unchanged."""
    full_section = "Full section text."
    chunk_with_parent = RetrievedChunk(
        text="fragment", source="a.pdf", section="S", doc_type="real",
        score=0.9, parent_section_text=full_section,
    )
    chunk_without_parent = RetrievedChunk(
        text="standalone", source="b.pdf", section="T", doc_type="mock", score=0.8,
    )
    result = _expand_parent_sections([chunk_with_parent, chunk_without_parent])
    assert len(result) == 2
    assert result[0].text == full_section
    assert result[1].text == "standalone"


# ---------------------------------------------------------------------------
# Canonical source injection — fakes
# ---------------------------------------------------------------------------

_SECR_SUMMARY_SOURCE = "secr_guidelines_summary.md"

_CANONICAL_SOURCE_MAP: dict[str, str] = {
    "secr": _SECR_SUMMARY_SOURCE,
    "streamlined energy": _SECR_SUMMARY_SOURCE,
    "streamlined carbon": _SECR_SUMMARY_SOURCE,
}


def _secr_summary_chunk(score: float = 0.75) -> RetrievedChunk:
    return RetrievedChunk(
        text="SECR threshold: 250 employees or £36m turnover or £18m balance sheet.",
        source=_SECR_SUMMARY_SOURCE,
        section="Thresholds",
        doc_type="real",
        score=score,
    )


class FakeCanonicalRetriever:
    """Returns a preset chunk when retrieve_by_source matches, empty list otherwise."""

    def __init__(self, chunks_by_source: dict[str, list[RetrievedChunk]]):
        self._chunks = chunks_by_source
        self.calls: list[tuple[str, str]] = []

    def retrieve_by_source(self, query: str, source: str) -> list[RetrievedChunk]:
        self.calls.append((query, source))
        return list(self._chunks.get(source, []))


def _make_non_secr_chunks(n: int = 3, base_score: float = 0.9) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text=f"chunk {i}", source=f"doc{i}.pdf",
            section="S", doc_type="real", score=base_score - 0.05 * i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _inject_canonical_sources — standalone function tests
# ---------------------------------------------------------------------------

def test_inject_secr_question_adds_summary_when_absent():
    chunks = _make_non_secr_chunks(2)
    canonical = FakeCanonicalRetriever({_SECR_SUMMARY_SOURCE: [_secr_summary_chunk()]})
    result = _inject_canonical_sources(
        "What is SECR and who does it apply to?",
        chunks,
        canonical_retriever=canonical,
        source_map=_CANONICAL_SOURCE_MAP,
        top_k=8,
    )
    sources = [c.source for c in result]
    assert _SECR_SUMMARY_SOURCE in sources


def test_inject_secr_summary_already_present_no_duplicate():
    existing_summary = _secr_summary_chunk(score=0.8)
    chunks = _make_non_secr_chunks(1) + [existing_summary]
    canonical = FakeCanonicalRetriever({_SECR_SUMMARY_SOURCE: [_secr_summary_chunk()]})
    result = _inject_canonical_sources(
        "What is SECR?",
        chunks,
        canonical_retriever=canonical,
        source_map=_CANONICAL_SOURCE_MAP,
        top_k=8,
    )
    summary_count = sum(1 for c in result if c.source == _SECR_SUMMARY_SOURCE)
    assert summary_count == 1
    assert not canonical.calls  # canonical retriever must not be called


def test_inject_non_secr_question_does_not_inject():
    chunks = _make_non_secr_chunks(3)
    canonical = FakeCanonicalRetriever({_SECR_SUMMARY_SOURCE: [_secr_summary_chunk()]})
    result = _inject_canonical_sources(
        "What is ESOS and who does it apply to?",
        chunks,
        canonical_retriever=canonical,
        source_map=_CANONICAL_SOURCE_MAP,
        top_k=8,
    )
    sources = [c.source for c in result]
    assert _SECR_SUMMARY_SOURCE not in sources
    assert not canonical.calls


def test_inject_trigger_matching_is_case_insensitive():
    chunks = _make_non_secr_chunks(2)
    canonical = FakeCanonicalRetriever({_SECR_SUMMARY_SOURCE: [_secr_summary_chunk()]})
    result = _inject_canonical_sources(
        "WHAT IS SECR AND WHO DOES IT APPLY TO?",
        chunks,
        canonical_retriever=canonical,
        source_map=_CANONICAL_SOURCE_MAP,
        top_k=8,
    )
    sources = [c.source for c in result]
    assert _SECR_SUMMARY_SOURCE in sources


def test_inject_streamlined_energy_trigger_injects_secr_summary():
    chunks = _make_non_secr_chunks(2)
    canonical = FakeCanonicalRetriever({_SECR_SUMMARY_SOURCE: [_secr_summary_chunk()]})
    result = _inject_canonical_sources(
        "Who must comply with streamlined energy reporting?",
        chunks,
        canonical_retriever=canonical,
        source_map=_CANONICAL_SOURCE_MAP,
        top_k=8,
    )
    sources = [c.source for c in result]
    assert _SECR_SUMMARY_SOURCE in sources


def test_inject_at_top_k_capacity_replaces_lowest_non_canonical_chunk():
    # 3 chunks at top_k=3; injection must replace the lowest-scoring one.
    chunks = [
        RetrievedChunk(text="a", source="a.pdf", section="S", doc_type="real", score=0.9),
        RetrievedChunk(text="b", source="b.pdf", section="S", doc_type="real", score=0.8),
        RetrievedChunk(text="c", source="c.pdf", section="S", doc_type="real", score=0.5),  # lowest
    ]
    canonical = FakeCanonicalRetriever({_SECR_SUMMARY_SOURCE: [_secr_summary_chunk(score=0.75)]})
    result = _inject_canonical_sources(
        "What is SECR?",
        chunks,
        canonical_retriever=canonical,
        source_map=_CANONICAL_SOURCE_MAP,
        top_k=3,
    )
    assert len(result) == 3
    sources = [c.source for c in result]
    assert _SECR_SUMMARY_SOURCE in sources
    assert "c.pdf" not in sources  # lowest-scoring chunk was displaced


def test_inject_at_top_k_capacity_injected_chunk_remains():
    """The injected canonical chunk must survive — not be sliced off."""
    chunks = [
        RetrievedChunk(text="a", source="a.pdf", section="S", doc_type="real", score=0.9),
        RetrievedChunk(text="b", source="b.pdf", section="S", doc_type="real", score=0.8),
        RetrievedChunk(text="c", source="c.pdf", section="S", doc_type="real", score=0.5),
    ]
    canonical = FakeCanonicalRetriever({_SECR_SUMMARY_SOURCE: [_secr_summary_chunk(score=0.75)]})
    result = _inject_canonical_sources(
        "SECR thresholds",
        chunks,
        canonical_retriever=canonical,
        source_map=_CANONICAL_SOURCE_MAP,
        top_k=3,
    )
    injected = [c for c in result if c.retrieval_reason == "canonical_source_injection"]
    assert len(injected) == 1


def test_inject_marks_injected_chunk_with_retrieval_reason():
    chunks = _make_non_secr_chunks(1)
    canonical = FakeCanonicalRetriever({_SECR_SUMMARY_SOURCE: [_secr_summary_chunk()]})
    result = _inject_canonical_sources(
        "What is SECR?",
        chunks,
        canonical_retriever=canonical,
        source_map=_CANONICAL_SOURCE_MAP,
        top_k=8,
    )
    injected = next(c for c in result if c.source == _SECR_SUMMARY_SOURCE)
    assert injected.retrieval_reason == "canonical_source_injection"


def test_inject_marks_injected_chunk_with_canonical_trigger():
    chunks = _make_non_secr_chunks(1)
    canonical = FakeCanonicalRetriever({_SECR_SUMMARY_SOURCE: [_secr_summary_chunk()]})
    result = _inject_canonical_sources(
        "What is SECR?",
        chunks,
        canonical_retriever=canonical,
        source_map=_CANONICAL_SOURCE_MAP,
        top_k=8,
    )
    injected = next(c for c in result if c.source == _SECR_SUMMARY_SOURCE)
    assert injected.canonical_trigger == "secr"


# ---------------------------------------------------------------------------
# QueryEngine integration — canonical injection via ask()
# ---------------------------------------------------------------------------

def test_ask_with_canonical_retriever_injects_secr_summary():
    chunks = _make_non_secr_chunks(2)
    canonical = FakeCanonicalRetriever({_SECR_SUMMARY_SOURCE: [_secr_summary_chunk()]})
    engine = QueryEngine(
        retriever=FakeRetriever(chunks),
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
        canonical_retriever=canonical,
        top_k=8,
    )
    result = engine.ask("What is SECR and who does it apply to?")
    sources = [c["source"] for c in result.retrieved_chunks]
    assert _SECR_SUMMARY_SOURCE in sources


def test_ask_without_canonical_retriever_skips_injection():
    chunks = _make_non_secr_chunks(2)
    engine = QueryEngine(
        retriever=FakeRetriever(chunks),
        generator=FakeGenerator(_answer()),
        classifier=_clear_classifier(),
    )
    result = engine.ask("What is SECR and who does it apply to?")
    sources = [c["source"] for c in result.retrieved_chunks]
    assert _SECR_SUMMARY_SOURCE not in sources
