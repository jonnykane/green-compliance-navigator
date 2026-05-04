from dataclasses import replace as _dataclass_replace
from typing import Protocol

from src.classifier.classifier import CLARIFICATION_OPTIONS, CLARIFICATION_QUESTION
from src.config import settings
from src.generator.generator import Answer
from src.models import ClassificationResult, QueryResult
from src.retriever.retriever import RetrievedChunk

_OUT_OF_SCOPE_MESSAGE = (
    "This tool covers UK green regulation and sustainability compliance. "
    "Your question does not appear to be in that area. "
    "Please rephrase or ask about a specific regulation."
)

# ---------------------------------------------------------------------------
# Retrieval-query enrichment
# ---------------------------------------------------------------------------

# Vocabulary appended to queries that mention a specific regulation keyword.
# The extra terms close the gap between natural-language questions and the
# technical language embedded in policy document chunks.
_REGULATION_VOCAB: dict[str, str] = {
    "secr": (
        "large unquoted companies LLPs Directors Report "
        "250 employees £36m turnover £18m balance sheet "
        "qualifying thresholds energy carbon reporting"
    ),
    "streamlined energy": (
        "large companies 250 employees £36m turnover Directors Report"
    ),
}

# Canonical source trigger map — loaded from settings so this module stays data-free.
CANONICAL_SOURCE_MAP: dict[str, str] = settings.CANONICAL_SOURCE_MAP


def _expand_parent_sections(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Replace chunk text with its full parent section where available.

    Retrieval precision is preserved — the original chunk text and score are
    unchanged in the QueryResult. Only the copy sent to the generator is
    expanded, so the generator sees column headers and labels that may live
    outside the matched sub-chunk.
    """
    result: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.parent_section_text:
            result.append(RetrievedChunk(
                text=chunk.parent_section_text,
                source=chunk.source,
                section=chunk.section,
                doc_type=chunk.doc_type,
                score=chunk.score,
            ))
        else:
            result.append(chunk)
    return result


def _enrich_retrieval_query(question: str) -> str:
    """Return *question* with regulation-specific vocabulary appended.

    If the question (case-insensitively) contains any keyword in
    ``_REGULATION_VOCAB`` the corresponding vocabulary string is appended.
    Questions that match no keyword are returned unchanged.
    """
    q_lower = question.lower()
    extra_terms: list[str] = []
    for keyword, vocab in _REGULATION_VOCAB.items():
        if keyword in q_lower:
            extra_terms.append(vocab)
    if extra_terms:
        return f"{question} {' '.join(extra_terms)}"
    return question


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str) -> list[RetrievedChunk]: ...


class CanonicalRetrieverProtocol(Protocol):
    def retrieve_by_source(self, query: str, source: str) -> list[RetrievedChunk]: ...


class GeneratorProtocol(Protocol):
    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        company_context: str = "",
    ) -> Answer: ...


class ClassifierProtocol(Protocol):
    def classify(self, question: str) -> ClassificationResult: ...


def _inject_canonical_sources(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    canonical_retriever: CanonicalRetrieverProtocol,
    source_map: dict[str, str],
    top_k: int,
) -> list[RetrievedChunk]:
    """Guarantee that trigger-matched canonical sources appear in retrieved chunks.

    Triggers are matched against the original user question only (case-insensitive).
    Injection is skipped when the canonical source is already represented.
    When the chunk list is at top_k capacity the lowest-scoring non-canonical
    chunk is replaced so the injected chunk is never discarded.
    """
    q_lower = question.lower()
    injected = list(chunks)

    for trigger, source in source_map.items():
        if trigger not in q_lower:
            continue

        # Already represented — skip.
        if any(c.source == source for c in injected):
            continue

        canonical_chunks = canonical_retriever.retrieve_by_source(question, source)
        if not canonical_chunks:
            continue

        tagged = [
            _dataclass_replace(
                c,
                retrieval_reason="canonical_source_injection",
                canonical_trigger=trigger,
            )
            for c in canonical_chunks
        ]

        for canonical_chunk in tagged:
            if len(injected) < top_k:
                injected.append(canonical_chunk)
            else:
                # Replace the lowest-scoring non-canonical chunk.
                non_canonical = [
                    (i, c) for i, c in enumerate(injected)
                    if c.retrieval_reason != "canonical_source_injection"
                ]
                if not non_canonical:
                    break
                min_idx, _ = min(non_canonical, key=lambda x: x[1].score)
                injected[min_idx] = canonical_chunk

    return injected


class QueryEngine:
    def __init__(
        self,
        retriever: RetrieverProtocol,
        generator: GeneratorProtocol,
        classifier: ClassifierProtocol,
        canonical_retriever: CanonicalRetrieverProtocol | None = None,
        top_k: int = settings.RETRIEVER_TOP_K,
    ):
        self._retriever = retriever
        self._generator = generator
        self._classifier = classifier
        self._canonical_retriever = canonical_retriever
        self._top_k = top_k

    def ask(self, question: str, company_context: str = "") -> QueryResult:
        classification = self._classifier.classify(question)

        if classification.state == "out_of_scope":
            return QueryResult(kind="out_of_scope", answer=_OUT_OF_SCOPE_MESSAGE)

        if classification.state == "needs_clarification" and not company_context:
            return QueryResult(
                kind="clarification_needed",
                clarification_question=CLARIFICATION_QUESTION,
                options=CLARIFICATION_OPTIONS,
            )

        # Augment company_context with missing-fields note for partial answers.
        if (
            classification.state == "partial_answer_needs_clarification"
            and classification.missing_fields
        ):
            note = (
                "Note: the following context is missing and would affect the "
                f"completeness of this answer: {', '.join(classification.missing_fields)}"
            )
            company_context = f"{company_context}\n{note}".strip() if company_context else note

        # "clear", "partial_answer_needs_clarification", or "needs_clarification" with
        # context provided — retrieve and generate.
        base_query = (
            f"{question} {company_context}".strip() if company_context else question
        )
        retrieval_query = _enrich_retrieval_query(base_query)
        chunks = self._retriever.retrieve(retrieval_query)

        if self._canonical_retriever is not None:
            chunks = _inject_canonical_sources(
                question,
                chunks,
                canonical_retriever=self._canonical_retriever,
                source_map=CANONICAL_SOURCE_MAP,
                top_k=self._top_k,
            )

        gen_chunks = _expand_parent_sections(chunks)
        answer = self._generator.generate(
            query=question,
            chunks=gen_chunks,
            company_context=company_context,
        )
        return QueryResult(
            kind="answer",
            answer=answer.text,
            sources=answer.sources,
            retrieved_chunks=[
                {"source": c.source, "text": c.text, "score": c.score}
                for c in chunks
            ],
        )
