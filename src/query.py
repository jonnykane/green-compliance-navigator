from typing import Protocol

from src.classifier.classifier import CLARIFICATION_OPTIONS, CLARIFICATION_QUESTION
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


class GeneratorProtocol(Protocol):
    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        company_context: str = "",
    ) -> Answer: ...


class ClassifierProtocol(Protocol):
    def classify(self, question: str) -> ClassificationResult: ...



class QueryEngine:
    def __init__(
        self,
        retriever: RetrieverProtocol,
        generator: GeneratorProtocol,
        classifier: ClassifierProtocol,
    ):
        self._retriever = retriever
        self._generator = generator
        self._classifier = classifier

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
