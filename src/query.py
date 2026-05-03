from dataclasses import dataclass
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


# Kept for backwards compatibility — not returned by ask() any more.
@dataclass
class QueryResponse:
    answer: str
    sources: list[str]
    chunks_used: int


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

        # "clear" or "needs_clarification" with context provided — retrieve and generate.
        retrieval_query = (
            f"{question} {company_context}".strip() if company_context else question
        )
        chunks = self._retriever.retrieve(retrieval_query)
        answer = self._generator.generate(
            query=question,
            chunks=chunks,
            company_context=company_context,
        )
        return QueryResult(
            kind="answer",
            answer=answer.text,
            sources=answer.sources,
        )
