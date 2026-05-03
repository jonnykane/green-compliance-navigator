from dataclasses import dataclass
from typing import Protocol

from src.generator.generator import Answer
from src.retriever.retriever import RetrievedChunk


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str) -> list[RetrievedChunk]: ...


class GeneratorProtocol(Protocol):
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer: ...


@dataclass
class QueryResponse:
    answer: str
    sources: list[str]
    chunks_used: int


class QueryEngine:
    def __init__(self, retriever: RetrieverProtocol, generator: GeneratorProtocol):
        self._retriever = retriever
        self._generator = generator

    def ask(self, question: str) -> QueryResponse:
        chunks = self._retriever.retrieve(question)
        answer = self._generator.generate(query=question, chunks=chunks)
        return QueryResponse(
            answer=answer.text,
            sources=answer.sources,
            chunks_used=answer.chunks_used,
        )
