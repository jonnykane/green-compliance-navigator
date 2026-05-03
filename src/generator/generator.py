from dataclasses import dataclass, field
from typing import Any, Protocol

from src.retriever.retriever import RetrievedChunk


class AnthropicClientProtocol(Protocol):
    @property
    def messages(self) -> Any: ...


@dataclass
class Answer:
    text: str
    sources: list[str]
    chunks_used: int


_SYSTEM_PROMPT = """\
You are an expert UK green compliance adviser. You answer questions about UK \
environmental and sustainability regulations accurately and concisely.

You will be given a set of numbered source excerpts retrieved from regulatory \
documents. Use ONLY these excerpts to answer the question. After your answer, \
list the sources you cited using their document names exactly as provided.

Rules:
- Cite every factual claim with the source document name in parentheses, e.g. (esos_overview.md).
- If the excerpts do not contain enough information to answer, say so clearly.
- Do not speculate beyond what the sources state.
- Keep answers focused and professional.
"""


class Generator:
    def __init__(self, anthropic_client: AnthropicClientProtocol, model: str):
        self._client = anthropic_client
        self._model = model

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        user_message = self._build_user_message(query, chunks)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = response.content[0].text
        sources = list(dict.fromkeys(c.source for c in chunks))  # deduplicated, ordered
        return Answer(text=answer_text, sources=sources, chunks_used=len(chunks))

    def _build_user_message(self, query: str, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        if chunks:
            parts.append("SOURCE EXCERPTS:")
            for i, chunk in enumerate(chunks, 1):
                parts.append(
                    f"[{i}] Document: {chunk.source} | Section: {chunk.section}\n{chunk.text}"
                )
            parts.append("")
        parts.append(f"QUESTION: {query}")
        return "\n\n".join(parts)
