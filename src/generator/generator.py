import re
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


_DEFAULT_SYSTEM_PROMPT = """\
You are a UK green regulation compliance assistant. You help \
businesses understand what sustainability regulations apply to them.

Answer the question using only the provided source excerpts. \
Every factual claim must cite the specific regulation name and \
section from the source. Do not infer obligations not stated \
in the sources.

If the sources do not contain enough information to answer \
fully, state clearly which obligations can be assessed from \
the available context and which cannot be determined without \
additional information.

{company_context}"""

# Matches the placeholder and the blank line that precedes it so that
# removing it when empty leaves no double-blank gap.
_CONTEXT_PLACEHOLDER_RE = re.compile(r"\n\n\{company_context\}")


class Generator:
    def __init__(
        self,
        anthropic_client: AnthropicClientProtocol,
        model: str,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ):
        self._client = anthropic_client
        self._model = model
        self._system_prompt = system_prompt

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        company_context: str = "",
    ) -> Answer:
        system = self._render_system_prompt(company_context)
        user_message = self._build_user_message(query, chunks)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = response.content[0].text
        sources = list(dict.fromkeys(c.source for c in chunks))
        return Answer(text=answer_text, sources=sources, chunks_used=len(chunks))

    def _render_system_prompt(self, company_context: str) -> str:
        ctx = company_context.strip()
        if ctx:
            return self._system_prompt.replace("{company_context}", ctx)
        # Remove placeholder and the preceding blank line to avoid triple newlines.
        return _CONTEXT_PLACEHOLDER_RE.sub("", self._system_prompt)

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
