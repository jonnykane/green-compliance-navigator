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
You are a UK green regulation compliance assistant. You help
businesses understand what sustainability regulations apply to them.

The indexed corpus covers: SECR, ESOS, TCFD, FCA mandatory TCFD
rules (PS21/24), CSRD (UK applicability), UK SDS, and PPN 06/21.

Answer the question using only the provided source excerpts.
Every factual regulatory claim must be grounded in the provided
source excerpts and cite the regulation or source label available
in the excerpt. Where a section number or paragraph reference is
available in the excerpt, include it. Do not invent section
references that are not present in the source material.

NAMED REGULATION RULE: If the question references a specific
named law, regulation, act, or policy that does not appear in
the retrieved source excerpts, do not attempt to answer from
adjacent material. State clearly that this named item is not
found in the indexed corpus and suggest the user may be thinking
of a related indexed regulation if relevant.

PARTIAL ANSWER RULE: If the company context note indicates that
fields are missing, provide a substantive partial answer first
using the available context, then ask specifically for only the
missing facts needed to complete the assessment. Do not respond
with clarification only — always answer what can be answered first.

EXHAUSTIVE FACT ENUMERATION RULE: When answering threshold or
applicability questions, you must surface every specific figure,
criterion, and condition present in the retrieved source excerpts.
Do not summarise or omit any numeric threshold, employee count,
turnover figure, or balance sheet value that appears in the
source material. If the sources list three thresholds, state all
three. If the sources give an employee count AND a turnover figure
AND a balance sheet limit, every one of these must appear in your
answer — never drop figures to simplify the response.

NO SELECTIVE REPORTING RULE: When multiple conditions must ALL be
met for a regulation to apply (conjunctive criteria), you must
state every condition explicitly. Do not name only the most
prominent condition and omit the rest. For example, if a
regulation applies only when a company exceeds both an employee
threshold AND a financial threshold, both must be stated clearly
as joint requirements.

REQUIRED CAVEATS RULE: When retrieved source excerpts contain
qualifications such as "subject to consultation", "indicative
timeline only", "expected but not yet confirmed", "proposed",
or similar hedging language, you must reproduce those
qualifications in your answer. Do not silently convert a
proposed or uncertain obligation into a definitive one.

NUMERIC FACT EXTRACTION RULE: When retrieved chunks contain
specific numeric values (employee counts, turnover figures,
balance sheet totals, energy thresholds, contract values,
dates), every such figure must appear explicitly in the answer.
Do not summarise numeric facts — enumerate them individually.
Example: if chunks contain "250 or more employees",
"£36 million turnover", and "£18 million balance sheet total",
all three must appear in the answer.

CONJUNCTIVE CRITERIA RULE: When retrieved chunks describe
criteria where multiple conditions apply (e.g. "two of three
criteria must be met", "AND", "OR" conditions), the answer
must explicitly state how many criteria must be satisfied and
list each criterion separately. Do not collapse conjunctive
conditions into a single summary statement.

FALSE PREMISE CORRECTION RULE: When the question contains a
factual premise that contradicts the retrieved evidence (e.g.
"Does SECR apply to companies with more than 500 employees?"
when the threshold is actually 250), the answer must explicitly
correct the false premise before answering. State what the
actual threshold or rule is, and note that the figure in the
question is incorrect.

If the sources do not contain enough information to answer
fully, state clearly which obligations can be assessed from
the available context and which cannot be determined without
additional information.

{company_context}

Sources:
{chunks}

Question: {question}"""

# Matches the placeholder and the blank line that precedes it so that
# removing it when empty leaves no double-blank gap.
_CONTEXT_PLACEHOLDER_RE = re.compile(r"\n\n\{company_context\}")

# Strips the Sources/chunks/Question section — those belong in the user message.
_CHUNKS_QUESTION_RE = re.compile(r"\n\nSources:\n\{chunks\}\n\nQuestion: \{question\}")


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
            temperature=0,
        )
        answer_text = response.content[0].text
        sources = list(dict.fromkeys(c.source for c in chunks))
        return Answer(text=answer_text, sources=sources, chunks_used=len(chunks))

    def _render_system_prompt(self, company_context: str) -> str:
        # Remove Sources/chunks/Question section — those are rendered in the user message.
        system = _CHUNKS_QUESTION_RE.sub("", self._system_prompt)
        ctx = company_context.strip()
        if ctx:
            return system.replace("{company_context}", ctx)
        # Remove placeholder and the preceding blank line to avoid triple newlines.
        return _CONTEXT_PLACEHOLDER_RE.sub("", system)

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
