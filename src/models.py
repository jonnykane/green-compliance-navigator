from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DetectedContext:
    company_size: Literal["large", "sme", "unknown"] = "unknown"
    quoted_or_listed: Literal["yes", "no", "unknown"] = "unknown"
    fca_regulated: Literal["yes", "no", "unknown"] = "unknown"
    public_procurement: Literal["yes", "no", "unknown"] = "unknown"
    eu_operations: Literal["yes", "no", "unknown"] = "unknown"


@dataclass
class ClassificationResult:
    state: Literal["clear", "needs_clarification", "out_of_scope", "partial_answer_needs_clarification"]
    reason: str
    detected_context: DetectedContext = field(default_factory=DetectedContext)
    missing_fields: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    kind: Literal["answer", "clarification_needed", "out_of_scope"]
    answer: str | None = None
    clarification_question: str | None = None
    options: list[str] | None = None
    sources: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
