import json
from typing import Protocol

from src.models import ClassificationResult, DetectedContext


CLASSIFIER_SYSTEM_PROMPT = """\
You are a classifier for a UK green regulation compliance tool.

Analyse the question and return a JSON object with this exact structure:
{
  "state": "clear" | "needs_clarification" | "out_of_scope",
  "reason": "<one sentence>",
  "detected_context": {
    "company_size": "large" | "sme" | "unknown",
    "listing_or_regulated_status": "listed_or_fca_regulated" | "not_listed" | "unknown",
    "public_procurement": "yes" | "no" | "unknown",
    "eu_operations": "yes" | "no" | "unknown"
  },
  "missing_fields": ["<field_name>", ...]
}

Use "clear" if the question contains enough organisation context to retrieve relevant UK \
green regulations accurately. Sufficient context means at least company_size OR \
listing_or_regulated_status is known.

Use "needs_clarification" if the question is about UK green regulation or sustainability \
compliance but lacks organisation context that would materially affect which regulations apply.

Use "out_of_scope" if the question is not related to UK green regulation, sustainability \
reporting, or environmental compliance.

Return only valid JSON. No preamble, no markdown, no explanation outside the JSON object.\
"""

CLARIFICATION_QUESTION = (
    "To give you an accurate answer, which description is "
    "closest to your organisation?"
)

CLARIFICATION_OPTIONS = [
    "Large UK company — 250+ employees, or high turnover/balance sheet",
    "Listed or FCA-regulated — listed company, asset manager, "
    "insurer, or pension provider",
    "UK SME — under 250 employees and below large-company thresholds",
    "UK supplier bidding for public sector contracts",
    "UK company with significant EU operations — EU subsidiary, "
    "branch, or major EU turnover",
    "Not sure",
]

CONTEXT_FROM_OPTION = {
    0: "The user has confirmed they are a large UK company with "
       "250+ employees or £36m+ turnover or £18m+ balance sheet.",
    1: "The user has confirmed they are listed or FCA-regulated — "
       "a listed company, asset manager, insurer, or pension provider.",
    2: "The user has confirmed they are a UK SME with under 250 "
       "employees and below large-company thresholds.",
    3: "The user has confirmed they are bidding for UK public sector "
       "contracts above £5 million per year.",
    4: "The user has confirmed they have significant EU operations — "
       "an EU subsidiary, branch, or major EU turnover above €150m.",
    5: "The user is unsure which organisation category applies.",
}


class ClassifierClientProtocol(Protocol):
    def complete(self, prompt: str) -> str: ...


class Classifier:
    def __init__(self, client: ClassifierClientProtocol):
        self._client = client

    def classify(self, question: str) -> ClassificationResult:
        raw = self._client.complete(question)
        # Strip markdown code fences that some models add despite instructions.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]  # drop opening fence line
            cleaned = cleaned.rsplit("```", 1)[0]  # drop closing fence
        _VALID_STATES = {"clear", "needs_clarification", "out_of_scope"}
        try:
            data = json.loads(cleaned)
            state = data["state"]
            if state not in _VALID_STATES:
                raise ValueError(
                    f"Failed to parse classifier response: unexpected state {state!r} "
                    f"(expected one of {sorted(_VALID_STATES)})"
                )
            reason = data["reason"]
            ctx_data = data.get("detected_context", {})
            detected_context = DetectedContext(
                company_size=ctx_data.get("company_size", "unknown"),
                listing_or_regulated_status=ctx_data.get(
                    "listing_or_regulated_status", "unknown"
                ),
                public_procurement=ctx_data.get("public_procurement", "unknown"),
                eu_operations=ctx_data.get("eu_operations", "unknown"),
            )
            missing_fields = data.get("missing_fields", [])
            return ClassificationResult(
                state=state,
                reason=reason,
                detected_context=detected_context,
                missing_fields=missing_fields,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Failed to parse classifier response: {exc!r}\nRaw response: {raw!r}"
            ) from exc
