import json
import pytest

from src.classifier.classifier import (
    Classifier,
    CLARIFICATION_QUESTION,
    CLARIFICATION_OPTIONS,
    CONTEXT_FROM_OPTION,
)
from src.models import ClassificationResult, DetectedContext


# ---------------------------------------------------------------------------
# Fake classifier LLM client
# ---------------------------------------------------------------------------

class FakeClassifierClient:
    def __init__(self, response: str):
        self._response = response
        self.last_prompt: str = ""

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


def _make_response(
    state: str = "clear",
    reason: str = "Has context.",
    company_size: str = "large",
    quoted_or_listed: str = "no",
    fca_regulated: str = "no",
    procurement: str = "no",
    eu: str = "no",
    missing: list[str] | None = None,
) -> str:
    return json.dumps({
        "state": state,
        "reason": reason,
        "detected_context": {
            "company_size": company_size,
            "quoted_or_listed": quoted_or_listed,
            "fca_regulated": fca_regulated,
            "public_procurement": procurement,
            "eu_operations": eu,
        },
        "missing_fields": missing or [],
    })


# ---------------------------------------------------------------------------
# ClassificationResult mapping
# ---------------------------------------------------------------------------

def test_classify_clear_state():
    classifier = Classifier(client=FakeClassifierClient(_make_response(state="clear")))
    result = classifier.classify("We have 300 employees and £45m turnover.")
    assert result.state == "clear"


def test_classify_needs_clarification_state():
    resp = _make_response(
        state="needs_clarification",
        reason="Company size unknown.",
        company_size="unknown",
        quoted_or_listed="unknown",
        missing=["company_size", "quoted_or_listed"],
    )
    classifier = Classifier(client=FakeClassifierClient(resp))
    result = classifier.classify("What sustainability reporting do I need?")
    assert result.state == "needs_clarification"


def test_classify_out_of_scope_state():
    resp = _make_response(state="out_of_scope", reason="Not related to UK green regulation.")
    classifier = Classifier(client=FakeClassifierClient(resp))
    result = classifier.classify("What is the best recipe for pasta?")
    assert result.state == "out_of_scope"


def test_classify_reason_propagated():
    resp = _make_response(state="clear", reason="Sufficient context provided.")
    classifier = Classifier(client=FakeClassifierClient(resp))
    result = classifier.classify("q")
    assert result.reason == "Sufficient context provided."


def test_classify_detected_context_company_size_large():
    classifier = Classifier(client=FakeClassifierClient(_make_response(company_size="large")))
    result = classifier.classify("q")
    assert result.detected_context.company_size == "large"


def test_classify_detected_context_company_size_sme():
    classifier = Classifier(client=FakeClassifierClient(_make_response(company_size="sme")))
    result = classifier.classify("q")
    assert result.detected_context.company_size == "sme"


def test_classify_detected_context_quoted_or_listed():
    classifier = Classifier(client=FakeClassifierClient(
        _make_response(quoted_or_listed="yes")
    ))
    result = classifier.classify("q")
    assert result.detected_context.quoted_or_listed == "yes"


def test_classify_detected_context_fca_regulated():
    classifier = Classifier(client=FakeClassifierClient(
        _make_response(fca_regulated="yes")
    ))
    result = classifier.classify("q")
    assert result.detected_context.fca_regulated == "yes"


def test_classify_detected_context_procurement():
    classifier = Classifier(client=FakeClassifierClient(_make_response(procurement="yes")))
    result = classifier.classify("q")
    assert result.detected_context.public_procurement == "yes"


def test_classify_detected_context_eu_operations():
    classifier = Classifier(client=FakeClassifierClient(_make_response(eu="yes")))
    result = classifier.classify("q")
    assert result.detected_context.eu_operations == "yes"


def test_classify_missing_fields_populated():
    resp = _make_response(
        state="needs_clarification",
        company_size="unknown",
        missing=["company_size", "quoted_or_listed"],
    )
    classifier = Classifier(client=FakeClassifierClient(resp))
    result = classifier.classify("q")
    assert "company_size" in result.missing_fields
    assert "quoted_or_listed" in result.missing_fields


def test_classify_partial_context_some_known_some_unknown():
    """Procurement known, company size unknown — partial context handled correctly."""
    resp = _make_response(
        state="needs_clarification",
        company_size="unknown",
        procurement="yes",
        missing=["company_size"],
    )
    classifier = Classifier(client=FakeClassifierClient(resp))
    result = classifier.classify("We are bidding for a government contract.")
    assert result.detected_context.public_procurement == "yes"
    assert result.detected_context.company_size == "unknown"
    assert result.missing_fields == ["company_size"]


def test_classify_malformed_json_raises_value_error():
    classifier = Classifier(client=FakeClassifierClient("not valid json {{ at all"))
    with pytest.raises(ValueError, match="Failed to parse"):
        classifier.classify("q")


def test_classify_missing_state_key_raises_value_error():
    classifier = Classifier(client=FakeClassifierClient('{"reason": "ok"}'))
    with pytest.raises(ValueError, match="Failed to parse"):
        classifier.classify("q")


def test_classify_strips_markdown_code_fences():
    """Model sometimes wraps JSON in ```json ... ``` — must be handled gracefully."""
    fenced = "```json\n" + _make_response(state="clear") + "\n```"
    classifier = Classifier(client=FakeClassifierClient(fenced))
    result = classifier.classify("q")
    assert result.state == "clear"


def test_classify_partial_answer_needs_clarification_state_is_valid():
    """partial_answer_needs_clarification must pass validation without raising ValueError."""
    resp = _make_response(
        state="partial_answer_needs_clarification",
        reason="Enough to partially answer; listing status unknown.",
        company_size="large",
        missing=["quoted_or_listed"],
    )
    classifier = Classifier(client=FakeClassifierClient(resp))
    result = classifier.classify("We have 300 employees and £45m turnover.")
    assert result.state == "partial_answer_needs_clarification"
    assert result.missing_fields == ["quoted_or_listed"]


def test_classify_unexpected_state_raises_value_error():
    """Valid JSON with an unrecognised state value must raise ValueError, not silently route."""
    resp = json.dumps({
        "state": "hallucinated_state",
        "reason": "something unexpected",
        "detected_context": {
            "company_size": "unknown",
            "quoted_or_listed": "unknown",
            "fca_regulated": "unknown",
            "public_procurement": "unknown",
            "eu_operations": "unknown",
        },
        "missing_fields": [],
    })
    classifier = Classifier(client=FakeClassifierClient(resp))
    with pytest.raises(ValueError, match="hallucinated_state"):
        classifier.classify("q")


def test_classify_client_receives_the_question():
    fake = FakeClassifierClient(_make_response())
    classifier = Classifier(client=fake)
    classifier.classify("specific ESOS compliance question")
    assert "specific ESOS compliance question" in fake.last_prompt


def test_classify_returns_classification_result_instance():
    classifier = Classifier(client=FakeClassifierClient(_make_response()))
    result = classifier.classify("q")
    assert isinstance(result, ClassificationResult)


# ---------------------------------------------------------------------------
# Module-level constants (Steps 3 & 4)
# ---------------------------------------------------------------------------

def test_clarification_question_is_string():
    assert isinstance(CLARIFICATION_QUESTION, str)
    assert len(CLARIFICATION_QUESTION) > 0


def test_clarification_options_has_six_items():
    assert len(CLARIFICATION_OPTIONS) == 6


def test_clarification_options_are_strings():
    assert all(isinstance(o, str) for o in CLARIFICATION_OPTIONS)


def test_context_from_option_maps_all_six_indices():
    assert set(CONTEXT_FROM_OPTION.keys()) == {0, 1, 2, 3, 4, 5}


def test_context_from_option_values_are_strings():
    assert all(isinstance(v, str) for v in CONTEXT_FROM_OPTION.values())


def test_context_from_option_large_company():
    assert "large" in CONTEXT_FROM_OPTION[0].lower()


def test_context_from_option_listed():
    assert "listed" in CONTEXT_FROM_OPTION[1].lower() or "fca" in CONTEXT_FROM_OPTION[1].lower()


def test_context_from_option_sme():
    assert "sme" in CONTEXT_FROM_OPTION[2].lower()


# ---------------------------------------------------------------------------
# Retrieval gap — CONTEXT_FROM_OPTION[1] must contain UK SDS vocabulary
# ---------------------------------------------------------------------------

def test_context_from_option_listed_contains_uk_sds():
    """Option 1 must mention UK SDS to surface uk_sds_status_timeline.md chunks."""
    assert "uk sds" in CONTEXT_FROM_OPTION[1].lower()


def test_context_from_option_listed_contains_2026():
    """Option 1 must include '2026' to match the incoming obligation timeline."""
    assert "2026" in CONTEXT_FROM_OPTION[1]


def test_context_from_option_listed_contains_sustainability_disclosure():
    """Option 1 must include 'sustainability disclosure' retrieval vocabulary."""
    assert "sustainability disclosure" in CONTEXT_FROM_OPTION[1].lower()
