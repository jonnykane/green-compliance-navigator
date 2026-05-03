import pytest
from src.models import DetectedContext, ClassificationResult, QueryResult


# ---------------------------------------------------------------------------
# DetectedContext
# ---------------------------------------------------------------------------

def test_detected_context_defaults_to_unknown():
    dc = DetectedContext()
    assert dc.company_size == "unknown"
    assert dc.quoted_or_listed == "unknown"
    assert dc.fca_regulated == "unknown"
    assert dc.public_procurement == "unknown"
    assert dc.eu_operations == "unknown"


def test_detected_context_accepts_literal_values():
    dc = DetectedContext(
        company_size="large",
        quoted_or_listed="yes",
        fca_regulated="no",
        public_procurement="yes",
        eu_operations="no",
    )
    assert dc.company_size == "large"
    assert dc.quoted_or_listed == "yes"
    assert dc.fca_regulated == "no"
    assert dc.public_procurement == "yes"
    assert dc.eu_operations == "no"


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------

def test_classification_result_clear_state():
    cr = ClassificationResult(state="clear", reason="Company size known.")
    assert cr.state == "clear"
    assert cr.reason == "Company size known."
    assert isinstance(cr.detected_context, DetectedContext)
    assert cr.missing_fields == []


def test_classification_result_needs_clarification():
    cr = ClassificationResult(
        state="needs_clarification",
        reason="Missing company size.",
        missing_fields=["company_size", "quoted_or_listed"],
    )
    assert cr.state == "needs_clarification"
    assert "company_size" in cr.missing_fields


def test_classification_result_out_of_scope():
    cr = ClassificationResult(state="out_of_scope", reason="Unrelated question.")
    assert cr.state == "out_of_scope"


def test_classification_result_detected_context_propagated():
    dc = DetectedContext(company_size="sme")
    cr = ClassificationResult(state="clear", reason="ok", detected_context=dc)
    assert cr.detected_context.company_size == "sme"


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------

def test_query_result_answer_kind():
    qr = QueryResult(kind="answer", answer="ESOS applies.", sources=["esos.md"])
    assert qr.kind == "answer"
    assert qr.answer == "ESOS applies."
    assert qr.sources == ["esos.md"]
    assert qr.clarification_question is None
    assert qr.options is None


def test_query_result_clarification_kind():
    qr = QueryResult(
        kind="clarification_needed",
        clarification_question="What size is your organisation?",
        options=["Large", "SME", "Not sure"],
    )
    assert qr.kind == "clarification_needed"
    assert qr.clarification_question == "What size is your organisation?"
    assert qr.options == ["Large", "SME", "Not sure"]
    assert qr.answer is None
    assert qr.sources == []


def test_query_result_out_of_scope_kind():
    qr = QueryResult(kind="out_of_scope", answer="Not in scope.")
    assert qr.kind == "out_of_scope"
    assert qr.answer == "Not in scope."


def test_query_result_sources_defaults_to_empty_list():
    qr = QueryResult(kind="answer", answer="text")
    assert qr.sources == []
