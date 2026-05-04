"""
Tests for src/evals/runner.py — EvalResult dataclass and EvalRunner scoring.

All tests use fakes; no real API calls are made.
"""
import pytest
from src.evals.runner import EvalResult, EvalRunner
from src.models import QueryResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeQueryEngine:
    """Returns a fixed QueryResult for every ask() call."""

    def __init__(self, result: QueryResult):
        self._result = result

    def ask(self, question: str, company_context: str = "") -> QueryResult:
        return self._result


def _answer_result(text: str = "Some answer text.", sources: list[str] | None = None) -> QueryResult:
    return QueryResult(kind="answer", answer=text, sources=sources or [])


def _clarification_result() -> QueryResult:
    return QueryResult(
        kind="clarification_needed",
        clarification_question="Please tell us more about your company.",
        options=["Option A", "Option B"],
    )


def _out_of_scope_result() -> QueryResult:
    return QueryResult(
        kind="out_of_scope",
        answer="This tool covers UK green regulation and sustainability compliance.",
    )


def _eval_case(
    *,
    id: str = "eval_001",
    category: str = "threshold",
    question: str = "What size triggers SECR?",
    company_context: str = "",
    expected_state: str = "clear",
    expected_facts: list[str] | None = None,
    expected_citations: list[str] | None = None,
    expected_retrieved_documents: list[str] | None = None,
    required_caveats: list[str] | None = None,
    must_not_contain: list[str] | None = None,
    forbidden_certainty: list[str] | None = None,
) -> dict:
    return {
        "id": id,
        "category": category,
        "question": question,
        "company_context": company_context,
        "expected_state": expected_state,
        "expected_facts": expected_facts or [],
        "expected_citations": expected_citations or [],
        "expected_retrieved_documents": expected_retrieved_documents or [],
        "required_caveats": required_caveats or [],
        "must_not_contain": must_not_contain or [],
        "forbidden_certainty": forbidden_certainty or [],
    }


def _runner(result: QueryResult) -> EvalRunner:
    return EvalRunner(engine=FakeQueryEngine(result))


# ---------------------------------------------------------------------------
# EvalResult structure — identity fields
# ---------------------------------------------------------------------------

def test_run_eval_returns_eval_result():
    result = _runner(_answer_result()).run_eval(_eval_case())
    assert isinstance(result, EvalResult)


def test_eval_id_propagated():
    result = _runner(_answer_result()).run_eval(_eval_case(id="eval_042"))
    assert result.eval_id == "eval_042"


def test_question_propagated():
    result = _runner(_answer_result()).run_eval(_eval_case(question="What is ESOS?"))
    assert result.question == "What is ESOS?"


def test_expected_state_propagated():
    result = _runner(_answer_result()).run_eval(_eval_case(expected_state="clear"))
    assert result.expected_state == "clear"


def test_category_propagated():
    result = _runner(_answer_result()).run_eval(_eval_case(category="hallucination_traps"))
    assert result.category == "hallucination_traps"


# ---------------------------------------------------------------------------
# State mapping — QueryResult.kind → actual_state string
# ---------------------------------------------------------------------------

def test_answer_kind_maps_to_clear_actual_state():
    result = _runner(_answer_result()).run_eval(_eval_case(expected_state="clear"))
    assert result.actual_state == "clear"


def test_clarification_needed_kind_maps_to_needs_clarification_actual_state():
    result = _runner(_clarification_result()).run_eval(
        _eval_case(expected_state="needs_clarification")
    )
    assert result.actual_state == "needs_clarification"


def test_out_of_scope_kind_maps_to_out_of_scope_actual_state():
    result = _runner(_out_of_scope_result()).run_eval(
        _eval_case(expected_state="out_of_scope")
    )
    assert result.actual_state == "out_of_scope"


# ---------------------------------------------------------------------------
# State match — clear
# ---------------------------------------------------------------------------

def test_clear_expected_answer_actual_is_state_match():
    result = _runner(_answer_result()).run_eval(_eval_case(expected_state="clear"))
    assert result.state_match is True


def test_clear_expected_clarification_actual_is_not_state_match():
    result = _runner(_clarification_result()).run_eval(_eval_case(expected_state="clear"))
    assert result.state_match is False


def test_clear_expected_out_of_scope_actual_is_not_state_match():
    result = _runner(_out_of_scope_result()).run_eval(_eval_case(expected_state="clear"))
    assert result.state_match is False


# ---------------------------------------------------------------------------
# State match — needs_clarification
# ---------------------------------------------------------------------------

def test_needs_clarification_expected_clarification_kind_is_state_match():
    result = _runner(_clarification_result()).run_eval(
        _eval_case(expected_state="needs_clarification")
    )
    assert result.state_match is True


def test_needs_clarification_expected_answer_actual_is_not_state_match():
    result = _runner(_answer_result()).run_eval(
        _eval_case(expected_state="needs_clarification")
    )
    assert result.state_match is False


# ---------------------------------------------------------------------------
# State match — out_of_scope
# ---------------------------------------------------------------------------

def test_out_of_scope_expected_and_actual_is_state_match():
    result = _runner(_out_of_scope_result()).run_eval(
        _eval_case(expected_state="out_of_scope")
    )
    assert result.state_match is True


def test_out_of_scope_expected_answer_actual_is_not_state_match():
    result = _runner(_answer_result()).run_eval(
        _eval_case(expected_state="out_of_scope")
    )
    assert result.state_match is False


# ---------------------------------------------------------------------------
# State match — partial_answer_needs_clarification (approximate scoring)
# ---------------------------------------------------------------------------

def test_partial_state_answer_with_fact_and_citation_is_state_match():
    """Approximate pass: kind=answer, fact present, citation present."""
    case = _eval_case(
        expected_state="partial_answer_needs_clarification",
        expected_facts=["250 employees"],
        expected_citations=["secr.pdf"],
    )
    result = _runner(
        _answer_result(
            text="SECR applies to companies with 250 employees or more.",
            sources=["secr.pdf"],
        )
    ).run_eval(case)
    assert result.state_match is True


def test_partial_state_answer_no_fact_is_not_state_match():
    """Missing fact: approximate scoring fails."""
    case = _eval_case(
        expected_state="partial_answer_needs_clarification",
        expected_facts=["250 employees"],
        expected_citations=["secr.pdf"],
    )
    result = _runner(
        _answer_result(
            text="Some general answer with no specific fact.",
            sources=["secr.pdf"],
        )
    ).run_eval(case)
    assert result.state_match is False


def test_partial_state_answer_no_citation_is_not_state_match():
    """Missing citation: approximate scoring fails."""
    case = _eval_case(
        expected_state="partial_answer_needs_clarification",
        expected_facts=["250 employees"],
        expected_citations=["secr.pdf"],
    )
    result = _runner(
        _answer_result(
            text="SECR applies to companies with 250 employees or more.",
            sources=["other.pdf"],
        )
    ).run_eval(case)
    assert result.state_match is False


def test_partial_state_clarification_with_fact_and_citation_is_state_match():
    """kind=clarification_needed also accepted for partial state if fact+citation present in notes."""
    # For clarification_needed, answer is None so no fact can be found — should fail.
    case = _eval_case(
        expected_state="partial_answer_needs_clarification",
        expected_facts=["250 employees"],
        expected_citations=["secr.pdf"],
    )
    result = _runner(_clarification_result()).run_eval(case)
    # clarification_needed has no answer text, so fact check fails
    assert result.state_match is False


def test_partial_state_notes_contain_warning():
    case = _eval_case(
        expected_state="partial_answer_needs_clarification",
        expected_facts=["250 employees"],
        expected_citations=["secr.pdf"],
    )
    result = _runner(
        _answer_result(
            text="SECR applies to companies with 250 employees.",
            sources=["secr.pdf"],
        )
    ).run_eval(case)
    assert "partial_answer_needs_clarification" in result.notes.lower()
    assert "warning" in result.notes.lower()


# ---------------------------------------------------------------------------
# Citation accuracy
# ---------------------------------------------------------------------------

def test_citation_accuracy_true_when_all_expected_citations_in_sources():
    case = _eval_case(expected_citations=["secr.pdf", "esos.md"])
    result = _runner(_answer_result(sources=["secr.pdf", "esos.md", "other.pdf"])).run_eval(case)
    assert result.citation_accuracy is True


def test_citation_accuracy_false_when_citation_missing():
    case = _eval_case(expected_citations=["secr.pdf", "esos.md"])
    result = _runner(_answer_result(sources=["secr.pdf"])).run_eval(case)
    assert result.citation_accuracy is False


def test_citation_accuracy_true_when_no_expected_citations():
    case = _eval_case(expected_citations=[])
    result = _runner(_clarification_result()).run_eval(case)
    assert result.citation_accuracy is True


def test_citation_accuracy_false_when_expected_citations_but_sources_empty():
    case = _eval_case(expected_citations=["secr.pdf"])
    result = _runner(_answer_result(sources=[])).run_eval(case)
    assert result.citation_accuracy is False


# ---------------------------------------------------------------------------
# Retrieval recall
# ---------------------------------------------------------------------------

def test_retrieval_recall_one_point_zero_when_all_docs_found():
    case = _eval_case(expected_retrieved_documents=["secr.pdf", "esos.md"])
    result = _runner(_answer_result(sources=["secr.pdf", "esos.md"])).run_eval(case)
    assert result.retrieval_recall == 1.0


def test_retrieval_recall_half_when_one_of_two_found():
    case = _eval_case(expected_retrieved_documents=["secr.pdf", "esos.md"])
    result = _runner(_answer_result(sources=["secr.pdf"])).run_eval(case)
    assert result.retrieval_recall == 0.5


def test_retrieval_recall_zero_when_no_docs_found():
    case = _eval_case(expected_retrieved_documents=["secr.pdf", "esos.md"])
    result = _runner(_answer_result(sources=[])).run_eval(case)
    assert result.retrieval_recall == 0.0


def test_retrieval_recall_one_when_expected_docs_is_empty():
    case = _eval_case(expected_retrieved_documents=[])
    result = _runner(_clarification_result()).run_eval(case)
    assert result.retrieval_recall == 1.0


# ---------------------------------------------------------------------------
# Fact presence — case-insensitive substring match
# ---------------------------------------------------------------------------

def test_facts_present_case_insensitive():
    case = _eval_case(expected_facts=["250 Employees"])
    result = _runner(
        _answer_result(text="SECR applies to companies with 250 employees or more.")
    ).run_eval(case)
    assert "250 Employees" in result.facts_present
    assert result.facts_missing == []


def test_facts_missing_when_not_in_answer():
    case = _eval_case(expected_facts=["£36 million"])
    result = _runner(_answer_result(text="There is no mention of the threshold here.")).run_eval(case)
    assert "£36 million" in result.facts_missing
    assert result.facts_present == []


def test_facts_split_correctly_between_present_and_missing():
    case = _eval_case(expected_facts=["250 employees", "£36 million"])
    result = _runner(
        _answer_result(text="SECR applies to companies with 250 employees.")
    ).run_eval(case)
    assert "250 employees" in result.facts_present
    assert "£36 million" in result.facts_missing


def test_facts_empty_when_answer_is_none():
    """clarification_needed returns None answer — no facts can be found."""
    case = _eval_case(expected_facts=["250 employees"])
    result = _runner(_clarification_result()).run_eval(case)
    assert result.facts_present == []
    assert "250 employees" in result.facts_missing


# ---------------------------------------------------------------------------
# Caveat presence — case-insensitive substring match
# ---------------------------------------------------------------------------

def test_caveats_present_when_in_answer():
    case = _eval_case(required_caveats=["subject to consultation"])
    result = _runner(
        _answer_result(text="This is indicative only, subject to consultation and review.")
    ).run_eval(case)
    assert "subject to consultation" in result.caveats_present
    assert result.caveats_missing == []


def test_caveats_missing_when_not_in_answer():
    case = _eval_case(required_caveats=["subject to consultation"])
    result = _runner(_answer_result(text="The deadline is 5 June 2024.")).run_eval(case)
    assert "subject to consultation" in result.caveats_missing
    assert result.caveats_present == []


# ---------------------------------------------------------------------------
# Forbidden phrases — must_not_contain and forbidden_certainty
# ---------------------------------------------------------------------------

def test_forbidden_phrase_found_sets_hallucination_flag():
    case = _eval_case(must_not_contain=["500 employees"])
    result = _runner(
        _answer_result(text="SECR applies to companies with 500 employees.")
    ).run_eval(case)
    assert result.hallucination_flag is True
    assert "500 employees" in result.forbidden_phrases_found


def test_forbidden_certainty_found_sets_hallucination_flag():
    case = _eval_case(forbidden_certainty=["CSRD applies to all UK companies"])
    result = _runner(
        _answer_result(text="Note that CSRD applies to all UK companies equally.")
    ).run_eval(case)
    assert result.hallucination_flag is True
    assert "CSRD applies to all UK companies" in result.forbidden_certainty_found


def test_no_hallucination_when_no_forbidden_phrase_matches():
    case = _eval_case(must_not_contain=["500 employees"], forbidden_certainty=["all companies"])
    result = _runner(
        _answer_result(text="SECR applies to companies with 250 employees.")
    ).run_eval(case)
    assert result.hallucination_flag is False
    assert result.forbidden_phrases_found == []
    assert result.forbidden_certainty_found == []


def test_forbidden_phrase_check_is_case_insensitive():
    case = _eval_case(must_not_contain=["500 EMPLOYEES"])
    result = _runner(
        _answer_result(text="SECR applies to companies with 500 employees.")
    ).run_eval(case)
    assert result.hallucination_flag is True


def test_forbidden_phrases_empty_when_answer_is_none():
    """clarification_needed has no answer text — no forbidden phrases can trigger."""
    case = _eval_case(must_not_contain=["SECR applies to your company"])
    result = _runner(_clarification_result()).run_eval(case)
    assert result.hallucination_flag is False


# ---------------------------------------------------------------------------
# passed — True only when state_match AND citation_accuracy AND NOT hallucination
# ---------------------------------------------------------------------------

def test_passed_true_when_all_conditions_met():
    case = _eval_case(
        expected_state="clear",
        expected_citations=["secr.pdf"],
        must_not_contain=["500 employees"],
    )
    result = _runner(
        _answer_result(
            text="SECR applies to companies with 250 employees.",
            sources=["secr.pdf"],
        )
    ).run_eval(case)
    assert result.passed is True


def test_passed_false_when_state_mismatch():
    case = _eval_case(expected_state="out_of_scope")
    result = _runner(_answer_result()).run_eval(case)
    assert result.passed is False


def test_passed_false_when_citation_accuracy_false():
    case = _eval_case(
        expected_state="clear",
        expected_citations=["secr.pdf"],
    )
    result = _runner(_answer_result(sources=["esos.md"])).run_eval(case)
    assert result.passed is False


def test_passed_false_when_hallucination_flag():
    case = _eval_case(
        expected_state="clear",
        expected_citations=[],
        must_not_contain=["500 employees"],
    )
    result = _runner(
        _answer_result(text="SECR applies to companies with 500 employees.")
    ).run_eval(case)
    assert result.passed is False


def test_passed_false_when_state_mismatch_even_if_no_hallucination():
    case = _eval_case(expected_state="needs_clarification")
    result = _runner(_answer_result()).run_eval(case)
    assert result.state_match is False
    assert result.passed is False


# ---------------------------------------------------------------------------
# Bug 1 — Source normalisation (path prefix, casing, whitespace)
# ---------------------------------------------------------------------------

def test_citation_accuracy_true_with_path_prefixed_source():
    """Path-prefixed actual source must match bare expected citation."""
    case = _eval_case(expected_citations=["fca_tcfd_ps21_24_summary.md"])
    result = _runner(
        _answer_result(sources=["corpus/mock/fca_tcfd_ps21_24_summary.md"])
    ).run_eval(case)
    assert result.citation_accuracy is True


def test_citation_accuracy_true_with_casing_difference():
    """Case difference between expected and actual citation must not prevent match."""
    case = _eval_case(expected_citations=["FCA_TCFD_PS21_24_summary.md"])
    result = _runner(
        _answer_result(sources=["fca_tcfd_ps21_24_summary.md"])
    ).run_eval(case)
    assert result.citation_accuracy is True


def test_citation_accuracy_true_with_leading_whitespace_in_source():
    """Leading/trailing whitespace in source must not prevent match."""
    case = _eval_case(expected_citations=["secr_guidance.md"])
    result = _runner(
        _answer_result(sources=["  secr_guidance.md  "])
    ).run_eval(case)
    assert result.citation_accuracy is True


def test_retrieval_recall_normalises_path_prefix():
    """Path-prefixed actual source must count toward retrieval recall."""
    case = _eval_case(expected_retrieved_documents=["fca_tcfd_ps21_24_summary.md"])
    result = _runner(
        _answer_result(sources=["corpus/mock/fca_tcfd_ps21_24_summary.md"])
    ).run_eval(case)
    assert result.retrieval_recall == 1.0


def test_retrieval_recall_normalises_casing():
    """Upper-cased actual source must count toward retrieval recall."""
    case = _eval_case(expected_retrieved_documents=["esos_guidance.md"])
    result = _runner(
        _answer_result(sources=["ESOS_GUIDANCE.md"])
    ).run_eval(case)
    assert result.retrieval_recall == 1.0


def test_retrieval_recall_normalises_windows_path_prefix():
    """Windows-style backslash path must be stripped correctly."""
    case = _eval_case(expected_retrieved_documents=["secr_summary.md"])
    result = _runner(
        _answer_result(sources=["corpus\\mock\\secr_summary.md"])
    ).run_eval(case)
    assert result.retrieval_recall == 1.0


# ---------------------------------------------------------------------------
# Bug 2 — Fact presence key-term matching (paraphrase tolerance)
# ---------------------------------------------------------------------------

def test_fact_presence_scores_paraphrase_as_present():
    """Paraphrased equivalent of a fact must score as present."""
    case = _eval_case(
        expected_facts=["260 employees exceeds the SECR employee criterion of 250"]
    )
    result = _runner(
        _answer_result(
            text="Your 260 staff exceeds the 250-employee threshold under SECR reporting rules."
        )
    ).run_eval(case)
    assert result.facts_present != []
    assert result.facts_missing == []


def test_fact_presence_exact_match_still_works():
    """Exact substring match must still score as present after the change."""
    case = _eval_case(expected_facts=["250 employees"])
    result = _runner(
        _answer_result(text="SECR applies to companies with 250 employees or more.")
    ).run_eval(case)
    assert "250 employees" in result.facts_present


def test_fact_presence_numeric_values_anchor_match():
    """Shared numeric values are key terms that anchor fact matching."""
    case = _eval_case(expected_facts=["annual turnover exceeds £36m threshold"])
    result = _runner(
        _answer_result(
            text="Your annual turnover of £36m puts you above the reporting threshold."
        )
    ).run_eval(case)
    assert result.facts_present != []
    assert result.facts_missing == []


def test_fact_presence_employees_not_stop_word():
    """'employees' must contribute as a key term so '500 employees threshold' can match."""
    case = _eval_case(expected_facts=["500 employees threshold"])
    result = _runner(
        _answer_result(
            text="Companies with more than 500 employees meet the size criterion."
        )
    ).run_eval(case)
    assert result.facts_present != []
    assert result.facts_missing == []


def test_fact_presence_turnover_not_stop_word():
    """'turnover' must contribute as a key term so a paraphrase can match."""
    case = _eval_case(expected_facts=["£36m turnover limit"])
    result = _runner(
        _answer_result(
            text="Organisations whose turnover exceeds £36m are in scope."
        )
    ).run_eval(case)
    assert result.facts_present != []
    assert result.facts_missing == []


def test_fact_presence_company_not_stop_word():
    """'company' must contribute as a key term and not be silently dropped."""
    case = _eval_case(expected_facts=["250 company employees"])
    result = _runner(
        _answer_result(
            text="A company with 250 employees triggers the reporting duty."
        )
    ).run_eval(case)
    assert result.facts_present != []
    assert result.facts_missing == []


def test_fact_presence_short_abbreviation_is_key_term():
    """3-char regulatory abbreviations like 'kWh' must now be treated as key terms."""
    case = _eval_case(expected_facts=["100 kWh Ltd limit"])
    result = _runner(
        _answer_result(
            text="A Ltd entity using 100 kWh or more must register."
        )
    ).run_eval(case)
    assert result.facts_present != []
    assert result.facts_missing == []


def test_fact_presence_completely_different_statement_is_absent():
    """A statement sharing no key terms with the fact must score as absent."""
    case = _eval_case(
        expected_facts=["260 employees exceeds the SECR employee criterion of 250"]
    )
    result = _runner(
        _answer_result(
            text="The plastic packaging tax applies to producers of 10 tonnes or more per year."
        )
    ).run_eval(case)
    assert result.facts_missing != []


# ---------------------------------------------------------------------------
# Step 1 — Tightened pass/fail gate
# ---------------------------------------------------------------------------

def test_passed_false_when_facts_missing():
    """Facts missing from the answer must cause a pass failure."""
    case = _eval_case(expected_state="clear", expected_facts=["250 employees"])
    result = _runner(_answer_result(text="No relevant facts here.")).run_eval(case)
    assert result.passed is False


def test_passed_false_when_caveats_missing():
    """Required caveats absent from the answer must cause a pass failure."""
    case = _eval_case(expected_state="clear", required_caveats=["subject to change"])
    result = _runner(_answer_result(text="SECR applies to large companies.")).run_eval(case)
    assert result.passed is False


def test_passed_false_when_retrieval_recall_below_default_threshold():
    """Retrieval recall of 0.0 is below the default 0.5 threshold — must fail."""
    case = _eval_case(
        expected_state="clear",
        expected_retrieved_documents=["secr.pdf", "esos.md"],
    )
    result = _runner(_answer_result(sources=[])).run_eval(case)
    assert result.passed is False


def test_passed_true_when_retrieval_recall_meets_default_threshold():
    """All expected docs retrieved — recall = 1.0 >= 0.5 — must pass (other conditions met)."""
    case = _eval_case(
        expected_state="clear",
        expected_retrieved_documents=["secr.pdf", "esos.md"],
    )
    result = _runner(_answer_result(sources=["secr.pdf", "esos.md"])).run_eval(case)
    assert result.passed is True


def test_required_retrieval_recall_per_eval_overrides_default():
    """required_retrieval_recall in the eval case overrides the 0.5 default."""
    case = _eval_case(
        expected_state="clear",
        expected_retrieved_documents=["secr.pdf", "esos.md"],
    )
    case["required_retrieval_recall"] = 0.8
    # recall = 0.5 (1/2 found), threshold = 0.8 → should fail
    result = _runner(_answer_result(sources=["secr.pdf"])).run_eval(case)
    assert result.passed is False


def test_required_retrieval_recall_higher_threshold_passes_when_met():
    """With required_retrieval_recall = 0.8 and recall = 1.0, must pass."""
    case = _eval_case(
        expected_state="clear",
        expected_retrieved_documents=["secr.pdf", "esos.md"],
    )
    case["required_retrieval_recall"] = 0.8
    result = _runner(_answer_result(sources=["secr.pdf", "esos.md"])).run_eval(case)
    assert result.passed is True


def test_required_retrieval_recall_stored_in_result():
    """required_retrieval_recall from the eval case must be stored on EvalResult."""
    case = _eval_case(expected_state="clear")
    case["required_retrieval_recall"] = 0.75
    result = _runner(_answer_result()).run_eval(case)
    assert result.required_retrieval_recall == 0.75


def test_required_retrieval_recall_defaults_to_half():
    """Default required_retrieval_recall must be 0.5 when absent from eval case."""
    case = _eval_case(expected_state="clear")
    result = _runner(_answer_result()).run_eval(case)
    assert result.required_retrieval_recall == 0.5


# ---------------------------------------------------------------------------
# Step 2 — Failure taxonomy
# ---------------------------------------------------------------------------

def test_failure_type_none_when_passed():
    """failure_type must be None for a passing eval."""
    case = _eval_case(expected_state="clear")
    result = _runner(_answer_result()).run_eval(case)
    assert result.passed is True
    assert result.failure_type is None


def test_failure_type_classification_failure_on_state_mismatch():
    """State mismatch → failure_type == 'classification_failure'."""
    case = _eval_case(expected_state="out_of_scope")
    result = _runner(_answer_result()).run_eval(case)
    assert result.failure_type == "classification_failure"


def test_failure_type_retrieval_failure_when_recall_below_threshold():
    """Recall below threshold → failure_type == 'retrieval_failure'."""
    case = _eval_case(
        expected_state="clear",
        expected_retrieved_documents=["secr.pdf", "esos.md"],
    )
    # sources=[] → recall = 0.0 < 0.5
    result = _runner(_answer_result(sources=[])).run_eval(case)
    assert result.failure_type == "retrieval_failure"


def test_failure_type_missing_facts_when_facts_absent():
    """Facts absent from answer → failure_type == 'missing_facts'."""
    case = _eval_case(expected_state="clear", expected_facts=["250 employees"])
    result = _runner(_answer_result(text="No relevant content here.")).run_eval(case)
    assert result.failure_type == "missing_facts"


def test_failure_type_missing_caveats_when_caveats_absent():
    """Required caveats absent → failure_type == 'missing_caveats'."""
    case = _eval_case(expected_state="clear", required_caveats=["subject to change"])
    result = _runner(_answer_result(text="SECR applies to large companies.")).run_eval(case)
    assert result.failure_type == "missing_caveats"


def test_failure_type_citation_failure_when_citation_missing():
    """Expected citation absent from sources → failure_type == 'citation_failure'."""
    case = _eval_case(expected_state="clear", expected_citations=["secr.pdf"])
    result = _runner(_answer_result(sources=["esos.md"])).run_eval(case)
    assert result.failure_type == "citation_failure"


def test_failure_type_hallucination_when_forbidden_phrase_found():
    """must_not_contain phrase in answer → failure_type == 'hallucination'."""
    case = _eval_case(expected_state="clear", must_not_contain=["500 employees"])
    result = _runner(_answer_result(text="SECR applies to 500 employees.")).run_eval(case)
    assert result.failure_type == "hallucination"


def test_failure_type_forbidden_certainty_when_only_certainty_triggered():
    """forbidden_certainty (no must_not_contain) → failure_type == 'forbidden_certainty'."""
    case = _eval_case(expected_state="clear", forbidden_certainty=["CSRD applies to all UK"])
    result = _runner(
        _answer_result(text="Note that CSRD applies to all UK companies.")
    ).run_eval(case)
    assert result.failure_type == "forbidden_certainty"


def test_failure_type_priority_classification_over_retrieval():
    """Classification failure takes priority over retrieval failure."""
    case = _eval_case(
        expected_state="out_of_scope",
        expected_retrieved_documents=["secr.pdf"],
    )
    # State mismatch AND retrieval failure both apply — classification wins
    result = _runner(_answer_result(sources=[])).run_eval(case)
    assert result.failure_type == "classification_failure"


def test_failure_type_priority_retrieval_over_missing_facts():
    """Retrieval failure takes priority over missing facts."""
    case = _eval_case(
        expected_state="clear",
        expected_retrieved_documents=["secr.pdf", "esos.md"],
        expected_facts=["250 employees"],
    )
    # Recall = 0.0 < 0.5 AND facts missing — retrieval wins
    result = _runner(_answer_result(sources=[], text="No content.")).run_eval(case)
    assert result.failure_type == "retrieval_failure"


def test_failure_type_answer_posture_failure_for_partial_state_approx_pass():
    """Partial state eval that passes approximate scoring fails native → 'answer_posture_failure'."""
    case = _eval_case(
        expected_state="partial_answer_needs_clarification",
        expected_facts=["250 employees"],
        expected_citations=["secr.pdf"],
    )
    result = _runner(
        _answer_result(
            text="SECR applies to companies with 250 employees.",
            sources=["secr.pdf"],
        )
    ).run_eval(case)
    # Approximate scoring passes (fact + citation found), native always fails for this state
    assert result.failure_type == "answer_posture_failure"


# ---------------------------------------------------------------------------
# Step 3 — passed_architecture_compatible
# ---------------------------------------------------------------------------

def test_passed_architecture_compatible_field_exists():
    """EvalResult must have a passed_architecture_compatible field."""
    case = _eval_case(expected_state="clear")
    result = _runner(_answer_result()).run_eval(case)
    assert hasattr(result, "passed_architecture_compatible")


def test_partial_state_native_false_arch_compatible_true():
    """Partial state eval that approx-passes: native=False, arch_compatible=True."""
    case = _eval_case(
        expected_state="partial_answer_needs_clarification",
        expected_facts=["250 employees"],
        expected_citations=["secr.pdf"],
    )
    result = _runner(
        _answer_result(
            text="SECR applies to companies with 250 employees.",
            sources=["secr.pdf"],
        )
    ).run_eval(case)
    assert result.passed is False
    assert result.passed_architecture_compatible is True


def test_non_partial_state_native_and_arch_compatible_agree_on_pass():
    """Non-partial passing eval: both native and arch_compatible are True."""
    case = _eval_case(expected_state="clear")
    result = _runner(_answer_result()).run_eval(case)
    assert result.passed is True
    assert result.passed_architecture_compatible is True


def test_non_partial_state_native_and_arch_compatible_agree_on_fail():
    """Non-partial failing eval: both native and arch_compatible are False."""
    case = _eval_case(expected_state="out_of_scope")
    result = _runner(_answer_result()).run_eval(case)
    assert result.passed is False
    assert result.passed_architecture_compatible is False


def test_forbidden_certainty_flag_field_exists():
    """EvalResult must expose forbidden_certainty_flag as a separate field."""
    case = _eval_case(expected_state="clear")
    result = _runner(_answer_result()).run_eval(case)
    assert hasattr(result, "forbidden_certainty_flag")


def test_forbidden_certainty_flag_true_when_certainty_triggered():
    """forbidden_certainty_flag must be True when a forbidden_certainty phrase is found."""
    case = _eval_case(forbidden_certainty=["CSRD applies to all UK companies"])
    result = _runner(
        _answer_result(text="Note that CSRD applies to all UK companies equally.")
    ).run_eval(case)
    assert result.forbidden_certainty_flag is True


def test_forbidden_certainty_flag_false_when_only_must_not_contain_triggered():
    """forbidden_certainty_flag must be False when only must_not_contain is triggered."""
    case = _eval_case(must_not_contain=["500 employees"])
    result = _runner(
        _answer_result(text="SECR applies to companies with 500 employees.")
    ).run_eval(case)
    assert result.forbidden_certainty_flag is False


# ---------------------------------------------------------------------------
# Change 2 — EvalResult.retrieved_chunks
# ---------------------------------------------------------------------------

def _answer_result_with_chunks(
    text: str = "Some answer text.",
    sources: list[str] | None = None,
    chunks: list[dict] | None = None,
) -> QueryResult:
    return QueryResult(
        kind="answer",
        answer=text,
        sources=sources or [],
        retrieved_chunks=chunks or [],
    )


def test_eval_result_has_retrieved_chunks_field():
    """EvalResult must expose a retrieved_chunks field."""
    result = _runner(_answer_result()).run_eval(_eval_case())
    assert hasattr(result, "retrieved_chunks")


def test_run_eval_retrieved_chunks_empty_when_query_result_has_none():
    """retrieved_chunks must be [] when the engine returns no chunks."""
    result = _runner(_answer_result()).run_eval(_eval_case())
    assert result.retrieved_chunks == []


def test_run_eval_populates_retrieved_chunks_from_query_result():
    """retrieved_chunks on EvalResult must mirror query_result.retrieved_chunks."""
    chunks = [
        {"source": "secr.pdf", "text": "SECR applies.", "score": 0.9},
        {"source": "esos.md", "text": "ESOS applies.", "score": 0.8},
    ]
    result = _runner(_answer_result_with_chunks(chunks=chunks)).run_eval(_eval_case())
    assert result.retrieved_chunks == chunks


def test_run_eval_retrieved_chunks_length_matches_query_result():
    """Number of retrieved_chunks on EvalResult must equal number in query_result."""
    chunks = [{"source": f"doc{i}.pdf", "text": f"text {i}", "score": 0.9 - i * 0.1}
              for i in range(4)]
    result = _runner(_answer_result_with_chunks(chunks=chunks)).run_eval(_eval_case())
    assert len(result.retrieved_chunks) == 4
