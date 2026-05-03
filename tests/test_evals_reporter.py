"""
Tests for src/evals/reporter.py — generate_report().

Builds EvalResult objects directly; no QueryEngine involved.
"""
import pytest
from src.evals.runner import EvalResult
from src.evals.reporter import generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(
    *,
    eval_id: str = "eval_001",
    category: str = "threshold",
    question: str = "What triggers SECR?",
    expected_state: str = "clear",
    actual_state: str = "clear",
    state_match: bool = True,
    citation_accuracy: bool = True,
    retrieval_recall: float = 1.0,
    facts_present: list[str] | None = None,
    facts_missing: list[str] | None = None,
    caveats_present: list[str] | None = None,
    caveats_missing: list[str] | None = None,
    forbidden_phrases_found: list[str] | None = None,
    forbidden_certainty_found: list[str] | None = None,
    hallucination_flag: bool = False,
    passed: bool = True,
    notes: str = "",
) -> EvalResult:
    return EvalResult(
        eval_id=eval_id,
        category=category,
        question=question,
        expected_state=expected_state,
        actual_state=actual_state,
        state_match=state_match,
        citation_accuracy=citation_accuracy,
        retrieval_recall=retrieval_recall,
        facts_present=facts_present or [],
        facts_missing=facts_missing or [],
        caveats_present=caveats_present or [],
        caveats_missing=caveats_missing or [],
        forbidden_phrases_found=forbidden_phrases_found or [],
        forbidden_certainty_found=forbidden_certainty_found or [],
        hallucination_flag=hallucination_flag,
        passed=passed,
        notes=notes,
    )


def _passing(**kwargs) -> EvalResult:
    return _result(passed=True, state_match=True, citation_accuracy=True,
                   hallucination_flag=False, **kwargs)


def _failing(**kwargs) -> EvalResult:
    return _result(passed=False, state_match=False, citation_accuracy=False,
                   hallucination_flag=False, **kwargs)


def _hallucinated(**kwargs) -> EvalResult:
    return _result(
        passed=False, state_match=True, citation_accuracy=True,
        hallucination_flag=True,
        forbidden_phrases_found=["bad phrase"],
        **kwargs,
    )


def _partial_state(**kwargs) -> EvalResult:
    return _result(
        expected_state="partial_answer_needs_clarification",
        actual_state="clear",
        state_match=True,
        notes="WARNING: partial_answer_needs_clarification is not natively supported.",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_generate_report_returns_string():
    report = generate_report([_passing()])
    assert isinstance(report, str)


# ---------------------------------------------------------------------------
# SUMMARY section
# ---------------------------------------------------------------------------

def test_report_contains_summary_section():
    report = generate_report([_passing()])
    assert "SUMMARY" in report


def test_report_total_evals_count():
    results = [_passing(), _passing(), _failing()]
    report = generate_report(results)
    assert "3" in report


def test_report_pass_rate_100_percent_when_all_pass():
    results = [_passing(), _passing()]
    report = generate_report(results)
    assert "100.0%" in report or "100%" in report


def test_report_pass_rate_50_percent_when_half_pass():
    results = [_passing(), _failing()]
    report = generate_report(results)
    assert "50.0%" in report or "50%" in report


def test_report_state_match_rate():
    results = [
        _result(state_match=True, passed=True),
        _result(state_match=False, passed=False),
    ]
    report = generate_report(results)
    assert "50.0%" in report or "50%" in report


def test_report_citation_accuracy_rate():
    results = [
        _result(citation_accuracy=True, passed=True),
        _result(citation_accuracy=True, passed=True),
        _result(citation_accuracy=False, passed=False),
    ]
    report = generate_report(results)
    # 2/3 ≈ 66.7%
    assert "66.7%" in report or "66%" in report


def test_report_mean_retrieval_recall():
    results = [
        _result(retrieval_recall=1.0),
        _result(retrieval_recall=0.5),
    ]
    report = generate_report(results)
    assert "75.0%" in report or "75%" in report


def test_report_hallucination_rate_zero_when_none():
    results = [_passing(), _passing()]
    report = generate_report(results)
    assert "0.0%" in report or "0%" in report


def test_report_hallucination_rate_nonzero_when_triggered():
    results = [_passing(), _hallucinated()]
    report = generate_report(results)
    assert "50.0%" in report or "50%" in report


# ---------------------------------------------------------------------------
# BY CATEGORY section
# ---------------------------------------------------------------------------

def test_report_contains_by_category_section():
    report = generate_report([_passing()])
    assert "BY CATEGORY" in report


def test_report_lists_categories_present_in_results():
    results = [
        _passing(category="threshold"),
        _passing(category="out_of_scope"),
    ]
    report = generate_report(results)
    assert "threshold" in report
    assert "out_of_scope" in report


def test_report_category_pass_rate_correct():
    results = [
        _passing(category="threshold"),
        _failing(category="threshold"),
        _passing(category="out_of_scope"),
    ]
    report = generate_report(results)
    # threshold: 1/2 = 50%
    assert "threshold" in report
    assert "1/2" in report


# ---------------------------------------------------------------------------
# FAILURES section
# ---------------------------------------------------------------------------

def test_report_contains_failures_section():
    report = generate_report([_failing()])
    assert "FAILURES" in report


def test_report_failed_eval_appears_in_failures():
    result = _failing(eval_id="eval_007", question="What does SECR require from companies?")
    report = generate_report([result])
    assert "eval_007" in report


def test_report_passed_eval_not_in_failures():
    result = _passing(eval_id="eval_001")
    report = generate_report([result])
    failures_section = report.split("FAILURES")[-1]
    assert "eval_001" not in failures_section


def test_report_question_truncated_to_60_chars_in_failures():
    long_q = "A" * 80
    result = _failing(question=long_q)
    report = generate_report([result])
    assert long_q not in report
    assert "A" * 60 in report


def test_report_failures_shows_expected_and_actual_state():
    result = _failing(
        expected_state="out_of_scope", actual_state="clear",
        question="What is SECR?",
    )
    report = generate_report([result])
    failures_section = report.split("FAILURES")[-1]
    assert "out_of_scope" in failures_section
    assert "clear" in failures_section


def test_report_no_failures_entry_when_all_pass():
    results = [_passing(), _passing()]
    report = generate_report(results)
    # FAILURES section exists but should say nothing failed or be empty of eval IDs
    failures_section = report.split("FAILURES")[-1].split("HALLUCINATION")[0]
    assert "eval_" not in failures_section


# ---------------------------------------------------------------------------
# HALLUCINATION FLAGS section
# ---------------------------------------------------------------------------

def test_report_contains_hallucination_flags_section():
    report = generate_report([_passing()])
    assert "HALLUCINATION" in report


def test_report_hallucinated_eval_appears_in_flags():
    result = _hallucinated(eval_id="eval_030")
    report = generate_report([result])
    flags_section = report.split("HALLUCINATION")[-1]
    assert "eval_030" in flags_section


def test_report_clean_eval_not_in_hallucination_flags():
    result = _passing(eval_id="eval_001")
    report = generate_report([result])
    flags_section = report.split("HALLUCINATION FLAGS")[-1].split("WARNINGS")[0]
    assert "eval_001" not in flags_section


# ---------------------------------------------------------------------------
# WARNINGS section
# ---------------------------------------------------------------------------

def test_report_contains_warnings_section():
    report = generate_report([_passing()])
    assert "WARNINGS" in report


def test_report_partial_state_eval_appears_in_warnings():
    result = _partial_state(eval_id="eval_005")
    report = generate_report([result])
    warnings_section = report.split("WARNINGS")[-1]
    assert "eval_005" in warnings_section


def test_report_partial_state_warning_mentions_unsupported_state():
    result = _partial_state(eval_id="eval_017")
    report = generate_report([result])
    warnings_section = report.split("WARNINGS")[-1]
    assert "partial_answer_needs_clarification" in warnings_section


def test_report_non_partial_eval_not_in_warnings():
    result = _passing(eval_id="eval_001")
    report = generate_report([result])
    warnings_section = report.split("WARNINGS")[-1]
    assert "eval_001" not in warnings_section
