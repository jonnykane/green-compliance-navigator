import pytest
from src.agent.eval_flagger import (
    EvalFlagger,
    GoldenEval,
    FlaggingResult,
    FlaggedEval,
)
from src.agent.rechunker import RechunkResult


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _result(framework_id: str, chunk_count: int = 2) -> RechunkResult:
    return RechunkResult(
        framework_id=framework_id,
        chunk_count=chunk_count,
        vector_ids=[f"{framework_id}_{i}_abc" for i in range(chunk_count)],
    )


def _eval(eval_id: str, framework_id: str, question: str = "What is the threshold?") -> GoldenEval:
    return GoldenEval(
        eval_id=eval_id,
        framework_id=framework_id,
        question=question,
        expected_answer="Some expected answer",
    )


# ------------------------------------------------------------------ #
# GoldenEval model
# ------------------------------------------------------------------ #

class TestGoldenEval:
    def test_construction(self):
        e = _eval("eval_001", "SECR")
        assert e.eval_id == "eval_001"
        assert e.framework_id == "SECR"

    def test_framework_id_uppercased(self):
        e = _eval("eval_001", "secr")
        assert e.framework_id == "SECR"

    def test_requires_non_empty_question(self):
        with pytest.raises(ValueError):
            GoldenEval(
                eval_id="e1",
                framework_id="SECR",
                question="",
                expected_answer="answer",
            )


# ------------------------------------------------------------------ #
# Flagging logic
# ------------------------------------------------------------------ #

class TestEvalFlagger:
    def test_no_evals_for_framework_returns_empty(self):
        flagger = EvalFlagger(evals=[_eval("e1", "ESOS")])
        result = flagger.flag(rechunk_result=_result("SECR"))
        assert result.flagged == []

    def test_matching_framework_id_is_flagged(self):
        flagger = EvalFlagger(evals=[_eval("e1", "SECR")])
        result = flagger.flag(rechunk_result=_result("SECR"))
        assert len(result.flagged) == 1
        assert result.flagged[0].eval_id == "e1"

    def test_non_matching_framework_id_not_flagged(self):
        flagger = EvalFlagger(evals=[_eval("e1", "ESOS"), _eval("e2", "TCFD")])
        result = flagger.flag(rechunk_result=_result("SECR"))
        assert result.flagged == []

    def test_multiple_evals_same_framework_all_flagged(self):
        flagger = EvalFlagger(evals=[
            _eval("e1", "SECR"),
            _eval("e2", "SECR"),
            _eval("e3", "ESOS"),
        ])
        result = flagger.flag(rechunk_result=_result("SECR"))
        flagged_ids = {f.eval_id for f in result.flagged}
        assert flagged_ids == {"e1", "e2"}

    def test_flagged_eval_includes_reason(self):
        flagger = EvalFlagger(evals=[_eval("e1", "SECR")])
        result = flagger.flag(rechunk_result=_result("SECR"))
        assert result.flagged[0].reason != ""

    def test_flagged_eval_references_framework(self):
        flagger = EvalFlagger(evals=[_eval("e1", "SECR")])
        result = flagger.flag(rechunk_result=_result("SECR"))
        assert "SECR" in result.flagged[0].reason

    def test_result_includes_rechunk_result(self):
        rechunk = _result("SECR")
        flagger = EvalFlagger(evals=[_eval("e1", "SECR")])
        result = flagger.flag(rechunk_result=rechunk)
        assert result.rechunk_result is rechunk

    def test_unflagged_count_is_correct(self):
        flagger = EvalFlagger(evals=[
            _eval("e1", "SECR"),
            _eval("e2", "ESOS"),
        ])
        result = flagger.flag(rechunk_result=_result("SECR"))
        assert result.unflagged_count == 1


# ------------------------------------------------------------------ #
# Loading from list of dicts
# ------------------------------------------------------------------ #

class TestEvalFlaggerFromDicts:
    def test_loads_from_list_of_dicts(self):
        raw = [
            {
                "eval_id": "e1",
                "framework_id": "SECR",
                "question": "What is the threshold?",
                "expected_answer": "40,000 kWh",
            }
        ]
        flagger = EvalFlagger.from_dicts(raw)
        result = flagger.flag(rechunk_result=_result("SECR"))
        assert len(result.flagged) == 1

    def test_invalid_dict_raises(self):
        with pytest.raises(ValueError, match="eval"):
            EvalFlagger.from_dicts([{"eval_id": "e1"}])  # missing fields
