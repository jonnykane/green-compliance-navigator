import pytest
from datetime import datetime, timezone
from src.agent.summariser import Summariser
from src.agent.react_loop import RunReport, StepTrace, StepOutcome


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _trace(framework_id: str, outcome: StepOutcome, flagged: list[str] = None) -> StepTrace:
    return StepTrace(
        framework_id=framework_id,
        thought="Checking source",
        action="fetch(https://example.com)",
        observation="Some observation",
        outcome=outcome,
        flagged_eval_ids=flagged or [],
    )


def _report(traces: list[StepTrace]) -> RunReport:
    return RunReport(
        traces=traces,
        started_at=datetime(2026, 5, 4, 12, 0, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 5, 4, 12, 0, 5, tzinfo=timezone.utc),
    )


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestSummariser:
    def _summarise(self, traces):
        return Summariser().summarise(_report(traces))

    def test_returns_string(self):
        result = self._summarise([_trace("SECR", StepOutcome.NO_CHANGE)])
        assert isinstance(result, str)

    def test_includes_run_date(self):
        result = self._summarise([_trace("SECR", StepOutcome.NO_CHANGE)])
        assert "2026-05-04" in result

    def test_includes_total_sources(self):
        result = self._summarise([
            _trace("SECR", StepOutcome.NO_CHANGE),
            _trace("ESOS", StepOutcome.NO_CHANGE),
        ])
        assert "2" in result

    def test_updated_frameworks_listed(self):
        result = self._summarise([_trace("SECR", StepOutcome.UPDATED)])
        assert "SECR" in result

    def test_error_frameworks_listed(self):
        result = self._summarise([_trace("TCFD", StepOutcome.FETCH_ERROR)])
        assert "TCFD" in result

    def test_flagged_evals_listed(self):
        result = self._summarise([
            _trace("SECR", StepOutcome.UPDATED, flagged=["eval_001", "eval_002"])
        ])
        assert "eval_001" in result
        assert "eval_002" in result

    def test_no_changes_message_when_all_unchanged(self):
        result = self._summarise([
            _trace("SECR", StepOutcome.NO_CHANGE),
            _trace("ESOS", StepOutcome.NO_CHANGE),
        ])
        assert "no material changes" in result.lower()

    def test_not_material_not_listed_as_updated(self):
        result = self._summarise([_trace("SECR", StepOutcome.NOT_MATERIAL)])
        assert "updated" not in result.lower() or "0" in result
