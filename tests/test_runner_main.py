"""
Tests for the src/evals/runner.py __main__ block.

Uses subprocess so the block runs in a real interpreter with the project
root as CWD — no live API calls because the golden set is loaded from disk
and we inject a fake engine via --dry-run flag (defined inside __main__).

The __main__ tests are deliberately thin: they verify the block runs without
error, writes output to stdout, and produces a results file. The underlying
EvalRunner, print_report, and scoring logic are covered by their own test
files.
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_main(*extra_args: str) -> subprocess.CompletedProcess:
    """Run `python -m src.evals.runner` in the project root."""
    return subprocess.run(
        [sys.executable, "-m", "src.evals.runner", *extra_args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


# ---------------------------------------------------------------------------
# __main__ — dry-run mode (no live API calls)
# ---------------------------------------------------------------------------

def test_runner_main_dry_run_exits_zero():
    """python -m src.evals.runner --dry-run must exit with code 0."""
    result = _run_main("--dry-run")
    assert result.returncode == 0, result.stderr


def test_runner_main_dry_run_prints_summary_section():
    """stdout must contain the SUMMARY section from generate_report."""
    result = _run_main("--dry-run")
    assert "SUMMARY" in result.stdout


def test_runner_main_dry_run_prints_failures_section():
    """stdout must contain the FAILURES section."""
    result = _run_main("--dry-run")
    assert "FAILURES" in result.stdout


def test_runner_main_dry_run_writes_results_json(tmp_path):
    """--results-dir flag must write a JSON results file to the given directory."""
    result = _run_main("--dry-run", "--results-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    json_files = list(tmp_path.glob("run_*.json"))
    assert len(json_files) == 1, f"Expected 1 results file, got: {json_files}"


def test_runner_main_results_json_is_valid(tmp_path):
    """The written JSON must be a list of eval result dicts."""
    _run_main("--dry-run", "--results-dir", str(tmp_path))
    json_files = list(tmp_path.glob("run_*.json"))
    payload = json.loads(json_files[0].read_text())
    assert isinstance(payload, list)
    assert len(payload) > 0
    assert "eval_id" in payload[0]
    assert "passed" in payload[0]


def test_runner_main_results_json_includes_retrieved_chunks(tmp_path):
    """Each result dict must carry the retrieved_chunks field."""
    _run_main("--dry-run", "--results-dir", str(tmp_path))
    json_files = list(tmp_path.glob("run_*.json"))
    payload = json.loads(json_files[0].read_text())
    assert "retrieved_chunks" in payload[0]
