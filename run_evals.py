"""
Eval runner CLI for the UK Green Compliance Navigator.

Usage:
    python run_evals.py              # live run against real APIs
    python run_evals.py --dry-run    # static fake engine, no API calls
"""
import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

from src.evals.reporter import generate_report
from src.evals.runner import EvalResult, EvalRunner
from src.models import QueryResult

GOLDEN_SET_PATH = Path("evals/golden_set.json")
RESULTS_DIR = Path("evals/results")

# ---------------------------------------------------------------------------
# Dry-run fake engine
# ---------------------------------------------------------------------------

_DRY_RUN_ANSWER = (
    "Based on the available sources, SECR applies to large UK companies meeting two of "
    "three criteria: 250 or more employees, £36 million or more annual turnover, or "
    "£18 million or more balance sheet total. ESOS applies to organisations with 250 or "
    "more employees OR meeting the financial thresholds (£44m turnover and £38m balance "
    "sheet). Subject to company type, group structure, and professional advice for "
    "definitive confirmation. Subject to consultation and indicative timeline only."
)

_DRY_RUN_SOURCES = [
    "secr_environmental_reporting_guidelines_2019.pdf",
    "esos_overview.md",
]


class _DryRunQueryEngine:
    """Returns a static answer for every question. No API calls made."""

    def ask(self, question: str, company_context: str = "") -> QueryResult:
        return QueryResult(
            kind="answer",
            answer=_DRY_RUN_ANSWER,
            sources=_DRY_RUN_SOURCES,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the UK Green Compliance Navigator eval suite."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score against a static fake engine without making real API calls.",
    )
    args = parser.parse_args()

    golden = _load_golden_set()
    eval_cases = golden["evals"]
    total = len(eval_cases)

    if args.dry_run:
        print(f"[dry-run] Using static fake engine — no API calls will be made.")
        engine = _DryRunQueryEngine()
    else:
        print("Wiring real QueryEngine via wiring.py …")
        from src.wiring import build_query_engine
        engine = build_query_engine()

    runner = EvalRunner(engine=engine)
    results: list[EvalResult] = []

    print(f"Running {total} evals …\n")
    for i, case in enumerate(eval_cases, 1):
        print(f"  [{i:>2}/{total}] {case['id']} — {case['question'][:60]}", end="", flush=True)
        result = runner.run_eval(case)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}]")

    report = generate_report(results)
    print()
    print(report)

    _save_results(results)

    pass_rate = sum(1 for r in results if r.passed) / total if total else 0.0
    sys.exit(0 if pass_rate >= 0.70 else 1)


def _load_golden_set() -> dict:
    if not GOLDEN_SET_PATH.exists():
        print(f"ERROR: golden set not found at {GOLDEN_SET_PATH}", file=sys.stderr)
        sys.exit(1)
    with GOLDEN_SET_PATH.open() as f:
        return json.load(f)


def _save_results(results: list[EvalResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"run_{timestamp}.json"
    payload = [dataclasses.asdict(r) for r in results]
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
