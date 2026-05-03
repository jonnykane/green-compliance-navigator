from collections import defaultdict

from src.evals.runner import EvalResult

_PARTIAL_STATE = "partial_answer_needs_clarification"
_WIDTH = 72


def generate_report(results: list[EvalResult]) -> str:
    lines: list[str] = []

    lines.append("=" * _WIDTH)
    lines.append("UK GREEN COMPLIANCE NAVIGATOR — EVAL REPORT")
    lines.append("=" * _WIDTH)

    _add_summary(lines, results)
    _add_by_category(lines, results)
    _add_failures(lines, results)
    _add_hallucination_flags(lines, results)
    _add_warnings(lines, results)

    lines.append("=" * _WIDTH)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _add_summary(lines: list[str], results: list[EvalResult]) -> None:
    n = len(results)
    passed = sum(1 for r in results if r.passed)
    state_matches = sum(1 for r in results if r.state_match)
    citation_hits = sum(1 for r in results if r.citation_accuracy)
    mean_recall = sum(r.retrieval_recall for r in results) / n if n else 0.0
    hallucinations = sum(1 for r in results if r.hallucination_flag)

    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * _WIDTH)
    lines.append(f"  Total evals run       : {n}")
    lines.append(f"  Pass rate             : {_pct(passed, n)}")
    lines.append(f"  State match rate      : {_pct(state_matches, n)}")
    lines.append(f"  Citation accuracy rate: {_pct(citation_hits, n)}")
    lines.append(f"  Mean retrieval recall : {mean_recall * 100:.1f}%")
    lines.append(f"  Hallucination rate    : {_pct(hallucinations, n)}")
    lines.append("")


def _add_by_category(lines: list[str], results: list[EvalResult]) -> None:
    by_cat: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    lines.append("BY CATEGORY")
    lines.append("-" * _WIDTH)
    for category, group in sorted(by_cat.items()):
        n = len(group)
        passed = sum(1 for r in group if r.passed)
        lines.append(f"  {category:<40} {passed}/{n}  ({_pct(passed, n)})")
    lines.append("")


def _add_failures(lines: list[str], results: list[EvalResult]) -> None:
    failed = [r for r in results if not r.passed]

    lines.append("FAILURES")
    lines.append("-" * _WIDTH)
    if not failed:
        lines.append("  None.")
    else:
        for r in failed:
            q_short = r.question[:60] + ("..." if len(r.question) > 60 else "")
            reasons: list[str] = []
            if not r.state_match:
                reasons.append(f"state_match (expected={r.expected_state!r}, actual={r.actual_state!r})")
            if not r.citation_accuracy:
                reasons.append("citation_accuracy")
            if r.hallucination_flag:
                reasons.append("hallucination")
            lines.append(f"  {r.eval_id} | {q_short!r}")
            lines.append(f"    Failed: {', '.join(reasons)}")
    lines.append("")


def _add_hallucination_flags(lines: list[str], results: list[EvalResult]) -> None:
    flagged = [r for r in results if r.hallucination_flag]

    lines.append("HALLUCINATION FLAGS")
    lines.append("-" * _WIDTH)
    if not flagged:
        lines.append("  None.")
    else:
        for r in flagged:
            lines.append(f"  {r.eval_id} | {r.question[:60]!r}")
            for phrase in r.forbidden_phrases_found:
                lines.append(f"    must_not_contain    : {phrase!r}")
            for phrase in r.forbidden_certainty_found:
                lines.append(f"    forbidden_certainty : {phrase!r}")
    lines.append("")


def _add_warnings(lines: list[str], results: list[EvalResult]) -> None:
    partial = [r for r in results if r.expected_state == _PARTIAL_STATE]

    lines.append("WARNINGS")
    lines.append("-" * _WIDTH)
    if not partial:
        lines.append("  None.")
    else:
        for r in partial:
            lines.append(
                f"  {r.eval_id} | partial_answer_needs_clarification not natively "
                f"supported by Phase 3 classifier."
            )
            lines.append(f"    actual_state={r.actual_state!r} | state_match={r.state_match} | {r.notes}")
    lines.append("")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator * 100:.1f}%"
