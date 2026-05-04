from __future__ import annotations

from src.agent.react_loop import RunReport, StepOutcome


class Summariser:
    def summarise(self, report: RunReport) -> str:
        lines: list[str] = []

        date_str = report.started_at.strftime("%Y-%m-%d %H:%M UTC")
        elapsed = (report.finished_at - report.started_at).total_seconds()

        lines.append(f"Regulatory Change Monitor — {date_str} ({elapsed:.1f}s)")
        lines.append(f"Sources checked: {report.total_sources}")
        lines.append("")

        updated = [t for t in report.traces if t.outcome == StepOutcome.UPDATED]
        errors = [t for t in report.traces if t.outcome == StepOutcome.FETCH_ERROR]
        not_material = [t for t in report.traces if t.outcome == StepOutcome.NOT_MATERIAL]

        if not updated and not errors:
            lines.append("No material changes detected.")
        else:
            if updated:
                lines.append(f"Updated ({len(updated)}):")
                for t in updated:
                    lines.append(f"  • {t.framework_id}: {t.observation}")
                    if t.flagged_eval_ids:
                        lines.append(f"    Flagged evals: {', '.join(t.flagged_eval_ids)}")

            if errors:
                lines.append(f"Errors ({len(errors)}):")
                for t in errors:
                    lines.append(f"  • {t.framework_id}: {t.observation}")

        if not_material:
            lines.append(f"Not material ({len(not_material)}): "
                         + ", ".join(t.framework_id for t in not_material))

        return "\n".join(lines)
