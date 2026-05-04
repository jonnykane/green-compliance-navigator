from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.agent.eval_flagger import EvalFlagger, FlaggingResult
from src.agent.fetcher import Changed, FetchError, Fetcher, Unchanged
from src.agent.materiality import Material, MaterialityClassifier
from src.agent.rechunker import Rechunker
from src.agent.registry import FrameworkRegistry


# ------------------------------------------------------------------ #
# Trace types
# ------------------------------------------------------------------ #

class StepOutcome(Enum):
    NO_CHANGE = "no_change"
    NOT_MATERIAL = "not_material"
    UPDATED = "updated"
    FETCH_ERROR = "fetch_error"


@dataclass
class StepTrace:
    framework_id: str
    thought: str
    action: str
    observation: str
    outcome: StepOutcome
    flagged_eval_ids: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    traces: list[StepTrace]
    started_at: datetime
    finished_at: datetime

    @property
    def total_sources(self) -> int:
        return len(self.traces)

    @property
    def updated_count(self) -> int:
        return sum(1 for t in self.traces if t.outcome == StepOutcome.UPDATED)

    @property
    def error_count(self) -> int:
        return sum(1 for t in self.traces if t.outcome == StepOutcome.FETCH_ERROR)


# ------------------------------------------------------------------ #
# Agent
# ------------------------------------------------------------------ #

class MonitoringAgent:
    def __init__(
        self,
        registry: FrameworkRegistry,
        fetcher: Fetcher,
        classifier: MaterialityClassifier,
        rechunker: Rechunker,
        flagger: EvalFlagger,
    ) -> None:
        self._registry = registry
        self._fetcher = fetcher
        self._classifier = classifier
        self._rechunker = rechunker
        self._flagger = flagger

    def run(self) -> RunReport:
        started_at = datetime.now(tz=timezone.utc)
        traces: list[StepTrace] = []

        for source in self._registry.all_sources():
            for url in source.urls:
                trace = self._process(source.framework_id, url, source)
                traces.append(trace)
                break  # one URL per source for now; multi-URL merging is a later concern

        return RunReport(
            traces=traces,
            started_at=started_at,
            finished_at=datetime.now(tz=timezone.utc),
        )

    def _process(self, framework_id: str, url: str, source) -> StepTrace:
        # --- Thought ---
        thought = (
            f"Checking {framework_id} source at {url}. "
            f"Last known hash: {source.last_content_hash or 'none'}."
        )

        # --- Action: Fetch ---
        action = f"fetch({url})"
        fetch_result = self._fetcher.fetch(
            url,
            previous_hash=source.last_content_hash,
            previous_etag=getattr(source, "last_etag", None),
        )

        # --- Observation ---
        if isinstance(fetch_result, FetchError):
            return StepTrace(
                framework_id=framework_id,
                thought=thought,
                action=action,
                observation=f"Fetch failed: {fetch_result.reason}",
                outcome=StepOutcome.FETCH_ERROR,
            )

        if isinstance(fetch_result, Unchanged):
            return StepTrace(
                framework_id=framework_id,
                thought=thought,
                action=action,
                observation="Content unchanged since last check.",
                outcome=StepOutcome.NO_CHANGE,
            )

        # Changed — classify
        action = f"fetch({url}) → classify"
        classification = self._classifier.classify(fetch_result.content)

        if isinstance(classification, Material):
            # Rechunk + flag
            rechunk_result = self._rechunker.rechunk(
                fetch_result.content,
                framework_id=framework_id,
            )
            flagging_result = self._flagger.flag(rechunk_result=rechunk_result)

            self._registry.update_check_result(
                framework_id,
                hash_value=fetch_result.content_hash,
                checked_at=datetime.now(tz=timezone.utc),
            )

            flagged_ids = [f.eval_id for f in flagging_result.flagged]
            obs = (
                f"Material change detected ({classification.source.value}: {classification.reasoning}). "
                f"Re-chunked into {rechunk_result.chunk_count} vectors. "
                f"Flagged evals: {flagged_ids or 'none'}."
            )
            return StepTrace(
                framework_id=framework_id,
                thought=thought,
                action=action,
                observation=obs,
                outcome=StepOutcome.UPDATED,
                flagged_eval_ids=flagged_ids,
            )

        # Not material
        obs = f"Change detected but not material ({classification.reasoning}). No update."
        return StepTrace(
            framework_id=framework_id,
            thought=thought,
            action=action,
            observation=obs,
            outcome=StepOutcome.NOT_MATERIAL,
        )
