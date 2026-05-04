import pytest
from src.agent.react_loop import MonitoringAgent, RunReport, StepTrace, StepOutcome
from src.agent.registry import FrameworkRegistry, FrameworkSource
from src.agent.fetcher import Changed, Unchanged, FetchError
from src.agent.materiality import Material, NotMaterial, ClassificationSource
from src.agent.rechunker import RechunkResult
from src.agent.eval_flagger import FlaggingResult, FlaggedEval, GoldenEval, EvalFlagger


# ------------------------------------------------------------------ #
# Fakes
# ------------------------------------------------------------------ #

def _source(framework_id: str, url: str = "https://example.com") -> FrameworkSource:
    return FrameworkSource(framework_id=framework_id, name=framework_id, urls=[url])


def _registry(*framework_ids: str) -> FrameworkRegistry:
    return FrameworkRegistry([_source(fid) for fid in framework_ids])


class FakeFetcher:
    def __init__(self, results: dict):
        # results: {url: FetchResult}
        self.calls: list[dict] = []
        self._results = results

    def fetch(self, url: str, *, previous_hash, previous_etag=None):
        self.calls.append({"url": url})
        return self._results.get(url, Unchanged(url=url))


class FakeMaterialityClassifier:
    def __init__(self, verdict):
        self._verdict = verdict
        self.calls: list[str] = []

    def classify(self, change_text: str):
        self.calls.append(change_text)
        if self._verdict == "material":
            return Material(reasoning="fake material", source=ClassificationSource.RULES)
        return NotMaterial(reasoning="fake not material", source=ClassificationSource.CLAUDE)


class FakeRechunker:
    def __init__(self):
        self.calls: list[dict] = []

    def rechunk(self, content: str, *, framework_id: str, previous_vector_ids=None):
        self.calls.append({"framework_id": framework_id})
        return RechunkResult(
            framework_id=framework_id,
            chunk_count=2,
            vector_ids=[f"{framework_id}_0_abc", f"{framework_id}_1_def"],
        )


class FakeEvalFlagger:
    def __init__(self, flagged_evals: list[FlaggedEval] = None):
        self._flagged = flagged_evals or []
        self.calls: list[RechunkResult] = []

    def flag(self, *, rechunk_result: RechunkResult) -> FlaggingResult:
        self.calls.append(rechunk_result)
        return FlaggingResult(
            rechunk_result=rechunk_result,
            flagged=self._flagged,
            unflagged_count=0,
        )


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _agent(
    registry=None,
    fetcher=None,
    classifier=None,
    rechunker=None,
    flagger=None,
):
    return MonitoringAgent(
        registry=registry or _registry("SECR"),
        fetcher=fetcher or FakeFetcher({}),
        classifier=classifier or FakeMaterialityClassifier("not_material"),
        rechunker=rechunker or FakeRechunker(),
        flagger=flagger or FakeEvalFlagger(),
    )


# ------------------------------------------------------------------ #
# Run structure
# ------------------------------------------------------------------ #

class TestRunStructure:
    def test_run_returns_report(self):
        report = _agent().run()
        assert isinstance(report, RunReport)

    def test_report_has_one_trace_per_source(self):
        agent = _agent(registry=_registry("SECR", "ESOS", "TCFD"))
        report = agent.run()
        assert len(report.traces) == 3

    def test_each_trace_has_framework_id(self):
        agent = _agent(registry=_registry("SECR", "ESOS"))
        report = agent.run()
        ids = {t.framework_id for t in report.traces}
        assert ids == {"SECR", "ESOS"}

    def test_trace_records_thought_action_observation(self):
        agent = _agent()
        report = agent.run()
        trace = report.traces[0]
        assert trace.thought
        assert trace.action
        assert trace.observation


# ------------------------------------------------------------------ #
# Unchanged path
# ------------------------------------------------------------------ #

class TestUnchangedPath:
    def test_unchanged_fetch_produces_no_change_outcome(self):
        fetcher = FakeFetcher({"https://example.com": Unchanged(url="https://example.com")})
        agent = _agent(registry=_registry("SECR"), fetcher=fetcher)
        report = agent.run()
        assert report.traces[0].outcome == StepOutcome.NO_CHANGE

    def test_unchanged_does_not_call_classifier(self):
        fetcher = FakeFetcher({"https://example.com": Unchanged(url="https://example.com")})
        classifier = FakeMaterialityClassifier("material")
        agent = _agent(registry=_registry("SECR"), fetcher=fetcher, classifier=classifier)
        agent.run()
        assert len(classifier.calls) == 0

    def test_unchanged_does_not_rechunk(self):
        fetcher = FakeFetcher({"https://example.com": Unchanged(url="https://example.com")})
        rechunker = FakeRechunker()
        agent = _agent(registry=_registry("SECR"), fetcher=fetcher, rechunker=rechunker)
        agent.run()
        assert len(rechunker.calls) == 0


# ------------------------------------------------------------------ #
# Fetch error path
# ------------------------------------------------------------------ #

class TestFetchErrorPath:
    def test_fetch_error_produces_error_outcome(self):
        fetcher = FakeFetcher({"https://example.com": FetchError(url="https://example.com", reason="timeout")})
        agent = _agent(registry=_registry("SECR"), fetcher=fetcher)
        report = agent.run()
        assert report.traces[0].outcome == StepOutcome.FETCH_ERROR

    def test_fetch_error_does_not_rechunk(self):
        fetcher = FakeFetcher({"https://example.com": FetchError(url="https://example.com", reason="timeout")})
        rechunker = FakeRechunker()
        agent = _agent(registry=_registry("SECR"), fetcher=fetcher, rechunker=rechunker)
        agent.run()
        assert len(rechunker.calls) == 0

    def test_fetch_error_reason_in_observation(self):
        fetcher = FakeFetcher({"https://example.com": FetchError(url="https://example.com", reason="timeout")})
        agent = _agent(registry=_registry("SECR"), fetcher=fetcher)
        report = agent.run()
        assert "timeout" in report.traces[0].observation


# ------------------------------------------------------------------ #
# Material change path
# ------------------------------------------------------------------ #

class TestMaterialChangePath:
    def _changed_fetcher(self):
        return FakeFetcher({
            "https://example.com": Changed(
                url="https://example.com",
                content="Threshold raised to 50,000 kWh",
                content_hash="newhash",
            )
        })

    def test_material_change_produces_updated_outcome(self):
        agent = _agent(
            fetcher=self._changed_fetcher(),
            classifier=FakeMaterialityClassifier("material"),
        )
        report = agent.run()
        assert report.traces[0].outcome == StepOutcome.UPDATED

    def test_material_change_calls_rechunker(self):
        rechunker = FakeRechunker()
        agent = _agent(
            fetcher=self._changed_fetcher(),
            classifier=FakeMaterialityClassifier("material"),
            rechunker=rechunker,
        )
        agent.run()
        assert len(rechunker.calls) == 1

    def test_material_change_calls_eval_flagger(self):
        flagger = FakeEvalFlagger()
        agent = _agent(
            fetcher=self._changed_fetcher(),
            classifier=FakeMaterialityClassifier("material"),
            flagger=flagger,
        )
        agent.run()
        assert len(flagger.calls) == 1

    def test_flagged_evals_in_trace(self):
        flagged = [FlaggedEval(eval_id="e1", reason="SECR content changed")]
        agent = _agent(
            fetcher=self._changed_fetcher(),
            classifier=FakeMaterialityClassifier("material"),
            flagger=FakeEvalFlagger(flagged_evals=flagged),
        )
        report = agent.run()
        assert report.traces[0].flagged_eval_ids == ["e1"]

    def test_registry_updated_after_material_change(self):
        registry = _registry("SECR")
        agent = _agent(
            registry=registry,
            fetcher=self._changed_fetcher(),
            classifier=FakeMaterialityClassifier("material"),
        )
        agent.run()
        source = registry.get_source("SECR")
        assert source.last_content_hash == "newhash"


# ------------------------------------------------------------------ #
# Not-material change path
# ------------------------------------------------------------------ #

class TestNotMaterialPath:
    def test_not_material_produces_no_change_outcome(self):
        fetcher = FakeFetcher({
            "https://example.com": Changed(
                url="https://example.com",
                content="Cosmetic reword",
                content_hash="newhash",
            )
        })
        agent = _agent(
            fetcher=fetcher,
            classifier=FakeMaterialityClassifier("not_material"),
        )
        report = agent.run()
        assert report.traces[0].outcome == StepOutcome.NOT_MATERIAL

    def test_not_material_does_not_rechunk(self):
        fetcher = FakeFetcher({
            "https://example.com": Changed(
                url="https://example.com",
                content="Cosmetic reword",
                content_hash="newhash",
            )
        })
        rechunker = FakeRechunker()
        agent = _agent(fetcher=fetcher, rechunker=rechunker,
                       classifier=FakeMaterialityClassifier("not_material"))
        agent.run()
        assert len(rechunker.calls) == 0


# ------------------------------------------------------------------ #
# Run summary
# ------------------------------------------------------------------ #

class TestRunSummary:
    def test_report_counts_updated(self):
        fetcher = FakeFetcher({
            "https://example.com": Changed(
                url="https://example.com", content="x", content_hash="h"
            )
        })
        agent = _agent(
            registry=_registry("SECR"),
            fetcher=fetcher,
            classifier=FakeMaterialityClassifier("material"),
        )
        report = agent.run()
        assert report.updated_count == 1

    def test_report_counts_errors(self):
        fetcher = FakeFetcher({
            "https://example.com": FetchError(url="https://example.com", reason="err")
        })
        report = _agent(registry=_registry("SECR"), fetcher=fetcher).run()
        assert report.error_count == 1

    def test_report_total_matches_registry_size(self):
        agent = _agent(registry=_registry("SECR", "ESOS", "TCFD"))
        report = agent.run()
        assert report.total_sources == 3
