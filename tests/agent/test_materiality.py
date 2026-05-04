import pytest
from src.agent.materiality import (
    MaterialityClassifier,
    ClassificationResult,
    Material,
    NotMaterial,
    ClassificationSource,
)


# ------------------------------------------------------------------ #
# Fake Claude client
# ------------------------------------------------------------------ #

class FakeClaudeClient:
    def __init__(self, verdict: str, reasoning: str = "Claude said so"):
        self._verdict = verdict
        self._reasoning = reasoning
        self.calls: list[dict] = []

    def classify(self, change_text: str) -> dict:
        self.calls.append({"change_text": change_text})
        return {"verdict": self._verdict, "reasoning": self._reasoning}


# ------------------------------------------------------------------ #
# Rules engine — automatically material
# ------------------------------------------------------------------ #

class TestRulesEngineMaterial:
    """Changes with threshold figures, deadlines, or scope definitions
    are automatically material — Claude is not called."""

    def _classifier(self, claude_client=None):
        return MaterialityClassifier(claude_client=claude_client or FakeClaudeClient("not_material"))

    def test_energy_threshold_figure_is_material(self):
        text = "Threshold changed from 40,000 kWh to 50,000 kWh per year"
        result = self._classifier().classify(text)
        assert isinstance(result, Material)
        assert result.source == ClassificationSource.RULES

    def test_turnover_threshold_is_material(self):
        text = "Qualifying turnover threshold raised to £36 million"
        result = self._classifier().classify(text)
        assert isinstance(result, Material)
        assert result.source == ClassificationSource.RULES

    def test_employee_count_threshold_is_material(self):
        text = "Applicable to organisations with 500 or more employees"
        result = self._classifier().classify(text)
        assert isinstance(result, Material)
        assert result.source == ClassificationSource.RULES

    def test_deadline_change_is_material(self):
        text = "Compliance deadline extended to 31 December 2026"
        result = self._classifier().classify(text)
        assert isinstance(result, Material)
        assert result.source == ClassificationSource.RULES

    def test_scope_expansion_is_material(self):
        text = "Scope 3 emissions now included in mandatory reporting"
        result = self._classifier().classify(text)
        assert isinstance(result, Material)
        assert result.source == ClassificationSource.RULES

    def test_penalty_figure_is_material(self):
        text = "Maximum civil penalty increased to £50,000"
        result = self._classifier().classify(text)
        assert isinstance(result, Material)
        assert result.source == ClassificationSource.RULES

    def test_rules_match_does_not_call_claude(self):
        claude = FakeClaudeClient("material")
        classifier = MaterialityClassifier(claude_client=claude)
        classifier.classify("Threshold changed from 40,000 kWh to 50,000 kWh")
        assert len(claude.calls) == 0


# ------------------------------------------------------------------ #
# Claude fallback — ambiguous prose
# ------------------------------------------------------------------ #

class TestClaudeFallback:
    def test_claude_material_verdict_returns_material(self):
        claude = FakeClaudeClient(verdict="material", reasoning="New obligation added")
        classifier = MaterialityClassifier(claude_client=claude)
        result = classifier.classify("Reporting entities should consider transition plans")
        assert isinstance(result, Material)
        assert result.source == ClassificationSource.CLAUDE

    def test_claude_not_material_verdict_returns_not_material(self):
        claude = FakeClaudeClient(verdict="not_material", reasoning="Cosmetic reword")
        classifier = MaterialityClassifier(claude_client=claude)
        result = classifier.classify("Updated guidance note on formatting of reports")
        assert isinstance(result, NotMaterial)
        assert result.source == ClassificationSource.CLAUDE

    def test_reasoning_is_preserved_from_claude(self):
        claude = FakeClaudeClient(verdict="material", reasoning="Added penalty clause")
        classifier = MaterialityClassifier(claude_client=claude)
        result = classifier.classify("Some ambiguous prose change")
        assert result.reasoning == "Added penalty clause"

    def test_ambiguous_prose_calls_claude(self):
        claude = FakeClaudeClient(verdict="not_material")
        classifier = MaterialityClassifier(claude_client=claude)
        classifier.classify("Minor clarification to guidance note wording")
        assert len(claude.calls) == 1

    def test_change_text_passed_to_claude(self):
        claude = FakeClaudeClient(verdict="not_material")
        classifier = MaterialityClassifier(claude_client=claude)
        classifier.classify("Some ambiguous prose change")
        assert claude.calls[0]["change_text"] == "Some ambiguous prose change"

    def test_unknown_claude_verdict_raises(self):
        claude = FakeClaudeClient(verdict="maybe")
        classifier = MaterialityClassifier(claude_client=claude)
        with pytest.raises(ValueError, match="verdict"):
            classifier.classify("Some change")


# ------------------------------------------------------------------ #
# Result types
# ------------------------------------------------------------------ #

class TestClassificationResult:
    def test_material_carries_reasoning(self):
        result = Material(reasoning="threshold change", source=ClassificationSource.RULES)
        assert result.reasoning == "threshold change"

    def test_not_material_carries_reasoning(self):
        result = NotMaterial(reasoning="cosmetic", source=ClassificationSource.CLAUDE)
        assert result.reasoning == "cosmetic"
