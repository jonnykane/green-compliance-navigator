from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


# ------------------------------------------------------------------ #
# Result types
# ------------------------------------------------------------------ #

class ClassificationSource(Enum):
    RULES = "rules"
    CLAUDE = "claude"


@dataclass(frozen=True)
class Material:
    reasoning: str
    source: ClassificationSource


@dataclass(frozen=True)
class NotMaterial:
    reasoning: str
    source: ClassificationSource


ClassificationResult = Material | NotMaterial


# ------------------------------------------------------------------ #
# Claude client protocol (injectable)
# ------------------------------------------------------------------ #

class ClaudeClassifierClient(Protocol):
    def classify(self, change_text: str) -> dict:
        ...


# ------------------------------------------------------------------ #
# Rules — patterns that are automatically material
# ------------------------------------------------------------------ #

_MATERIAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("numeric threshold (kWh/energy)", re.compile(r"\d[\d,]*\s*kWh", re.IGNORECASE)),
    ("turnover threshold", re.compile(r"£\s*\d[\d,.]*\s*(million|bn|billion|thousand)?", re.IGNORECASE)),
    ("employee count threshold", re.compile(r"\d[\d,]*\s*or more employees", re.IGNORECASE)),
    ("deadline / date change", re.compile(r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", re.IGNORECASE)),
    ("scope change", re.compile(r"\bscope\s+[123]\b.*\b(included|excluded|mandatory|required)\b", re.IGNORECASE)),
    ("penalty figure", re.compile(r"penalty.*£\s*\d[\d,]*|£\s*\d[\d,]*.*penalty", re.IGNORECASE)),
]


def _first_matching_rule(text: str) -> str | None:
    for label, pattern in _MATERIAL_PATTERNS:
        if pattern.search(text):
            return label
    return None


# ------------------------------------------------------------------ #
# Classifier
# ------------------------------------------------------------------ #

class MaterialityClassifier:
    def __init__(self, claude_client: ClaudeClassifierClient) -> None:
        self._claude = claude_client

    def classify(self, change_text: str) -> ClassificationResult:
        matched_rule = _first_matching_rule(change_text)
        if matched_rule:
            return Material(
                reasoning=f"Automatic: matched rule '{matched_rule}'",
                source=ClassificationSource.RULES,
            )

        response = self._claude.classify(change_text)
        verdict = response.get("verdict", "")
        reasoning = response.get("reasoning", "")

        if verdict == "material":
            return Material(reasoning=reasoning, source=ClassificationSource.CLAUDE)
        elif verdict == "not_material":
            return NotMaterial(reasoning=reasoning, source=ClassificationSource.CLAUDE)
        else:
            raise ValueError(
                f"Unexpected verdict from Claude classifier: {verdict!r}. "
                "Expected 'material' or 'not_material'."
            )
