from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator

from src.agent.rechunker import RechunkResult


# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #

class GoldenEval(BaseModel):
    eval_id: str
    framework_id: str
    question: str
    expected_answer: str

    @field_validator("framework_id", mode="before")
    @classmethod
    def uppercase_framework_id(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def question_not_empty(self) -> "GoldenEval":
        if not self.question.strip():
            raise ValueError("question must not be empty")
        return self


@dataclass(frozen=True)
class FlaggedEval:
    eval_id: str
    reason: str


@dataclass(frozen=True)
class FlaggingResult:
    rechunk_result: RechunkResult
    flagged: list[FlaggedEval]
    unflagged_count: int


# ------------------------------------------------------------------ #
# Flagger
# ------------------------------------------------------------------ #

class EvalFlagger:
    def __init__(self, evals: list[GoldenEval]) -> None:
        self._evals = evals

    def flag(self, *, rechunk_result: RechunkResult) -> FlaggingResult:
        framework_id = rechunk_result.framework_id
        flagged: list[FlaggedEval] = []
        unflagged = 0

        for eval_ in self._evals:
            if eval_.framework_id == framework_id:
                flagged.append(
                    FlaggedEval(
                        eval_id=eval_.eval_id,
                        reason=(
                            f"{framework_id} source content was re-chunked "
                            f"({rechunk_result.chunk_count} chunks updated) — "
                            "expected answer may no longer be valid"
                        ),
                    )
                )
            else:
                unflagged += 1

        return FlaggingResult(
            rechunk_result=rechunk_result,
            flagged=flagged,
            unflagged_count=unflagged,
        )

    @classmethod
    def from_dicts(cls, raw: list[dict]) -> "EvalFlagger":
        evals: list[GoldenEval] = []
        for i, entry in enumerate(raw):
            try:
                evals.append(GoldenEval(**entry))
            except Exception as exc:
                raise ValueError(f"Invalid eval at index {i}: {exc}") from exc
        return cls(evals)
