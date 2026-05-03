import re
from dataclasses import dataclass, field
from typing import Protocol

from src.models import QueryResult

def _normalise_source(s: str) -> str:
    """Strip path prefix, lowercase, and strip whitespace for source comparison."""
    return s.strip().lower().split("/")[-1].split("\\")[-1]


# Maps QueryResult.kind to the expected_state vocabulary used in the golden set.
_KIND_TO_STATE: dict[str, str] = {
    "answer": "clear",
    "clarification_needed": "needs_clarification",
    "out_of_scope": "out_of_scope",
}

_PARTIAL_STATE = "partial_answer_needs_clarification"


class QueryEngineProtocol(Protocol):
    def ask(self, question: str, company_context: str = "") -> QueryResult: ...


@dataclass
class EvalResult:
    eval_id: str
    category: str
    question: str
    expected_state: str
    actual_state: str
    state_match: bool
    citation_accuracy: bool
    retrieval_recall: float
    facts_present: list[str]
    facts_missing: list[str]
    caveats_present: list[str]
    caveats_missing: list[str]
    forbidden_phrases_found: list[str]
    forbidden_certainty_found: list[str]
    hallucination_flag: bool
    passed: bool
    notes: str


class EvalRunner:
    def __init__(self, engine: QueryEngineProtocol) -> None:
        self._engine = engine

    def run_eval(self, eval_case: dict) -> EvalResult:
        question = eval_case["question"]
        company_context = eval_case.get("company_context", "")
        expected_state = eval_case["expected_state"]
        expected_citations = eval_case.get("expected_citations", [])
        expected_retrieved_documents = eval_case.get("expected_retrieved_documents", [])
        expected_facts = eval_case.get("expected_facts", [])
        required_caveats = eval_case.get("required_caveats", [])
        must_not_contain = eval_case.get("must_not_contain", [])
        forbidden_certainty_list = eval_case.get("forbidden_certainty", [])

        query_result = self._engine.ask(question, company_context)

        actual_state = _KIND_TO_STATE.get(query_result.kind, query_result.kind)
        answer_text = query_result.answer or ""
        sources = query_result.sources or []

        notes, state_match = self._score_state(
            expected_state, actual_state, expected_facts, expected_citations,
            answer_text, sources,
        )

        citation_accuracy = self._score_citation_accuracy(expected_citations, sources)
        retrieval_recall = self._score_retrieval_recall(expected_retrieved_documents, sources)

        facts_present, facts_missing = self._score_facts(expected_facts, answer_text)
        caveats_present, caveats_missing = self._score_presence(required_caveats, answer_text)

        forbidden_phrases_found = self._find_forbidden(must_not_contain, answer_text)
        forbidden_certainty_found = self._find_forbidden(forbidden_certainty_list, answer_text)
        hallucination_flag = bool(forbidden_phrases_found or forbidden_certainty_found)

        passed = state_match and citation_accuracy and not hallucination_flag

        return EvalResult(
            eval_id=eval_case["id"],
            category=eval_case.get("category", ""),
            question=question,
            expected_state=expected_state,
            actual_state=actual_state,
            state_match=state_match,
            citation_accuracy=citation_accuracy,
            retrieval_recall=retrieval_recall,
            facts_present=facts_present,
            facts_missing=facts_missing,
            caveats_present=caveats_present,
            caveats_missing=caveats_missing,
            forbidden_phrases_found=forbidden_phrases_found,
            forbidden_certainty_found=forbidden_certainty_found,
            hallucination_flag=hallucination_flag,
            passed=passed,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_state(
        self,
        expected_state: str,
        actual_state: str,
        expected_facts: list[str],
        expected_citations: list[str],
        answer_text: str,
        sources: list[str],
    ) -> tuple[str, bool]:
        if expected_state != _PARTIAL_STATE:
            return "", actual_state == expected_state

        # Approximate scoring for partial_answer_needs_clarification.
        fact_hit = any(f.lower() in answer_text.lower() for f in expected_facts)
        citation_hit = any(c in sources for c in expected_citations)
        state_match = actual_state in ("clear", "needs_clarification") and fact_hit and citation_hit
        notes = (
            f"WARNING: partial_answer_needs_clarification is not natively supported by the "
            f"Phase 3 classifier. Scored approximately — actual_state={actual_state!r}, "
            f"fact_hit={fact_hit}, citation_hit={citation_hit}."
        )
        return notes, state_match

    @staticmethod
    def _score_citation_accuracy(expected_citations: list[str], sources: list[str]) -> bool:
        if not expected_citations:
            return True
        norm_sources = {_normalise_source(s) for s in sources}
        return all(_normalise_source(c) in norm_sources for c in expected_citations)

    @staticmethod
    def _score_retrieval_recall(
        expected_retrieved_documents: list[str], sources: list[str]
    ) -> float:
        if not expected_retrieved_documents:
            return 1.0
        norm_sources = {_normalise_source(s) for s in sources}
        found = sum(1 for d in expected_retrieved_documents if _normalise_source(d) in norm_sources)
        return found / len(expected_retrieved_documents)

    @staticmethod
    def _score_presence(phrases: list[str], text: str) -> tuple[list[str], list[str]]:
        present, missing = [], []
        lower_text = text.lower()
        for phrase in phrases:
            (present if phrase.lower() in lower_text else missing).append(phrase)
        return present, missing

    @staticmethod
    def _score_fact_presence(fact: str, answer: str) -> bool:
        """
        Score a fact as present if key terms from the fact appear in the answer.
        Tries exact substring match first; falls back to key-term matching on
        numbers, currency amounts, and significant words (>4 chars, non-stop).
        """
        answer_lower = answer.lower()
        if fact.lower() in answer_lower:
            return True

        _STOP_WORDS = {
            "that", "this", "with", "from", "they", "have", "been",
            "their", "which", "will", "would", "could", "should",
            "applies", "apply", "whether", "because", "likely",
            "already", "criterion", "criteria", "employees",
            "turnover", "threshold", "company", "companies",
        }
        key_terms: list[str] = []
        key_terms.extend(re.findall(r"[£€]?\d+[.]?\d*[bmk]?", fact.lower()))
        words = re.findall(r"\b[a-z]{5,}\b", fact.lower())
        key_terms.extend(w for w in words if w not in _STOP_WORDS)
        key_terms = list(set(key_terms))

        if len(key_terms) < 2:
            return False

        matches = sum(1 for term in key_terms if term in answer_lower)
        return matches >= max(2, len(key_terms) // 2)

    @classmethod
    def _score_facts(cls, expected_facts: list[str], answer_text: str) -> tuple[list[str], list[str]]:
        present, missing = [], []
        for fact in expected_facts:
            (present if cls._score_fact_presence(fact, answer_text) else missing).append(fact)
        return present, missing

    @staticmethod
    def _find_forbidden(phrases: list[str], text: str) -> list[str]:
        lower_text = text.lower()
        return [p for p in phrases if p.lower() in lower_text]
