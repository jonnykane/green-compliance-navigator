"""
Tests for api.py — FastAPI server wrapping QueryEngine.ask().

All tests use a FakeEngine injected via FastAPI's dependency_overrides;
no real API calls to Voyage AI, Anthropic, or Pinecone are made.
"""
import pytest
from fastapi.testclient import TestClient

from api import app, get_engine
from src.models import QueryResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEngine:
    """Returns a fixed QueryResult for every ask() call and records its inputs."""

    def __init__(self, result: QueryResult) -> None:
        self._result = result
        self.last_question: str = ""
        self.last_company_context: str = ""

    def ask(self, question: str, company_context: str = "") -> QueryResult:
        self.last_question = question
        self.last_company_context = company_context
        return self._result


class ErrorEngine:
    """Raises an exception on every ask() call."""

    def ask(self, question: str, company_context: str = "") -> QueryResult:
        raise RuntimeError("Upstream service unavailable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _answer_result(text: str = "ESOS applies to large undertakings.", sources: list[str] | None = None) -> QueryResult:
    return QueryResult(kind="answer", answer=text, sources=sources or ["esos.md"])


def _clarification_result() -> QueryResult:
    return QueryResult(
        kind="clarification_needed",
        clarification_question="Which best describes your organisation?",
        options=["Large UK company", "SME", "Listed company"],
    )


def _out_of_scope_result() -> QueryResult:
    return QueryResult(
        kind="out_of_scope",
        answer="This question is outside the scope of this tool.",
    )


def _inject(engine) -> TestClient:
    """Return a TestClient with the given engine injected as the dependency."""
    app.dependency_overrides[get_engine] = lambda: engine
    return TestClient(app)


# ---------------------------------------------------------------------------
# Fixture — always clean up dependency overrides between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_status_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /ask — answer kind
# ---------------------------------------------------------------------------

def test_ask_answer_kind_returns_200():
    client = _inject(FakeEngine(_answer_result()))
    response = client.post("/ask", json={"question": "What is ESOS?"})
    assert response.status_code == 200


def test_ask_answer_kind_in_response():
    client = _inject(FakeEngine(_answer_result()))
    data = client.post("/ask", json={"question": "What is ESOS?"}).json()
    assert data["kind"] == "answer"


def test_ask_answer_text_in_response():
    client = _inject(FakeEngine(_answer_result(text="ESOS applies.")))
    data = client.post("/ask", json={"question": "What is ESOS?"}).json()
    assert data["answer"] == "ESOS applies."


def test_ask_sources_in_response():
    client = _inject(FakeEngine(_answer_result(sources=["esos.md", "secr.pdf"])))
    data = client.post("/ask", json={"question": "q"}).json()
    assert data["sources"] == ["esos.md", "secr.pdf"]


# ---------------------------------------------------------------------------
# POST /ask — clarification_needed kind
# ---------------------------------------------------------------------------

def test_ask_clarification_kind_returns_200():
    client = _inject(FakeEngine(_clarification_result()))
    response = client.post("/ask", json={"question": "What do we need to report?"})
    assert response.status_code == 200


def test_ask_clarification_kind_in_response():
    client = _inject(FakeEngine(_clarification_result()))
    data = client.post("/ask", json={"question": "What do we need to report?"}).json()
    assert data["kind"] == "clarification_needed"


def test_ask_clarification_question_in_response():
    client = _inject(FakeEngine(_clarification_result()))
    data = client.post("/ask", json={"question": "What do we need to report?"}).json()
    assert data["clarification_question"] == "Which best describes your organisation?"


def test_ask_clarification_options_in_response():
    client = _inject(FakeEngine(_clarification_result()))
    data = client.post("/ask", json={"question": "What do we need to report?"}).json()
    assert data["options"] == ["Large UK company", "SME", "Listed company"]


# ---------------------------------------------------------------------------
# POST /ask — out_of_scope kind
# ---------------------------------------------------------------------------

def test_ask_out_of_scope_returns_200():
    client = _inject(FakeEngine(_out_of_scope_result()))
    response = client.post("/ask", json={"question": "How do I make pasta?"})
    assert response.status_code == 200


def test_ask_out_of_scope_kind_in_response():
    client = _inject(FakeEngine(_out_of_scope_result()))
    data = client.post("/ask", json={"question": "How do I make pasta?"}).json()
    assert data["kind"] == "out_of_scope"


def test_ask_out_of_scope_answer_in_response():
    client = _inject(FakeEngine(_out_of_scope_result()))
    data = client.post("/ask", json={"question": "How do I make pasta?"}).json()
    assert data["answer"] == "This question is outside the scope of this tool."


# ---------------------------------------------------------------------------
# POST /ask — request forwarding
# ---------------------------------------------------------------------------

def test_ask_forwards_question_to_engine():
    engine = FakeEngine(_answer_result())
    _inject(engine).post("/ask", json={"question": "What is SECR?"})
    assert engine.last_question == "What is SECR?"


def test_ask_forwards_company_context_to_engine():
    engine = FakeEngine(_answer_result())
    _inject(engine).post(
        "/ask",
        json={"question": "What applies to us?", "company_context": "We are a large UK company."},
    )
    assert engine.last_company_context == "We are a large UK company."


def test_ask_default_company_context_is_empty_string():
    engine = FakeEngine(_answer_result())
    _inject(engine).post("/ask", json={"question": "What is ESOS?"})
    assert engine.last_company_context == ""


# ---------------------------------------------------------------------------
# POST /ask — malformed request → 422
# ---------------------------------------------------------------------------

def test_ask_missing_question_field_returns_422():
    client = _inject(FakeEngine(_answer_result()))
    response = client.post("/ask", json={"company_context": "some context"})
    assert response.status_code == 422


def test_ask_empty_body_returns_422():
    client = _inject(FakeEngine(_answer_result()))
    response = client.post("/ask", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /ask — engine error → 500
# ---------------------------------------------------------------------------

def test_ask_engine_error_returns_500():
    client = _inject(ErrorEngine())
    response = client.post("/ask", json={"question": "What is ESOS?"})
    assert response.status_code == 500


def test_ask_engine_error_detail_in_response():
    client = _inject(ErrorEngine())
    data = client.post("/ask", json={"question": "What is ESOS?"}).json()
    assert "detail" in data
    assert "Upstream service unavailable" in data["detail"]
