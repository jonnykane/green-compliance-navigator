"""
Tests for api.py — FastAPI server wrapping QueryEngine.ask().

All tests use a FakeEngine injected via FastAPI's dependency_overrides;
no real API calls to Voyage AI, Anthropic, or Pinecone are made.
"""
import pytest
from fastapi.testclient import TestClient

from api import app, get_engine, _get_allowed_origins
from src.config import settings as settings_module
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


# ---------------------------------------------------------------------------
# Change 1 — _get_allowed_origins() unit tests
# ---------------------------------------------------------------------------

def test_get_allowed_origins_includes_lovable_in_production(monkeypatch):
    monkeypatch.setattr(settings_module, "ENVIRONMENT", "production")
    assert "https://green-nav-guide.lovable.app" in _get_allowed_origins()


def test_get_allowed_origins_excludes_localhost_in_production(monkeypatch):
    monkeypatch.setattr(settings_module, "ENVIRONMENT", "production")
    origins = _get_allowed_origins()
    assert not any("localhost" in o for o in origins)


def test_get_allowed_origins_includes_localhost_in_development(monkeypatch):
    monkeypatch.setattr(settings_module, "ENVIRONMENT", "development")
    origins = _get_allowed_origins()
    assert any("localhost" in o for o in origins)


def test_get_allowed_origins_still_includes_lovable_in_development(monkeypatch):
    monkeypatch.setattr(settings_module, "ENVIRONMENT", "development")
    assert "https://green-nav-guide.lovable.app" in _get_allowed_origins()


# ---------------------------------------------------------------------------
# Change 1 — CORS middleware behaviour (via preflight requests)
# ---------------------------------------------------------------------------

def test_cors_lovable_origin_allowed_in_preflight():
    client = TestClient(app)
    response = client.options(
        "/ask",
        headers={
            "Origin": "https://green-nav-guide.lovable.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "https://green-nav-guide.lovable.app"


def test_cors_unknown_origin_denied_in_preflight():
    client = TestClient(app)
    response = client.options(
        "/ask",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_cors_post_and_get_in_allowed_methods():
    client = TestClient(app)
    response = client.options(
        "/ask",
        headers={
            "Origin": "https://green-nav-guide.lovable.app",
            "Access-Control-Request-Method": "POST",
        },
    )
    allow_methods = response.headers.get("access-control-allow-methods", "")
    assert "POST" in allow_methods
    assert "GET" in allow_methods


def test_cors_credentials_not_allowed():
    client = TestClient(app)
    response = client.options(
        "/ask",
        headers={
            "Origin": "https://green-nav-guide.lovable.app",
            "Access-Control-Request-Method": "POST",
        },
    )
    # allow_credentials=False means this header should be absent or explicitly "false"
    creds = response.headers.get("access-control-allow-credentials", "false")
    assert creds.lower() != "true"


# ---------------------------------------------------------------------------
# Change 2 — Global unhandled exception handler
# ---------------------------------------------------------------------------

class _RaisingDependency:
    """Raises a raw RuntimeError when called as a FastAPI dependency.

    Because the exception is raised in the dependency (before the route's
    try/except runs), it reaches the global exception handler directly.
    """
    _SENTINEL = "sentinel_xyz_not_for_clients"

    def __call__(self):
        raise RuntimeError(self._SENTINEL)


_raising_dep = _RaisingDependency()


def test_unhandled_exception_returns_500():
    app.dependency_overrides[get_engine] = _raising_dep
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/ask", json={"question": "test"})
    assert response.status_code == 500


def test_unhandled_exception_body_is_clean_json():
    app.dependency_overrides[get_engine] = _raising_dep
    client = TestClient(app, raise_server_exceptions=False)
    data = client.post("/ask", json={"question": "test"}).json()
    assert data == {"error": "An unexpected error occurred. Please try again."}


def test_unhandled_exception_does_not_expose_internal_details():
    app.dependency_overrides[get_engine] = _raising_dep
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/ask", json={"question": "test"})
    assert _RaisingDependency._SENTINEL not in response.text


def test_http_exception_uses_detail_not_error_key():
    """HTTPException (from the route's own handler) must not be intercepted by the
    global exception handler — it must still return {"detail": ...}."""
    client = _inject(ErrorEngine())
    data = client.post("/ask", json={"question": "test"}).json()
    assert "detail" in data
    assert "error" not in data
