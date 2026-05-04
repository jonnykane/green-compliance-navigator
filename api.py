"""
FastAPI server wrapping QueryEngine.ask().

Start with:
    uvicorn api:app --reload

Environment variables:
    PINECONE_API_KEY   — if set, QueryEngine uses Pinecone; otherwise ChromaDB
    (all other env vars as documented in src/config/settings.py)
"""
from dataclasses import asdict
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models import QueryResult
from src.wiring import build_query_engine


# ---------------------------------------------------------------------------
# Protocol — keeps api.py decoupled from the concrete QueryEngine class and
# allows any object with a compatible ask() to be injected (e.g. in tests).
# ---------------------------------------------------------------------------

class _QueryEngineProtocol(Protocol):
    def ask(self, question: str, company_context: str = "") -> QueryResult: ...


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="UK Green Compliance Navigator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency — lazily wired singleton; overridable in tests via
# app.dependency_overrides[get_engine] = lambda: fake_engine
# ---------------------------------------------------------------------------

_engine: _QueryEngineProtocol | None = None


def get_engine() -> _QueryEngineProtocol:
    global _engine
    if _engine is None:
        _engine = build_query_engine()
    return _engine


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    company_context: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(
    request: AskRequest,
    engine: Annotated[_QueryEngineProtocol, Depends(get_engine)],
) -> dict:
    try:
        result = engine.ask(request.question, request.company_context)
        return asdict(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
