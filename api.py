"""
FastAPI server wrapping QueryEngine.ask().

Start with:
    uvicorn api:app --reload

Environment variables:
    PINECONE_API_KEY   — if set, QueryEngine uses Pinecone; otherwise ChromaDB
    ENVIRONMENT        — "development" enables localhost CORS origins (default: "production")
    (all other env vars as documented in src/config/settings.py)
"""
import logging
from dataclasses import asdict
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import settings
from src.models import QueryResult
from src.wiring import build_query_engine

logger = logging.getLogger(__name__)


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


def _get_allowed_origins() -> list[str]:
    """Return CORS allowed origins for the current environment.

    Production: Lovable app only.
    Development: also includes localhost dev-server ports.
    """
    origins = ["https://green-nav-guide.lovable.app"]
    if settings.ENVIRONMENT == "development":
        origins.extend([
            "http://localhost:3000",
            "http://localhost:5173",
        ])
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Global exception handler — catches anything that escapes route handlers.
# Logs the full traceback to Railway logs; returns a clean JSON body to the
# client so internal details are never exposed.
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred. Please try again."},
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
