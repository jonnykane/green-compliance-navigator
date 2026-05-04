from __future__ import annotations

import httpx
import json
import os
from datetime import datetime, timezone

import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agent.claude_classifier_client import AnthropicClassifierClient
from src.agent.eval_flagger import EvalFlagger
from src.agent.fetcher import Fetcher
from src.agent.materiality import MaterialityClassifier
from src.agent.react_loop import MonitoringAgent
from src.agent.rechunker import Rechunker, ChunkingConfig
from src.agent.registry import FrameworkRegistry, RegistryLoadError
from src.agent.summariser import Summariser


router = APIRouter(prefix="/agent", tags=["agent"])


class RunResponse(BaseModel):
    started_at: datetime
    finished_at: datetime
    total_sources: int
    updated_count: int
    error_count: int
    summary: str
    traces: list[dict]


@router.post("/run", response_model=RunResponse)
def run_agent():
    try:
        registry = FrameworkRegistry.from_env()
    except RegistryLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    class HttpxClient:
        def get(self, url, headers, timeout):
            return httpx.get(url, headers=headers, timeout=timeout)

    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    from src.embedder.embedding_client import EmbeddingClient
    import voyageai
    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    embedding_client = EmbeddingClient(client=voyage_client)

    class EmbedderAdapter:
        def embed(self, texts: list[str]) -> list[list[float]]:
            results = embedding_client.embed_texts(texts)
            return [r.vector for r in results]

    raw_evals = json.loads(os.environ.get("GOLDEN_EVALS_JSON", "[]"))
    flagger = EvalFlagger.from_dicts(raw_evals)

    agent = MonitoringAgent(
        registry=registry,
        fetcher=Fetcher(http_client=HttpxClient()),
        classifier=MaterialityClassifier(
            claude_client=AnthropicClassifierClient(client=anthropic_client)
        ),
        rechunker=Rechunker(
            embedder=EmbedderAdapter(),
            index=index,
            config=ChunkingConfig(),
        ),
        flagger=flagger,
    )

    report = agent.run()
    summary = Summariser().summarise(report)

    return RunResponse(
        started_at=report.started_at,
        finished_at=report.finished_at,
        total_sources=report.total_sources,
        updated_count=report.updated_count,
        error_count=report.error_count,
        summary=summary,
        traces=[
            {
                "framework_id": t.framework_id,
                "outcome": t.outcome.value,
                "thought": t.thought,
                "action": t.action,
                "observation": t.observation,
                "flagged_eval_ids": t.flagged_eval_ids,
            }
            for t in report.traces
        ],
    )
