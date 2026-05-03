import time
from dataclasses import dataclass
from typing import Any, Protocol


class VoyageClientProtocol(Protocol):
    def embed(self, texts: list[str], model: str, input_type: str) -> Any: ...


@dataclass
class EmbeddingResult:
    text: str
    vector: list[float]


_MAX_RETRIES = 6
_BACKOFF_BASE = 30.0  # seconds; doubles on each retry


class EmbeddingClient:
    def __init__(
        self,
        voyage_client: VoyageClientProtocol,
        model: str,
        batch_size: int = 128,
        request_interval: float = 0.0,
    ):
        self._client = voyage_client
        self._model = model
        self._batch_size = batch_size
        self._request_interval = request_interval

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        results: list[EmbeddingResult] = []
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        for idx, batch in enumerate(batches):
            if idx > 0 and self._request_interval > 0:
                time.sleep(self._request_interval)
            if len(batches) > 1:
                print(f"  Embedding batch {idx + 1}/{len(batches)} ({len(batch)} chunks)...")
            response = self._embed_with_backoff(batch, "document")
            for text, vector in zip(batch, response.embeddings):
                results.append(EmbeddingResult(text=text, vector=vector))
        return results

    def embed_query(self, query: str) -> list[float]:
        response = self._embed_with_backoff([query], "query")
        return response.embeddings[0]

    def _embed_with_backoff(self, texts: list[str], input_type: str) -> Any:
        delay = _BACKOFF_BASE
        for attempt in range(_MAX_RETRIES):
            try:
                return self._client.embed(texts, model=self._model, input_type=input_type)
            except Exception as exc:
                is_rate_limit = "rate" in str(exc).lower() or "429" in str(exc)
                if not is_rate_limit or attempt == _MAX_RETRIES - 1:
                    raise
                print(f"  Rate limit hit — waiting {delay:.0f}s before retry {attempt + 1}/{_MAX_RETRIES - 1}...")
                time.sleep(delay)
                delay = min(delay * 2, 300)
        raise RuntimeError("Unreachable")
