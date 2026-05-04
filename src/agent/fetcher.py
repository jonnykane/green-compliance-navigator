from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


# ------------------------------------------------------------------ #
# Result types
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class Changed:
    url: str
    content: str
    content_hash: str
    etag: Optional[str] = None


@dataclass(frozen=True)
class Unchanged:
    url: str


@dataclass(frozen=True)
class FetchError:
    url: str
    reason: str


FetchResult = Changed | Unchanged | FetchError


# ------------------------------------------------------------------ #
# HTTP client protocol (injectable)
# ------------------------------------------------------------------ #

@runtime_checkable
class HttpClient(Protocol):
    def get(self, url: str, headers: dict, timeout: int) -> object:
        ...


# ------------------------------------------------------------------ #
# Fetcher
# ------------------------------------------------------------------ #

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Fetcher:
    def __init__(self, http_client: HttpClient, timeout: int = 30) -> None:
        self._client = http_client
        self._timeout = timeout

    def fetch(
        self,
        url: str,
        *,
        previous_hash: Optional[str],
        previous_etag: Optional[str] = None,
    ) -> FetchResult:
        try:
            response = self._client.get(url, headers={}, timeout=self._timeout)
        except Exception as exc:
            return FetchError(url=url, reason=str(exc))

        if response.status_code != 200:
            return FetchError(url=url, reason=f"HTTP {response.status_code}")

        response_etag = response.headers.get("ETag")

        # ETag match — no change
        if response_etag and previous_etag and response_etag == previous_etag:
            return Unchanged(url=url)

        content = response.text
        content_hash = _sha256(content)

        # Content hash match — no change
        if previous_hash and content_hash == previous_hash:
            return Unchanged(url=url)

        return Changed(
            url=url,
            content=content,
            content_hash=content_hash,
            etag=response_etag,
        )
