import hashlib
import pytest
from src.agent.fetcher import Fetcher, FetchResult, Changed, Unchanged, FetchError


# ------------------------------------------------------------------ #
# Fake HTTP client
# ------------------------------------------------------------------ #

class FakeResponse:
    def __init__(self, *, text: str, status_code: int = 200, headers: dict = None):
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}


class FakeHttpClient:
    def __init__(self, responses: dict[str, FakeResponse]):
        self._responses = responses
        self.requests_made: list[dict] = []

    def get(self, url: str, headers: dict = None, timeout: int = 30) -> FakeResponse:
        self.requests_made.append({"url": url, "headers": headers or {}})
        if url not in self._responses:
            raise ConnectionError(f"No fake response for {url}")
        return self._responses[url]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestFetcherFirstFetch:
    """No previous hash — always Changed."""

    def test_new_content_returns_changed(self):
        client = FakeHttpClient({"https://example.com": FakeResponse(text="content v1")})
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch("https://example.com", previous_hash=None)
        assert isinstance(result, Changed)

    def test_changed_result_includes_content(self):
        client = FakeHttpClient({"https://example.com": FakeResponse(text="content v1")})
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch("https://example.com", previous_hash=None)
        assert isinstance(result, Changed)
        assert result.content == "content v1"

    def test_changed_result_includes_hash(self):
        client = FakeHttpClient({"https://example.com": FakeResponse(text="content v1")})
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch("https://example.com", previous_hash=None)
        assert isinstance(result, Changed)
        assert result.content_hash == _hash("content v1")


class TestFetcherContentHashDetection:
    """No ETag/Last-Modified — falls back to content hash comparison."""

    def test_same_content_returns_unchanged(self):
        content = "identical content"
        client = FakeHttpClient({"https://example.com": FakeResponse(text=content)})
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch("https://example.com", previous_hash=_hash(content))
        assert isinstance(result, Unchanged)

    def test_changed_content_returns_changed(self):
        client = FakeHttpClient({"https://example.com": FakeResponse(text="content v2")})
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch("https://example.com", previous_hash=_hash("content v1"))
        assert isinstance(result, Changed)
        assert result.content == "content v2"


class TestFetcherETagDetection:
    """Server returns ETag — use it to detect changes without full content hash."""

    def test_matching_etag_returns_unchanged(self):
        client = FakeHttpClient({
            "https://example.com": FakeResponse(
                text="content", headers={"ETag": '"abc123"'}
            )
        })
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch(
            "https://example.com",
            previous_hash=_hash("content"),
            previous_etag='"abc123"',
        )
        assert isinstance(result, Unchanged)

    def test_different_etag_returns_changed(self):
        client = FakeHttpClient({
            "https://example.com": FakeResponse(
                text="content v2", headers={"ETag": '"def456"'}
            )
        })
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch(
            "https://example.com",
            previous_hash=_hash("content v1"),
            previous_etag='"abc123"',
        )
        assert isinstance(result, Changed)

    def test_changed_result_includes_new_etag(self):
        client = FakeHttpClient({
            "https://example.com": FakeResponse(
                text="content v2", headers={"ETag": '"def456"'}
            )
        })
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch(
            "https://example.com",
            previous_hash=None,
            previous_etag=None,
        )
        assert isinstance(result, Changed)
        assert result.etag == '"def456"'

    def test_no_etag_in_response_falls_back_to_hash(self):
        content = "no etag content"
        client = FakeHttpClient({
            "https://example.com": FakeResponse(text=content, headers={})
        })
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch(
            "https://example.com",
            previous_hash=_hash(content),
            previous_etag='"old-etag"',
        )
        assert isinstance(result, Unchanged)


class TestFetcherErrorHandling:
    def test_http_error_status_returns_fetch_error(self):
        client = FakeHttpClient({
            "https://example.com": FakeResponse(text="", status_code=404)
        })
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch("https://example.com", previous_hash=None)
        assert isinstance(result, FetchError)
        assert "404" in result.reason

    def test_connection_error_returns_fetch_error(self):
        client = FakeHttpClient({})  # no response registered → raises ConnectionError
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch("https://example.com", previous_hash=None)
        assert isinstance(result, FetchError)

    def test_fetch_error_includes_url(self):
        client = FakeHttpClient({
            "https://example.com": FakeResponse(text="", status_code=500)
        })
        fetcher = Fetcher(http_client=client)
        result = fetcher.fetch("https://example.com", previous_hash=None)
        assert isinstance(result, FetchError)
        assert result.url == "https://example.com"
