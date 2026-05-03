import pytest
from unittest.mock import MagicMock, patch

from src.embedder.embedding_client import EmbeddingClient, EmbeddingResult


# ---------------------------------------------------------------------------
# Fake Voyage client used across tests
# ---------------------------------------------------------------------------

class FakeVoyageClient:
    """Deterministic fake that returns fixed-length vectors without network calls."""

    DIM = 8

    def embed(self, texts: list[str], model: str, input_type: str) -> MagicMock:
        result = MagicMock()
        result.embeddings = [[float(i) / 10] * self.DIM for i in range(len(texts))]
        return result


# ---------------------------------------------------------------------------
# EmbeddingResult
# ---------------------------------------------------------------------------

def test_embedding_result_stores_vector_and_text():
    er = EmbeddingResult(text="hello", vector=[0.1, 0.2, 0.3])
    assert er.text == "hello"
    assert er.vector == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# EmbeddingClient
# ---------------------------------------------------------------------------

def test_embed_texts_returns_correct_count():
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3")
    texts = ["text one", "text two", "text three"]
    results = client.embed_texts(texts)
    assert len(results) == 3


def test_embed_texts_returns_embedding_results():
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3")
    results = client.embed_texts(["hello"])
    assert isinstance(results[0], EmbeddingResult)
    assert results[0].text == "hello"
    assert len(results[0].vector) == FakeVoyageClient.DIM


def test_embed_texts_preserves_order():
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3")
    texts = ["alpha", "beta", "gamma"]
    results = client.embed_texts(texts)
    assert [r.text for r in results] == texts


def test_embed_texts_empty_list_returns_empty():
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3")
    assert client.embed_texts([]) == []


def test_embed_texts_batches_large_input():
    """Client should split into batches and still return all results."""
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3", batch_size=3)
    texts = [f"doc {i}" for i in range(10)]
    results = client.embed_texts(texts)
    assert len(results) == 10


def test_embed_texts_uses_document_input_type():
    """embed_texts must pass input_type='document' to the underlying client."""
    fake = FakeVoyageClient()
    spy = MagicMock(wraps=fake.embed)
    fake.embed = spy
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    client.embed_texts(["some text"])
    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs.get("input_type") == "document" or spy.call_args.args[2] == "document"


def test_embed_query_uses_query_input_type():
    """embed_query must pass input_type='query' to the underlying client."""
    fake = FakeVoyageClient()
    spy = MagicMock(wraps=fake.embed)
    fake.embed = spy
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    client.embed_query("what is ESOS?")
    spy.assert_called_once()
    call_args = spy.call_args
    passed_input_type = call_args.kwargs.get("input_type") or call_args.args[2]
    assert passed_input_type == "query"


def test_embed_query_returns_single_vector():
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3")
    vector = client.embed_query("ESOS threshold?")
    assert isinstance(vector, list)
    assert len(vector) == FakeVoyageClient.DIM


# ---------------------------------------------------------------------------
# Backoff retry behaviour (_embed_with_backoff)
# ---------------------------------------------------------------------------

class FailingVoyageClient:
    """Raises on first `fail_times` calls, then returns a real result."""

    DIM = 4

    def __init__(self, fail_times: int = 1, error_msg: str = "rate limit exceeded"):
        self._calls = 0
        self._fail_times = fail_times
        self._error_msg = error_msg

    def embed(self, texts: list[str], model: str, input_type: str) -> MagicMock:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise Exception(self._error_msg)
        result = MagicMock()
        result.embeddings = [[0.1] * self.DIM for _ in texts]
        return result

    @property
    def call_count(self) -> int:
        return self._calls


@patch("src.embedder.embedding_client.time.sleep")
def test_rate_limit_retries_until_success(mock_sleep):
    """A single rate-limit failure should be retried and ultimately succeed."""
    fake = FailingVoyageClient(fail_times=1, error_msg="rate limit exceeded")
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    results = client.embed_texts(["text"])
    assert len(results) == 1
    assert fake.call_count == 2  # 1 failure + 1 success


@patch("src.embedder.embedding_client.time.sleep")
def test_rate_limit_sleep_applied_between_retries(mock_sleep):
    """time.sleep must be called with the initial backoff delay on a rate-limit hit."""
    from src.embedder.embedding_client import _BACKOFF_BASE
    fake = FailingVoyageClient(fail_times=1, error_msg="429 Too Many Requests")
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    client.embed_texts(["text"])
    mock_sleep.assert_called_once_with(_BACKOFF_BASE)


@patch("src.embedder.embedding_client.time.sleep")
def test_rate_limit_delay_doubles_on_successive_retries(mock_sleep):
    """Backoff delay should double with each retry attempt."""
    from src.embedder.embedding_client import _BACKOFF_BASE
    fake = FailingVoyageClient(fail_times=2, error_msg="rate limit")
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    client.embed_texts(["text"])
    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert delays == [_BACKOFF_BASE, _BACKOFF_BASE * 2]


@patch("src.embedder.embedding_client.time.sleep")
def test_rate_limit_exhausted_reraises_after_max_retries(mock_sleep):
    """When all retries are consumed the original exception should propagate."""
    from src.embedder.embedding_client import _MAX_RETRIES
    fake = FailingVoyageClient(fail_times=_MAX_RETRIES, error_msg="rate limit exceeded")
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    with pytest.raises(Exception, match="rate limit exceeded"):
        client.embed_texts(["text"])


@patch("src.embedder.embedding_client.time.sleep")
def test_rate_limit_total_attempts_equals_max_retries(mock_sleep):
    """The underlying client should be called exactly _MAX_RETRIES times before giving up."""
    from src.embedder.embedding_client import _MAX_RETRIES
    fake = FailingVoyageClient(fail_times=_MAX_RETRIES, error_msg="rate limit")
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    with pytest.raises(Exception):
        client.embed_texts(["text"])
    assert fake.call_count == _MAX_RETRIES


@patch("src.embedder.embedding_client.time.sleep")
def test_non_rate_limit_exception_raised_immediately_without_retry(mock_sleep):
    """Non-rate-limit errors must propagate on the first attempt with no sleep."""
    fake = FailingVoyageClient(fail_times=1, error_msg="connection refused")
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    with pytest.raises(Exception, match="connection refused"):
        client.embed_texts(["text"])
    mock_sleep.assert_not_called()
    assert fake.call_count == 1  # no retries


@patch("src.embedder.embedding_client.time.sleep")
def test_rate_limit_detected_via_429_in_message(mock_sleep):
    """'429' anywhere in the exception message should trigger retry logic."""
    fake = FailingVoyageClient(fail_times=1, error_msg="HTTP 429")
    client = EmbeddingClient(voyage_client=fake, model="voyage-3")
    results = client.embed_texts(["text"])
    assert len(results) == 1
    mock_sleep.assert_called_once()


@patch("src.embedder.embedding_client.time.sleep")
def test_multi_batch_progress_line_executed(mock_sleep, capsys):
    """Progress output must be printed when there is more than one batch."""
    client = EmbeddingClient(voyage_client=FakeVoyageClient(), model="voyage-3", batch_size=1)
    client.embed_texts(["first", "second"])
    captured = capsys.readouterr()
    assert "batch 1/2" in captured.out
    assert "batch 2/2" in captured.out


@patch("src.embedder.embedding_client.time.sleep")
def test_request_interval_sleep_applied_between_batches(mock_sleep):
    """When request_interval > 0, time.sleep must be called between batches (not before first)."""
    client = EmbeddingClient(
        voyage_client=FakeVoyageClient(),
        model="voyage-3",
        batch_size=1,
        request_interval=0.5,
    )
    client.embed_texts(["first", "second", "third"])
    # 3 texts → 3 batches → sleep called between batch 0→1 and 1→2 (never before batch 0)
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(0.5)
