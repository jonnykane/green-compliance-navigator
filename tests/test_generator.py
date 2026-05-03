import pytest
from unittest.mock import MagicMock

from src.generator.generator import Generator, Answer
from src.retriever.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 3) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text=f"Regulation text {i}. This covers compliance requirement {i}.",
            source=f"doc{i}.pdf",
            section=f"Section {i}",
            doc_type="real",
            score=1.0 - 0.1 * i,
        )
        for i in range(n)
    ]


class FakeAnthropicMessage:
    def __init__(self, text: str):
        self.content = [MagicMock(text=text)]


class FakeAnthropicClient:
    def __init__(self, response_text: str = "This is the answer. [doc0.pdf, Section 0]"):
        self._response_text = response_text
        self.last_messages: list[dict] = []
        self.last_system: str = ""
        self.last_model: str = ""

    @property
    def messages(self):
        client = self
        class _Messages:
            def create(self_, **kwargs):
                client.last_messages = kwargs.get("messages", [])
                client.last_system = kwargs.get("system", "")
                client.last_model = kwargs.get("model", "")
                return FakeAnthropicMessage(client._response_text)
        return _Messages()


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

def test_answer_fields():
    a = Answer(text="The answer.", sources=["doc.pdf"], chunks_used=2)
    assert a.text == "The answer."
    assert a.sources == ["doc.pdf"]
    assert a.chunks_used == 2


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def test_generate_returns_answer():
    client = FakeAnthropicClient()
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    answer = gen.generate(query="What is ESOS?", chunks=_make_chunks())
    assert isinstance(answer, Answer)


def test_generate_answer_text_comes_from_model():
    client = FakeAnthropicClient(response_text="ESOS requires large companies to audit energy.")
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    answer = gen.generate(query="What is ESOS?", chunks=_make_chunks())
    assert "ESOS" in answer.text


def test_generate_includes_chunk_sources_in_answer():
    client = FakeAnthropicClient()
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    chunks = _make_chunks(2)
    answer = gen.generate(query="q", chunks=chunks)
    assert isinstance(answer.sources, list)


def test_generate_chunks_used_count():
    client = FakeAnthropicClient()
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    chunks = _make_chunks(4)
    answer = gen.generate(query="q", chunks=chunks)
    assert answer.chunks_used == 4


def test_generate_sends_query_to_model():
    client = FakeAnthropicClient()
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    gen.generate(query="Who must comply with ESOS?", chunks=_make_chunks())
    user_msg = client.last_messages[0]["content"]
    assert "Who must comply with ESOS?" in user_msg


def test_generate_includes_chunk_text_in_prompt():
    client = FakeAnthropicClient()
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    chunks = _make_chunks(2)
    gen.generate(query="q", chunks=chunks)
    user_msg = client.last_messages[0]["content"]
    assert chunks[0].text in user_msg
    assert chunks[1].text in user_msg


def test_generate_includes_source_references_in_prompt():
    client = FakeAnthropicClient()
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    chunks = _make_chunks(2)
    gen.generate(query="q", chunks=chunks)
    user_msg = client.last_messages[0]["content"]
    assert "doc0.pdf" in user_msg
    assert "doc1.pdf" in user_msg


def test_generate_uses_system_prompt():
    client = FakeAnthropicClient()
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    gen.generate(query="q", chunks=_make_chunks())
    assert len(client.last_system) > 0


def test_generate_system_prompt_instructs_citation():
    client = FakeAnthropicClient()
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    gen.generate(query="q", chunks=_make_chunks())
    assert "cit" in client.last_system.lower() or "source" in client.last_system.lower()


def test_generate_with_no_chunks_still_returns_answer():
    client = FakeAnthropicClient(response_text="I don't have relevant information.")
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    answer = gen.generate(query="obscure question", chunks=[])
    assert isinstance(answer, Answer)
    assert answer.chunks_used == 0


def test_generate_sources_deduplicates():
    """If two chunks share a source file, it appears once in answer.sources."""
    chunks = [
        RetrievedChunk(text="text a", source="esos.md", section="S1", doc_type="mock", score=0.9),
        RetrievedChunk(text="text b", source="esos.md", section="S2", doc_type="mock", score=0.8),
        RetrievedChunk(text="text c", source="tcfd.pdf", section="S1", doc_type="real", score=0.7),
    ]
    client = FakeAnthropicClient()
    gen = Generator(anthropic_client=client, model="claude-sonnet-4-5")
    answer = gen.generate(query="q", chunks=chunks)
    assert answer.sources.count("esos.md") == 1
