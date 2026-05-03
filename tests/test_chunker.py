import pytest
from src.loader.document_loader import Document
from src.chunker.chunker import Chunker, Chunk


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

def test_chunk_has_required_fields():
    chunk = Chunk(
        text="some text",
        source="doc.md",
        doc_type="mock",
        section="Introduction",
        chunk_index=0,
        metadata={},
    )
    assert chunk.text == "some text"
    assert chunk.section == "Introduction"
    assert chunk.chunk_index == 0


# ---------------------------------------------------------------------------
# Markdown chunking (section-aware)
# ---------------------------------------------------------------------------

MD_WITH_SECTIONS = """# Overview

This is the overview section with some content.

## Requirements

You must comply with regulation X.
You must also comply with regulation Y.

## Exemptions

Some organisations are exempt.

### Small business exemption

Businesses with fewer than 10 employees are exempt.
"""


def _md_doc(content: str, source: str = "test.md") -> Document:
    return Document(content=content, source=source, doc_type="mock")


def test_markdown_produces_one_chunk_per_section():
    chunker = Chunker(max_tokens=512)
    doc = _md_doc(MD_WITH_SECTIONS)
    chunks = chunker.chunk(doc)
    sections = [c.section for c in chunks]
    assert "Overview" in sections
    assert "Requirements" in sections
    assert "Exemptions" in sections


def test_markdown_chunk_section_names_stripped():
    chunker = Chunker(max_tokens=512)
    doc = _md_doc(MD_WITH_SECTIONS)
    chunks = chunker.chunk(doc)
    for c in chunks:
        assert not c.section.startswith("#")


def test_markdown_chunk_text_contains_section_content():
    chunker = Chunker(max_tokens=512)
    doc = _md_doc(MD_WITH_SECTIONS)
    chunks = chunker.chunk(doc)
    req_chunk = next(c for c in chunks if c.section == "Requirements")
    assert "regulation X" in req_chunk.text


def test_markdown_chunk_source_preserved():
    chunker = Chunker(max_tokens=512)
    doc = _md_doc(MD_WITH_SECTIONS, source="esos_overview.md")
    chunks = chunker.chunk(doc)
    assert all(c.source == "esos_overview.md" for c in chunks)


def test_markdown_chunk_indices_sequential():
    chunker = Chunker(max_tokens=512)
    doc = _md_doc(MD_WITH_SECTIONS)
    chunks = chunker.chunk(doc)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_markdown_no_headings_returns_single_chunk():
    chunker = Chunker(max_tokens=512)
    doc = _md_doc("Just some plain text without any headings at all.")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert "plain text" in chunks[0].text


def test_markdown_empty_content_returns_no_chunks():
    chunker = Chunker(max_tokens=512)
    doc = _md_doc("")
    chunks = chunker.chunk(doc)
    assert chunks == []


# ---------------------------------------------------------------------------
# Long section splitting (token overflow)
# ---------------------------------------------------------------------------

def test_long_section_splits_into_multiple_chunks():
    # Each "word" is a token; 600 words should exceed max_tokens=200
    long_body = " ".join(["word"] * 600)
    content = f"# Big Section\n\n{long_body}"
    chunker = Chunker(max_tokens=200)
    doc = _md_doc(content)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert all(c.section == "Big Section" for c in chunks)


def test_long_section_chunks_have_overlap():
    long_body = " ".join([f"w{i}" for i in range(300)])
    content = f"# Section\n\n{long_body}"
    chunker = Chunker(max_tokens=100, overlap_tokens=20)
    doc = _md_doc(content)
    chunks = chunker.chunk(doc)
    # The end of chunk N should appear at the start of chunk N+1
    if len(chunks) > 1:
        end_words_of_first = chunks[0].text.split()[-20:]
        start_words_of_second = chunks[1].text.split()[:20]
        overlap = set(end_words_of_first) & set(start_words_of_second)
        assert len(overlap) > 0


# ---------------------------------------------------------------------------
# PDF chunking (page-aware fallback)
# ---------------------------------------------------------------------------

PDF_CONTENT = "Page 1 content here.\n\nPage 2 content here.\n\nPage 3 content here."


def _pdf_doc(content: str, source: str = "test.pdf") -> Document:
    return Document(content=content, source=source, doc_type="real")


def test_pdf_chunk_doc_type_is_real():
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_CONTENT)
    chunks = chunker.chunk(doc)
    assert all(c.doc_type == "real" for c in chunks)


def test_pdf_produces_chunks():
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_CONTENT)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1


def test_pdf_source_preserved():
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_CONTENT, source="secr_2019.pdf")
    chunks = chunker.chunk(doc)
    assert all(c.source == "secr_2019.pdf" for c in chunks)


def test_pdf_chunk_indices_sequential():
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_CONTENT)
    chunks = chunker.chunk(doc)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


# ---------------------------------------------------------------------------
# Metadata pass-through
# ---------------------------------------------------------------------------

def test_document_metadata_propagated_to_chunks():
    chunker = Chunker(max_tokens=512)
    doc = Document(
        content="# Section\n\nSome text.",
        source="test.md",
        doc_type="mock",
        metadata={"issuing_body": "Cabinet Office", "publication_date": "2022-01-01"},
    )
    chunks = chunker.chunk(doc)
    for c in chunks:
        assert c.metadata.get("issuing_body") == "Cabinet Office"
        assert c.metadata.get("publication_date") == "2022-01-01"
