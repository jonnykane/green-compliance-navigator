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
# PDF heading-aware chunking (fix for retrieval dilution)
# ---------------------------------------------------------------------------

# Realistic SECR-like content: numbered heading + threshold definition + noise
PDF_WITH_NUMBERED_HEADING = """\
Introductory text about the reporting framework.
This text continues on the same page as the section below.

2. Who needs to report under SECR?

The definition of large is based on sections 465 and 466 of the Companies Act.
The qualifying conditions are met by a company in a year in which it satisfies
two or more of the following requirements:
• Turnover £36 million or more
• Balance sheet total £18 million or more
• Number of employees 250 or more
"""

PDF_WITH_CHAPTER_HEADING = """\
Chapter 2: Guidance on Streamlined Energy and Carbon Reporting

This chapter explains SECR requirements and obligations for UK businesses.
"""

PDF_WITH_SHORT_TITLE_HEADING = """\
Background text about the scheme.

Group Reporting

If you are reporting at group level you must include all subsidiaries.
"""

PDF_NO_DETECTABLE_HEADINGS = """\
This paragraph contains regular body text with full sentences that end normally.
The text wraps across lines and continues in normal prose style.

Another paragraph here that also has no headings.
Just body content that should be merged by the paragraph strategy.
"""

# SECR-like fixture replicating the actual retrieval failure.
# Uses numbered heading directly above threshold to test the core regression.
# (In the real PDF "Quoted companies" is a sub-heading — tested separately via
# test_pdf_detects_short_title_heading; here we isolate the numbered-heading case.)
PDF_SECR_THRESHOLD_LIKE = """\
Some preamble about the 2018 Regulations and disclosure requirements.
Early identification will enable the necessary changes to be made in time.

2. Who needs to report under SECR?

Organisations must comply if they qualify as a large company under the
Companies Act 2006. The definition of large is based on sections 465 and 466.
The qualifying conditions are met by a company or LLP in a year in which
it satisfies two or more of the following requirements:
• Turnover £36 million or more
• Balance sheet total £18 million or more
• Number of employees 250 or more

Group Reporting
If reporting at group level you must include subsidiaries in the consolidation.
This applies both to quoted companies and large unquoted companies.
"""


def test_pdf_detects_numbered_section_heading():
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_WITH_NUMBERED_HEADING)
    chunks = chunker.chunk(doc)
    sections = [c.section for c in chunks]
    assert any("Who needs to report" in s for s in sections)


def test_pdf_detects_chapter_heading():
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_WITH_CHAPTER_HEADING)
    chunks = chunker.chunk(doc)
    sections = [c.section for c in chunks]
    assert any("Chapter 2" in s for s in sections)


def test_pdf_detects_short_title_heading():
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_WITH_SHORT_TITLE_HEADING)
    chunks = chunker.chunk(doc)
    sections = [c.section for c in chunks]
    assert "Group Reporting" in sections


def test_pdf_heading_prepended_to_chunk_text():
    """The section heading must appear in the chunk text so the embedding captures it."""
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_WITH_NUMBERED_HEADING)
    chunks = chunker.chunk(doc)
    threshold_chunk = next((c for c in chunks if "250 or more" in c.text), None)
    assert threshold_chunk is not None, "No chunk contains the threshold text"
    # Heading must be in the embedded text, not only in metadata
    assert "SECR" in threshold_chunk.text or "Who needs to report" in threshold_chunk.text


def test_pdf_threshold_chunk_contains_secr_context():
    """Core regression: threshold numbers must land in a chunk that also carries SECR context."""
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_SECR_THRESHOLD_LIKE)
    chunks = chunker.chunk(doc)
    threshold_chunk = next((c for c in chunks if "250 or more" in c.text), None)
    assert threshold_chunk is not None, "Threshold content not found in any chunk"
    combined = threshold_chunk.text + " " + threshold_chunk.section
    assert "SECR" in combined, (
        f"SECR context absent from threshold chunk.\n"
        f"section={threshold_chunk.section!r}\n"
        f"text[:200]={threshold_chunk.text[:200]!r}"
    )


def test_pdf_no_false_heading_detection_on_body_text():
    """Regular body text paragraphs must not be mistaken for headings."""
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_NO_DETECTABLE_HEADINGS)
    chunks = chunker.chunk(doc)
    assert all(c.section == "" for c in chunks)


def test_pdf_heading_section_stored_in_metadata_field():
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_WITH_NUMBERED_HEADING)
    chunks = chunker.chunk(doc)
    headed_chunks = [c for c in chunks if c.section != ""]
    assert len(headed_chunks) > 0


def test_pdf_footnote_numbers_not_detected_as_headings():
    """PDF footnotes like '22 The CRC scheme...' must not be mistaken for section headings."""
    content = (
        "2. Who needs to report under SECR?\n\n"
        "22 The CRC Energy Efficiency Scheme will be closed following compliance year.\n"
        "Footnote text continues here.\n"
        "• Turnover £36 million or more\n"
        "• Number of employees 250 or more"
    )
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(content)
    chunks = chunker.chunk(doc)
    sections = [c.section for c in chunks]
    # The footnote "22 The CRC..." must NOT create its own section
    assert not any(s.startswith("22 ") for s in sections)
    # The threshold must still be in the numbered section's chunk
    threshold_chunk = next((c for c in chunks if "250 or more" in c.text), None)
    assert threshold_chunk is not None
    assert "SECR" in threshold_chunk.text


def test_pdf_body_text_before_first_heading_still_chunked():
    """Content that precedes the first heading should still appear in a chunk."""
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_WITH_NUMBERED_HEADING)
    chunks = chunker.chunk(doc)
    full_text = " ".join(c.text for c in chunks)
    assert "Introductory text" in full_text


def test_pdf_long_section_under_heading_splits_correctly():
    long_body = " ".join([f"word{i}" for i in range(600)])
    content = f"2. Big Section\n\n{long_body}"
    chunker = Chunker(max_tokens=200)
    doc = _pdf_doc(content)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert all("Big Section" in c.text or "Big Section" in c.section for c in chunks)


def test_pdf_heading_aware_indices_sequential():
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(PDF_SECR_THRESHOLD_LIKE)
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


# ---------------------------------------------------------------------------
# Parent-section metadata (parent-child chunking fix)
# ---------------------------------------------------------------------------

def test_pdf_split_section_chunks_have_parent_section_text():
    """When a PDF section is too long and split, each sub-chunk must carry parent_section_text."""
    long_body = " ".join([f"w{i}" for i in range(600)])
    content = f"2. Big Section\n\n{long_body}"
    chunker = Chunker(max_tokens=200)
    doc = _pdf_doc(content)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1, "section must be split for this test to be meaningful"
    for c in chunks:
        assert "parent_section_text" in c.metadata, (
            f"chunk {c.chunk_index} is missing parent_section_text"
        )


def test_pdf_parent_section_text_contains_full_body():
    """parent_section_text must contain the full section body including all split words."""
    long_body = " ".join([f"w{i}" for i in range(300)])
    content = f"2. Big Section\n\nIntro sentence. {long_body}"
    chunker = Chunker(max_tokens=100)
    doc = _pdf_doc(content)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    parent = chunks[0].metadata["parent_section_text"]
    # Every sub-chunk's parent text should cover the full body
    assert "Intro sentence." in parent
    assert "w299" in parent  # last word of the body is present


def test_pdf_parent_section_text_contains_heading():
    """parent_section_text must include the section heading so tables have context."""
    long_body = " ".join([f"row{i}" for i in range(300)])
    content = f"2. Threshold Tables\n\n{long_body}"
    chunker = Chunker(max_tokens=100)
    doc = _pdf_doc(content)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert "Threshold Tables" in c.metadata["parent_section_text"]


def test_pdf_short_section_has_no_parent_section_text():
    """Sections that fit in one chunk must NOT get parent_section_text."""
    content = "2. Short Section\n\nThis section fits in one chunk easily."
    chunker = Chunker(max_tokens=512)
    doc = _pdf_doc(content)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert "parent_section_text" not in chunks[0].metadata
