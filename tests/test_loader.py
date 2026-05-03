import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.loader.document_loader import DocumentLoader, Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_md(tmp_path):
    """A minimal markdown file on disk."""
    md = tmp_path / "test_doc.md"
    md.write_text("# Title\n\nSome content here.\n\n## Section 2\n\nMore content.")
    return md


@pytest.fixture
def tmp_pdf(tmp_path):
    """A minimal real-ish PDF stub — we mock pdfplumber, so content doesn't matter."""
    pdf = tmp_path / "test_doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------

def test_document_has_required_fields():
    doc = Document(
        content="hello",
        source="test.md",
        doc_type="mock",
        metadata={"title": "Test"},
    )
    assert doc.content == "hello"
    assert doc.source == "test.md"
    assert doc.doc_type == "mock"
    assert doc.metadata["title"] == "Test"


# ---------------------------------------------------------------------------
# Markdown loading
# ---------------------------------------------------------------------------

def test_load_markdown_returns_document(tmp_md):
    loader = DocumentLoader()
    doc = loader.load_markdown(tmp_md)
    assert isinstance(doc, Document)
    assert "Title" in doc.content
    assert doc.source == tmp_md.name
    assert doc.doc_type == "mock"


def test_load_markdown_preserves_full_text(tmp_md):
    loader = DocumentLoader()
    doc = loader.load_markdown(tmp_md)
    assert "Section 2" in doc.content
    assert "More content" in doc.content


def test_load_markdown_missing_file_raises():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_markdown(Path("/no/such/file.md"))


# ---------------------------------------------------------------------------
# PDF loading (pdfplumber mocked)
# ---------------------------------------------------------------------------

def _make_mock_pdf(pages_text: list[str]):
    """Return a context-manager mock that yields page mocks."""
    pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    pdf_obj = MagicMock()
    pdf_obj.pages = pages
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=pdf_obj)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_load_pdf_returns_document(tmp_pdf):
    mock_cm = _make_mock_pdf(["Page one text.", "Page two text."])
    with patch("src.loader.document_loader.pdfplumber.open", return_value=mock_cm):
        loader = DocumentLoader()
        doc = loader.load_pdf(tmp_pdf)
    assert isinstance(doc, Document)
    assert doc.doc_type == "real"
    assert doc.source == tmp_pdf.name


def test_load_pdf_concatenates_pages(tmp_pdf):
    mock_cm = _make_mock_pdf(["First page.", "Second page."])
    with patch("src.loader.document_loader.pdfplumber.open", return_value=mock_cm):
        loader = DocumentLoader()
        doc = loader.load_pdf(tmp_pdf)
    assert "First page." in doc.content
    assert "Second page." in doc.content


def test_load_pdf_skips_none_pages(tmp_pdf):
    mock_cm = _make_mock_pdf(["Real text.", None])
    with patch("src.loader.document_loader.pdfplumber.open", return_value=mock_cm):
        loader = DocumentLoader()
        doc = loader.load_pdf(tmp_pdf)
    assert "Real text." in doc.content


def test_load_pdf_missing_file_raises():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_pdf(Path("/no/such/file.pdf"))


# ---------------------------------------------------------------------------
# load_directory
# ---------------------------------------------------------------------------

def test_load_directory_loads_all_markdown(tmp_path):
    (tmp_path / "a.md").write_text("# Doc A\nContent A")
    (tmp_path / "b.md").write_text("# Doc B\nContent B")
    loader = DocumentLoader()
    docs = loader.load_directory(tmp_path, file_type="md")
    assert len(docs) == 2
    sources = {d.source for d in docs}
    assert "a.md" in sources
    assert "b.md" in sources


def test_load_directory_loads_all_pdfs(tmp_path):
    for name in ("x.pdf", "y.pdf"):
        (tmp_path / name).write_bytes(b"%PDF fake")
    mock_cm = _make_mock_pdf(["text"])
    with patch("src.loader.document_loader.pdfplumber.open", return_value=mock_cm):
        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path, file_type="pdf")
    assert len(docs) == 2


def test_load_directory_ignores_other_extensions(tmp_path):
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "doc.md").write_text("# Keep\nContent")
    loader = DocumentLoader()
    docs = loader.load_directory(tmp_path, file_type="md")
    assert len(docs) == 1


def test_load_directory_empty_dir_returns_empty(tmp_path):
    loader = DocumentLoader()
    docs = loader.load_directory(tmp_path, file_type="md")
    assert docs == []


# ---------------------------------------------------------------------------
# Manifest-based doc_context enrichment
# ---------------------------------------------------------------------------

def test_load_pdf_with_manifest_attaches_doc_context(tmp_path, tmp_pdf):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '[{"filename": "test_doc.pdf", "embed_context": "SECR sustainability reporting context"}]'
    )
    mock_cm = _make_mock_pdf(["Page content."])
    with patch("src.loader.document_loader.pdfplumber.open", return_value=mock_cm):
        loader = DocumentLoader(manifest_path=manifest)
        doc = loader.load_pdf(tmp_pdf)
    assert doc.metadata.get("doc_context") == "SECR sustainability reporting context"


def test_load_pdf_without_manifest_has_no_doc_context(tmp_pdf):
    mock_cm = _make_mock_pdf(["Page content."])
    with patch("src.loader.document_loader.pdfplumber.open", return_value=mock_cm):
        loader = DocumentLoader()
        doc = loader.load_pdf(tmp_pdf)
    assert "doc_context" not in doc.metadata


def test_load_pdf_filename_not_in_manifest_has_no_doc_context(tmp_path, tmp_pdf):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('[{"filename": "other_doc.pdf", "embed_context": "other context"}]')
    mock_cm = _make_mock_pdf(["Page content."])
    with patch("src.loader.document_loader.pdfplumber.open", return_value=mock_cm):
        loader = DocumentLoader(manifest_path=manifest)
        doc = loader.load_pdf(tmp_pdf)
    assert "doc_context" not in doc.metadata


def test_load_directory_propagates_doc_context(tmp_path):
    (tmp_path / "secr.pdf").write_bytes(b"%PDF fake")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('[{"filename": "secr.pdf", "embed_context": "SECR context"}]')
    mock_cm = _make_mock_pdf(["SECR content."])
    with patch("src.loader.document_loader.pdfplumber.open", return_value=mock_cm):
        loader = DocumentLoader(manifest_path=manifest)
        docs = loader.load_directory(tmp_path, file_type="pdf")
    assert len(docs) == 1
    assert docs[0].metadata.get("doc_context") == "SECR context"
