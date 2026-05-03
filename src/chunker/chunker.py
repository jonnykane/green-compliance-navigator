import re
from dataclasses import dataclass, field
from typing import Any

from src.loader.document_loader import Document


@dataclass
class Chunk:
    text: str
    source: str
    doc_type: str
    section: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

_PDF_NUMBERED_RE = re.compile(r"^\d+(\.\d+)*\.\s+\w")
_PDF_STRUCTURAL_RE = re.compile(r"^(Chapter|Annex|Appendix|Section)\s+[\dA-Z]", re.IGNORECASE)

# Words that start body sentences, not headings
_BODY_STARTERS = frozenset([
    "the", "if", "this", "an", "under", "for", "in", "it", "a", "as",
    "where", "while", "when", "however", "although", "but", "and", "or",
    "you", "we", "they", "all", "any", "each", "some", "most",
    "note", "remember", "additionally", "furthermore", "therefore",
    "there", "these", "those", "such", "so", "that", "its", "with",
    "by", "at", "on", "from", "to", "of", "part", "based", "under",
    "companies", "organisations", "businesses",
])


def _is_pdf_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    # Bullet / list items are never headings
    if line[0] in ("•", "-", "*", "–", "·"):
        return False
    # Numbered section: "2. Title" or "2.1 Title"
    if _PDF_NUMBERED_RE.match(line):
        return True
    # Chapter / Annex / Appendix structural headings
    if _PDF_STRUCTURAL_RE.match(line):
        return True
    # Short title-case line (1–6 words, uppercase start, no trailing sentence punctuation)
    words = line.split()
    if not (1 <= len(words) <= 8):
        return False
    if not line[0].isupper():
        return False
    if line[-1] in (".", ",", ";", ":", ")", "?"):
        return False
    first_word = words[0].lower()
    if first_word in _BODY_STARTERS:
        return False
    return True


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _split_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_tokens
    return chunks


class Chunker:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.content.strip():
            return []
        if doc.doc_type == "real":
            return self._chunk_pdf(doc)
        return self._chunk_markdown(doc)

    # ------------------------------------------------------------------
    # Markdown chunking
    # ------------------------------------------------------------------

    def _chunk_markdown(self, doc: Document) -> list[Chunk]:
        sections = self._split_markdown_sections(doc.content)
        chunks: list[Chunk] = []
        for section_name, body in sections:
            body = body.strip()
            if not body:
                continue
            if _approx_tokens(body) <= self._max_tokens:
                chunks.append(self._make_chunk(body, section_name, len(chunks), doc))
            else:
                for part in _split_tokens(body, self._max_tokens, self._overlap_tokens):
                    chunks.append(self._make_chunk(part, section_name, len(chunks), doc))
        return chunks

    def _split_markdown_sections(self, content: str) -> list[tuple[str, str]]:
        matches = list(_HEADING_RE.finditer(content))
        if not matches:
            return [("", content)]
        sections: list[tuple[str, str]] = []
        preamble = content[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))
        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            sections.append((heading, content[body_start:body_end]))
        return sections

    # ------------------------------------------------------------------
    # PDF chunking (heading-aware)
    # ------------------------------------------------------------------

    def _chunk_pdf(self, doc: Document) -> list[Chunk]:
        sections = self._extract_pdf_sections(doc.content)
        chunks: list[Chunk] = []
        for heading, body in sections:
            body = body.strip()
            if not body:
                continue
            # Prepend heading to chunk text so embeddings carry section context
            chunk_text = f"{heading}\n\n{body}" if heading else body
            if _approx_tokens(chunk_text) <= self._max_tokens:
                chunks.append(self._make_chunk(chunk_text, heading, len(chunks), doc))
            else:
                # Long section: split the body, re-prepend heading to each piece
                heading_tokens = (_approx_tokens(heading) + 2) if heading else 0
                body_max = max(self._max_tokens - heading_tokens, 50)
                for part in _split_tokens(body, body_max, self._overlap_tokens):
                    text = f"{heading}\n\n{part}" if heading else part
                    chunks.append(self._make_chunk(text, heading, len(chunks), doc))
        return chunks

    def _extract_pdf_sections(self, text: str) -> list[tuple[str, str]]:
        """Scan line-by-line and group content under detected heading lines."""
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []
        current_heading: str = ""
        current_body: list[str] = []

        def flush() -> None:
            body = "\n".join(current_body).strip()
            if body:
                sections.append((current_heading, body))
            current_body.clear()

        for line in lines:
            if _is_pdf_heading(line):
                flush()
                current_heading = line.strip()
            else:
                current_body.append(line)

        flush()
        return sections

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------

    def _make_chunk(self, text: str, section: str, index: int, doc: Document) -> Chunk:
        return Chunk(
            text=text,
            source=doc.source,
            doc_type=doc.doc_type,
            section=section,
            chunk_index=index,
            metadata=dict(doc.metadata),
        )
