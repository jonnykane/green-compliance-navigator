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


def _approx_tokens(text: str) -> int:
    """Rough token count: split on whitespace."""
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
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.content.strip():
            return []
        if doc.doc_type == "real":
            return self._chunk_pdf(doc)
        return self._chunk_markdown(doc)

    def _chunk_markdown(self, doc: Document) -> list[Chunk]:
        sections = self._split_into_sections(doc.content)
        chunks: list[Chunk] = []
        for section_name, body in sections:
            body = body.strip()
            if not body:
                continue
            if _approx_tokens(body) <= self.max_tokens:
                chunks.append(self._make_chunk(body, section_name, len(chunks), doc))
            else:
                for part in _split_tokens(body, self.max_tokens, self.overlap_tokens):
                    chunks.append(self._make_chunk(part, section_name, len(chunks), doc))
        return chunks

    def _chunk_pdf(self, doc: Document) -> list[Chunk]:
        # PDFs don't have reliable heading structure; use paragraph breaks as
        # natural boundaries and then apply token splitting.
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", doc.content) if p.strip()]
        current_parts: list[str] = []
        current_tokens = 0
        chunks: list[Chunk] = []

        def flush():
            if current_parts:
                text = "\n\n".join(current_parts)
                chunks.append(self._make_chunk(text, "", len(chunks), doc))
                current_parts.clear()

        for para in paragraphs:
            para_tokens = _approx_tokens(para)
            if para_tokens > self.max_tokens:
                flush()
                for part in _split_tokens(para, self.max_tokens, self.overlap_tokens):
                    chunks.append(self._make_chunk(part, "", len(chunks), doc))
            elif current_tokens + para_tokens > self.max_tokens:
                flush()
                current_parts.append(para)
                current_tokens = para_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        flush()
        return chunks

    def _split_into_sections(self, content: str) -> list[tuple[str, str]]:
        matches = list(_HEADING_RE.finditer(content))
        if not matches:
            return [("", content)]

        sections: list[tuple[str, str]] = []
        # Text before first heading
        preamble = content[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[body_start:body_end]
            sections.append((heading, body))

        return sections

    def _make_chunk(self, text: str, section: str, index: int, doc: Document) -> Chunk:
        return Chunk(
            text=text,
            source=doc.source,
            doc_type=doc.doc_type,
            section=section,
            chunk_index=index,
            metadata=dict(doc.metadata),
        )
