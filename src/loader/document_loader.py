import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber


@dataclass
class Document:
    content: str
    source: str
    doc_type: str  # "real" | "mock"
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    def __init__(self, manifest_path: Path | None = None):
        self._context_map: dict[str, str] = {}
        if manifest_path is not None:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in entries:
                if "filename" in entry and "embed_context" in entry:
                    self._context_map[entry["filename"]] = entry["embed_context"]

    def load_markdown(self, path: Path) -> Document:
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")
        content = path.read_text(encoding="utf-8")
        return Document(content=content, source=path.name, doc_type="mock")

    def load_pdf(self, path: Path) -> Document:
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True)
                if not text or not text.strip():
                    text = page.extract_text()
                if text:
                    pages.append(text)
        content = "\n\n".join(pages)
        metadata: dict[str, Any] = {}
        if path.name in self._context_map:
            metadata["doc_context"] = self._context_map[path.name]
        return Document(content=content, source=path.name, doc_type="real", metadata=metadata)

    def load_directory(self, directory: Path, file_type: str) -> list[Document]:
        suffix = f".{file_type.lstrip('.')}"
        files = sorted(directory.glob(f"*{suffix}"))
        docs: list[Document] = []
        for f in files:
            if file_type == "pdf":
                docs.append(self.load_pdf(f))
            else:
                docs.append(self.load_markdown(f))
        return docs
