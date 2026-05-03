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
                text = page.extract_text()
                if text:
                    pages.append(text)
        content = "\n\n".join(pages)
        return Document(content=content, source=path.name, doc_type="real")

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
