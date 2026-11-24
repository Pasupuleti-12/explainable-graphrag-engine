from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from api.schemas import DocumentChunk


class DocumentProcessor:
    def __init__(self, chunk_size: int = 900, overlap: int = 140) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunks_from_text(self, text: str, source: str, metadata: dict | None = None) -> list[DocumentChunk]:
        clean = " ".join(text.split())
        metadata = metadata or {}
        if not clean:
            return []
        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        while start < len(clean):
            end = min(start + self.chunk_size, len(clean))
            part = clean[start:end]
            digest = hashlib.sha1(f"{source}:{index}:{part[:80]}".encode()).hexdigest()[:12]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source.replace(':', '_')}_{digest}",
                    text=part,
                    source=source,
                    metadata={**metadata, "chunk_index": index},
                )
            )
            if end == len(clean):
                break
            start = max(end - self.overlap, start + 1)
            index += 1
        return chunks

    def chunks_from_pdf(self, path: str | Path, source: str | None = None) -> list[DocumentChunk]:
        pdf_path = Path(path)
        document = fitz.open(pdf_path)
        texts = []
        for page in document:
            texts.append(page.get_text("text"))
        return self.chunks_from_text("\n".join(texts), source or pdf_path.name, {"source_type": "pdf"})
