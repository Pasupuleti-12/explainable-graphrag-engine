from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils import embedding_functions

from api.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(self, persist_dir: str = "data/chroma", collection_name: str = "healthcare_chunks") -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunk_list],
            documents=[chunk.text for chunk in chunk_list],
            metadatas=[{**chunk.metadata, "source": chunk.source, "chunk_id": chunk.chunk_id} for chunk in chunk_list],
        )
        logger.info("Stored %s chunks in ChromaDB", len(chunk_list))
        return len(chunk_list)

    def query(self, query: str, n_results: int = 8) -> list[DocumentChunk]:
        result = self.collection.query(query_texts=[query], n_results=n_results)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        chunks: list[DocumentChunk] = []
        for text, metadata in zip(documents, metadatas):
            metadata = metadata or {}
            chunks.append(
                DocumentChunk(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    text=text,
                    source=str(metadata.get("source", "unknown")),
                    metadata=dict(metadata),
                )
            )
        return chunks

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            logger.info("Chroma collection did not exist during clear")
        self.collection = self._get_collection()
