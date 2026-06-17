"""Storage-agnostic vector retrieval for query-time document chunks."""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from ._types import VectorChunkHit, VectorDocumentChunk

__all__ = [
    "InMemoryVectorStore",
    "VectorStore",
    "cosine_similarity",
    "document_chunk_from_mapping",
]


@runtime_checkable
class VectorStore(Protocol):
    """Minimal protocol for semantic chunk retrieval.

    Chroma, Qdrant, Weaviate, Neo4j Vector, or any project-specific store can
    implement this without changing :class:`drg.query.GraphQuery`.
    """

    def search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
    ) -> list[VectorChunkHit]:
        """Return semantically similar chunks for ``query_embedding``."""
        ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity normalized to ``[0, 1]`` for ranking."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, (dot / (na * nb) + 1.0) / 2.0))


def document_chunk_from_mapping(data: dict[str, Any]) -> VectorDocumentChunk:
    """Create a vector document chunk from common chunk dictionary shapes."""
    chunk_id = str(data.get("chunk_id") or data.get("id") or "")
    if not chunk_id:
        raise ValueError("Document chunk requires chunk_id or id")
    text = str(data.get("text") or data.get("chunk_text") or "")
    embedding = data.get("embedding")
    if embedding is not None:
        embedding = [float(x) for x in embedding]

    document_id = data.get("document_id") or data.get("source_ref") or data.get("origin_file")
    metadata = {
        k: v
        for k, v in data.items()
        if k not in {"id", "chunk_id", "text", "chunk_text", "embedding", "document_id"}
    }
    return VectorDocumentChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=str(document_id) if document_id is not None else None,
        metadata=metadata,
        embedding=embedding,
    )


class InMemoryVectorStore:
    """Pure-stdlib vector store for tests, demos, and small projects."""

    def __init__(self, chunks: list[VectorDocumentChunk] | None = None) -> None:
        self._chunks: list[VectorDocumentChunk] = []
        if chunks:
            self.add_chunks(chunks)

    @classmethod
    def from_chunks(
        cls,
        chunks: list[VectorDocumentChunk | dict[str, Any]],
        *,
        embedding_provider: Any | None = None,
    ) -> "InMemoryVectorStore":
        """Build a store from chunks, embedding missing vectors if possible."""
        normalized = [
            c if isinstance(c, VectorDocumentChunk) else document_chunk_from_mapping(c)
            for c in chunks
        ]
        missing = [c for c in normalized if c.embedding is None]
        if missing and embedding_provider is not None:
            embeddings = embedding_provider.embed_batch([c.text for c in missing])
            by_id = {c.chunk_id: emb for c, emb in zip(missing, embeddings, strict=False)}
            normalized = [
                VectorDocumentChunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    document_id=c.document_id,
                    metadata=dict(c.metadata),
                    embedding=by_id.get(c.chunk_id, c.embedding),
                )
                for c in normalized
            ]
        return cls(normalized)

    def add_chunks(self, chunks: list[VectorDocumentChunk]) -> None:
        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
    ) -> list[VectorChunkHit]:
        hits: list[VectorChunkHit] = []
        for chunk in self._chunks:
            if chunk.embedding is None:
                continue
            score = cosine_similarity(query_embedding, chunk.embedding)
            if score > 0.0:
                hits.append(VectorChunkHit(chunk=chunk, score=score))
        hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
        return hits[: max(0, limit)]
