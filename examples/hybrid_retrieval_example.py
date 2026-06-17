#!/usr/bin/env python3
"""Hybrid retrieval demo: graph search + semantic document chunks.

Run:

    python examples/hybrid_retrieval_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drg.graph import EnhancedKG, KGEdge, KGNode
from drg.query import GraphQuery, InMemoryVectorStore, VectorDocumentChunk


class DemoEmbeddingProvider:
    """Tiny deterministic embedding provider for the demo."""

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        if "acquisition" in lowered or "microsoft" in lowered or "github" in lowered:
            return [1.0, 0.0, 0.0]
        if "sam" in lowered or "openai" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def build_kg() -> EnhancedKG:
    kg = EnhancedKG()
    for node in [
        KGNode(id="Sam Altman", type="Person", confidence=0.95),
        KGNode(id="OpenAI", type="Company", confidence=0.90),
        KGNode(id="Microsoft", type="Company", confidence=0.88),
        KGNode(id="GitHub", type="Company", confidence=0.86),
    ]:
        kg.add_node(node)

    for edge in [
        KGEdge(
            source="Sam Altman",
            target="OpenAI",
            relationship_type="WORKED_WITH",
            relationship_detail="Sam Altman worked with OpenAI.",
            metadata={"source_ref": "doc_sam"},
            confidence=0.92,
        ),
        KGEdge(
            source="Microsoft",
            target="OpenAI",
            relationship_type="INVESTED_IN",
            relationship_detail="Microsoft invested in OpenAI.",
            metadata={"source_ref": "doc_ms_openai"},
            confidence=0.87,
        ),
        KGEdge(
            source="Microsoft",
            target="GitHub",
            relationship_type="ACQUIRED",
            relationship_detail="Microsoft acquired GitHub.",
            metadata={"source_ref": "doc_github"},
            confidence=0.93,
        ),
    ]:
        kg.add_edge(edge)
    return kg


def main() -> int:
    embeddings = DemoEmbeddingProvider()
    chunks = [
        VectorDocumentChunk(
            chunk_id="doc_github_chunk_000",
            document_id="doc_github",
            text="Microsoft acquired GitHub and expanded its developer tools strategy.",
        ),
        VectorDocumentChunk(
            chunk_id="doc_sam_chunk_000",
            document_id="doc_sam",
            text="Sam Altman worked with OpenAI leadership on AI products.",
        ),
    ]
    store = InMemoryVectorStore.from_chunks(chunks, embedding_provider=embeddings)

    gq = GraphQuery(build_kg(), vector_store=store, embedding_provider=embeddings)
    results = gq.hybrid_search("What acquisitions involve Microsoft?", limit=5)

    for result in results:
        print(f"{result.entity.id}: score={result.score:.3f}")
        print(f"  why: {', '.join(result.explanation.why)}")
        print(f"  edges: {len(result.graph_edges)}")
        print(f"  chunks: {[h.chunk.chunk_id for h in result.document_chunks]}")
        print(f"  ranking: {result.explanation.ranking_components}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
