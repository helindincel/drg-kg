from __future__ import annotations

import pytest

from drg.graph import EnhancedKG, KGEdge, KGNode
from drg.query import GraphQuery, HybridRankingWeights, InMemoryVectorStore, VectorDocumentChunk


class _FakeEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        if "microsoft" in lowered or "acquisition" in lowered:
            return [1.0, 0.0, 0.0]
        if "sam" in lowered or "openai" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def hybrid_kg() -> EnhancedKG:
    kg = EnhancedKG()
    for node in [
        KGNode(id="Sam Altman", type="Person", confidence=0.95),
        KGNode(id="OpenAI", type="Company", confidence=0.90),
        KGNode(id="Microsoft", type="Company", confidence=0.88),
        KGNode(id="GitHub", type="Company", confidence=0.86),
        KGNode(id="GitHub Acquisition", type="Acquisition", confidence=0.82),
    ]:
        kg.add_node(node)

    for edge in [
        KGEdge(
            source="Sam Altman",
            target="OpenAI",
            relationship_type="WORKED_WITH",
            relationship_detail="Sam Altman worked with OpenAI as a leader.",
            metadata={"source_ref": "doc_sam", "evidence": "Sam Altman worked with OpenAI."},
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
            metadata={"source_ref": "doc_github", "evidence": "Microsoft acquired GitHub."},
            confidence=0.93,
        ),
        KGEdge(
            source="GitHub Acquisition",
            target="Microsoft",
            relationship_type="INVOLVES",
            relationship_detail="The GitHub acquisition involved Microsoft.",
            metadata={"source_ref": "doc_github_event"},
            confidence=0.84,
        ),
    ]:
        kg.add_edge(edge)
    return kg


def test_hybrid_search_graph_only_fallback(hybrid_kg: EnhancedKG):
    results = GraphQuery(hybrid_kg).hybrid_search("Who has worked with Sam Altman?")

    ids = [r.entity.id for r in results]
    assert "Sam Altman" in ids
    assert "OpenAI" in ids
    openai = next(r for r in results if r.entity.id == "OpenAI")
    assert openai.graph_edges
    assert "direct_graph_neighbor" in openai.explanation.why
    assert openai.explanation.ranking_components["vector_similarity"] == 0.0


def test_hybrid_search_uses_semantic_chunk_hits(hybrid_kg: EnhancedKG):
    store = InMemoryVectorStore.from_chunks(
        [
            VectorDocumentChunk(
                chunk_id="chunk_ms_github",
                text="Microsoft completed the GitHub acquisition and integrated developer tools.",
                document_id="doc_github",
                embedding=[1.0, 0.0, 0.0],
            ),
            VectorDocumentChunk(
                chunk_id="chunk_sam_openai",
                text="Sam Altman is connected to OpenAI leadership.",
                document_id="doc_sam",
                embedding=[0.0, 1.0, 0.0],
            ),
        ]
    )

    results = GraphQuery(
        hybrid_kg,
        vector_store=store,
        embedding_provider=_FakeEmbeddingProvider(),
    ).hybrid_search("What acquisitions involve Microsoft?", limit=5)

    microsoft = next(r for r in results if r.entity.id == "Microsoft")
    assert microsoft.document_chunks
    assert microsoft.document_chunks[0].chunk.chunk_id == "chunk_ms_github"
    assert "semantic_chunk_match" in microsoft.explanation.why
    assert microsoft.explanation.ranking_components["vector_similarity"] > 0.0


def test_hybrid_search_includes_events_and_evidence(hybrid_kg: EnhancedKG):
    results = GraphQuery(hybrid_kg).hybrid_search("Microsoft acquisitions", max_hops=1)

    microsoft = next(r for r in results if r.entity.id == "Microsoft")
    event_ids = {event.event.id for event in microsoft.events}
    assert "GitHub Acquisition" in event_ids
    assert microsoft.explanation.evidence
    assert any(item.source_ref == "doc_github" for item in microsoft.explanation.evidence)


def test_hybrid_ranking_weights_are_transparent(hybrid_kg: EnhancedKG):
    weights = HybridRankingWeights(
        graph_relevance=1.0,
        vector_similarity=0.0,
        confidence=0.0,
        evidence_quality=0.0,
        hop_closeness=0.0,
    )

    result = GraphQuery(hybrid_kg).hybrid_search(
        "Explain the relationship between Microsoft and OpenAI.",
        weights=weights,
        limit=1,
    )[0]

    components = result.explanation.ranking_components
    assert components["final"] == components["graph_relevance"]
    assert set(components) == {
        "graph_relevance",
        "vector_similarity",
        "confidence",
        "evidence_quality",
        "hop_closeness",
        "final",
    }
