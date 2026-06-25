"""Tests for the query layer (``drg.query``).

All tests are deterministic — no LLM, no DSPy import required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drg.graph import EnhancedKG, GraphMerger, KGEdge, KGNode
from drg.graph.builders import build_enhanced_kg
from drg.query import GraphQuery, QueryError
from drg.reasoning import MultiDocumentReasoner, ReasoningConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _apple_beats_kg() -> EnhancedKG:
    """Merged KG: Apple/Beats/Jimmy + location facts + reasoning."""
    base = EnhancedKG()
    merger = GraphMerger()

    merger.merge(
        base,
        build_enhanced_kg(
            entities_typed=[("Apple", "Company"), ("Beats", "Company")],
            triples=[("Apple", "ACQUIRED", "Beats")],
            document_id="doc_A",
        ),
        document_id="doc_A",
    )
    merger.merge(
        base,
        build_enhanced_kg(
            entities_typed=[("Jimmy Iovine", "Person"), ("Beats", "Company")],
            triples=[("Jimmy Iovine", "FOUNDED", "Beats")],
            document_id="doc_B",
        ),
        document_id="doc_B",
    )
    merger.merge(
        base,
        build_enhanced_kg(
            entities_typed=[
                ("Beats", "Company"),
                ("Culver City", "Place"),
            ],
            triples=[("Beats", "HEADQUARTERED_IN", "Culver City")],
            document_id="doc_C",
        ),
        document_id="doc_C",
    )

    from drg.graph.kg_core import Cluster

    base.add_cluster(
        Cluster(
            id="tech_cluster",
            node_ids={"Apple", "Beats", "Jimmy Iovine"},
            metadata={"label": "tech"},
        )
    )

    # Run reasoning to add inferred edges (inverse, symmetric, path-bridge)
    MultiDocumentReasoner(config=ReasoningConfig(min_confidence=0.1)).reason(base)
    return base


def _simple_kg() -> EnhancedKG:
    kg = EnhancedKG()
    for nid, ntype in [
        ("Alice", "Person"),
        ("Bob", "Person"),
        ("Acme Inc", "Company"),
        ("Globex", "Company"),
        ("Paris", "Place"),
    ]:
        kg.add_node(KGNode(id=nid, type=ntype))
    for s, t, r in [
        ("Alice", "Acme Inc", "works_at"),
        ("Bob", "Globex", "works_at"),
        ("Alice", "Paris", "lives_in"),
        ("Bob", "Paris", "lives_in"),
    ]:
        kg.add_edge(
            KGEdge(
                source=s,
                target=t,
                relationship_type=r,
                relationship_detail=f"{s} {r} {t}",
                metadata={"source_ref": "doc_fixture"},
                confidence=0.9,
            )
        )
    return kg


@pytest.fixture
def apple_kg() -> EnhancedKG:
    return _apple_beats_kg()


@pytest.fixture
def simple_kg() -> EnhancedKG:
    return _simple_kg()


@pytest.fixture
def gq_apple(apple_kg: EnhancedKG) -> GraphQuery:
    return GraphQuery(apple_kg)


@pytest.fixture
def gq_simple(simple_kg: EnhancedKG) -> GraphQuery:
    return GraphQuery(simple_kg)


# ---------------------------------------------------------------------------
# Entity lookup
# ---------------------------------------------------------------------------


def test_entity_exact_lookup(gq_simple: GraphQuery):
    view = gq_simple.entity("Alice")
    assert view.id == "Alice"
    assert view.type == "Person"


def test_entity_not_found_raises(gq_simple: GraphQuery):
    with pytest.raises(QueryError, match="Entity not found"):
        gq_simple.entity("Nobody")


def test_find_entities_substring(gq_simple: GraphQuery):
    matches = gq_simple.find_entities("acme")
    assert matches
    assert matches[0].entity.id == "Acme Inc"
    assert matches[0].score > 0


def test_find_entities_type_filter(gq_simple: GraphQuery):
    matches = gq_simple.find_entities("a", entity_type="Company", limit=5)
    assert all(m.entity.type == "Company" for m in matches)


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


def test_relations_filter_by_source(gq_simple: GraphQuery):
    edges = gq_simple.relations(source="Alice")
    rel_types = {e.relationship_type for e in edges}
    assert "works_at" in rel_types
    assert "lives_in" in rel_types


def test_relations_filter_by_type(gq_simple: GraphQuery):
    edges = gq_simple.relations(relationship_type="works_at")
    assert len(edges) == 2
    assert all(e.relationship_type == "works_at" for e in edges)


def test_evidence_for_includes_source_ref(gq_simple: GraphQuery):
    bundle = gq_simple.evidence_for("Alice", "works_at", "Acme Inc")
    assert bundle.edges
    assert "doc_fixture" in bundle.source_documents
    assert bundle.evidence


# ---------------------------------------------------------------------------
# Neighborhood & traversal
# ---------------------------------------------------------------------------


def test_neighbors_one_hop(gq_simple: GraphQuery):
    hood = gq_simple.neighbors("Alice", hops=1)
    ids = {e.id for e in hood.entities}
    assert "Alice" in ids
    assert "Acme Inc" in ids or "Paris" in ids


def test_neighbors_two_hops(gq_simple: GraphQuery):
    hood = gq_simple.neighbors("Alice", hops=2)
    ids = {e.id for e in hood.entities}
    assert "Bob" in ids or "Globex" in ids


def test_find_paths_alice_to_bob(gq_simple: GraphQuery):
    paths = gq_simple.find_paths("Alice", "Bob", max_hops=3)
    assert paths
    assert paths[0].nodes[0] == "Alice"
    assert paths[0].nodes[-1] == "Bob"


def test_shortest_path(gq_simple: GraphQuery):
    path = gq_simple.shortest_path("Alice", "Bob", max_hops=3)
    assert path is not None
    assert path.hop_count >= 1


def test_find_paths_not_found_raises_for_missing_entity(gq_simple: GraphQuery):
    with pytest.raises(QueryError):
        gq_simple.find_paths("Alice", "Nobody", max_hops=2)


# ---------------------------------------------------------------------------
# Graph analytics
# ---------------------------------------------------------------------------


def test_degree_centrality_ranks_bridge_node(gq_simple: GraphQuery):
    scores = gq_simple.centrality(limit=2)
    assert scores[0].metric == "degree_centrality"
    assert scores[0].entity.id in {"Alice", "Bob", "Paris"}
    assert scores[0].score > 0
    assert scores[0].rank == 1


def test_pagerank_returns_ranked_scores(gq_simple: GraphQuery):
    scores = gq_simple.pagerank(limit=3, iterations=5)
    assert len(scores) == 3
    assert scores[0].score >= scores[-1].score
    assert all(score.metric == "pagerank" for score in scores)


def test_influence_scores_blend_graph_signals(gq_simple: GraphQuery):
    scores = gq_simple.influence_scores(limit=3)
    assert len(scores) == 3
    assert scores[0].metric == "influence_score"
    assert scores[0].details["degree"] >= 1


# ---------------------------------------------------------------------------
# Explain & reasoning integration
# ---------------------------------------------------------------------------


def test_explain_apple_jimmy_connected(gq_apple: GraphQuery):
    exp = gq_apple.explain("Apple", "Jimmy Iovine", max_hops=3)
    assert exp.connected
    assert exp.paths
    assert exp.evidence
    assert "Apple" in exp.summary
    assert "Jimmy Iovine" in exp.summary


def test_explain_includes_inferred_evidence(gq_apple: GraphQuery):
    exp = gq_apple.explain("Apple", "Jimmy Iovine", max_hops=3, include_inferred=True)
    assert any(item.is_inferred for item in exp.evidence) or any(
        e.is_inferred for p in exp.paths for e in p.edges
    )


def test_explain_not_connected_within_one_hop(gq_simple: GraphQuery):
    exp = gq_simple.explain("Alice", "Globex", max_hops=1)
    assert not exp.connected or exp.paths[0].hop_count > 1


# ---------------------------------------------------------------------------
# Related entities & community
# ---------------------------------------------------------------------------


def test_related_entities_shared_neighbors(gq_simple: GraphQuery):
    related = gq_simple.related_entities("Alice", mode="shared_neighbors")
    ids = {r.entity.id for r in related}
    assert "Bob" in ids


def test_related_entities_by_type_company(gq_apple: GraphQuery):
    related = gq_apple.related_entities(
        "Apple",
        mode="shortest_path",
        entity_type="Company",
        hops=3,
        limit=5,
    )
    assert all(r.entity.type == "Company" for r in related)


def test_community_of(gq_apple: GraphQuery):
    comm = gq_apple.community_of("Apple")
    assert comm is not None
    assert comm.cluster_id == "tech_cluster"
    assert "Beats" in comm.node_ids


def test_community_neighbors(gq_apple: GraphQuery):
    neighbors = gq_apple.community_neighbors("Apple")
    assert "Beats" in neighbors
    assert "Apple" not in neighbors


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_events_for_returns_empty_when_no_event_types(simple_kg: EnhancedKG):
    gq = GraphQuery(simple_kg)
    events = gq.events_for("Alice")
    assert events == []


def test_events_for_finds_event_nodes():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Apple", type="Company"))
    kg.add_node(KGNode(id="WWDC 2024", type="Event"))
    kg.add_edge(
        KGEdge(
            source="Apple",
            target="WWDC 2024",
            relationship_type="hosted",
            relationship_detail="Apple hosted WWDC 2024",
        )
    )
    gq = GraphQuery(kg)
    events = gq.events_for("Apple")
    assert len(events) == 1
    assert events[0].event.id == "WWDC 2024"


def test_events_for_normalizes_prefixed_event_node_types():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Apple", type="Company"))
    kg.add_node(KGNode(id="evt-1", type="Event:Acquisition"))
    kg.add_edge(
        KGEdge(
            source="evt-1",
            target="Apple",
            relationship_type="role:buyer",
            relationship_detail="Acquisition event has buyer Apple",
        )
    )

    gq = GraphQuery(kg)

    assert [ev.event.id for ev in gq.events_for("Apple")] == ["evt-1"]
    assert [ev.event.id for ev in gq.events_for("Apple", event_types=("Acquisition",))] == [
        "evt-1"
    ]


# ---------------------------------------------------------------------------
# Search & query alias
# ---------------------------------------------------------------------------


def test_search_returns_query_answer(gq_simple: GraphQuery):
    answer = gq_simple.search("Alice")
    assert "Alice" in answer.seed_entities
    assert answer.edges
    assert answer.provenance.evidence or answer.provenance.source_documents


def test_query_alias(gq_simple: GraphQuery):
    answer = gq_simple.query("Alice")
    assert answer.query == "Alice"


# ---------------------------------------------------------------------------
# EnhancedKG.query() convenience
# ---------------------------------------------------------------------------


def test_enhanced_kg_query_method(simple_kg: EnhancedKG):
    gq = simple_kg.query()
    assert isinstance(gq, GraphQuery)
    assert gq.entity("Alice").id == "Alice"


# ---------------------------------------------------------------------------
# Serialization & from_json
# ---------------------------------------------------------------------------


def test_result_to_dict_round_trip(gq_apple: GraphQuery):
    exp = gq_apple.explain("Apple", "Jimmy Iovine", max_hops=3)
    data = exp.to_dict()
    assert data["connected"] is True
    assert data["paths"]
    assert isinstance(data["evidence"], list)


def test_from_json(tmp_path: Path, apple_kg: EnhancedKG):
    path = tmp_path / "kg.json"
    apple_kg.save_json(str(path))
    gq = GraphQuery.from_json(path)
    assert gq.entity("Apple").id == "Apple"


def test_json_round_trip_preserves_query(tmp_path: Path, apple_kg: EnhancedKG):
    path = tmp_path / "kg.json"
    apple_kg.save_json(str(path))
    gq = GraphQuery.from_json(path)
    exp = gq.explain("Apple", "Jimmy Iovine", max_hops=3)
    assert exp.connected


# ---------------------------------------------------------------------------
# Determinism & inferred filtering
# ---------------------------------------------------------------------------


def test_find_paths_deterministic_order(gq_apple: GraphQuery):
    a = gq_apple.find_paths("Apple", "Jimmy Iovine", max_hops=3, max_paths=5)
    b = gq_apple.find_paths("Apple", "Jimmy Iovine", max_hops=3, max_paths=5)
    assert [p.nodes for p in a] == [p.nodes for p in b]


def test_exclude_inferred_edges(gq_apple: GraphQuery):
    extracted_only = gq_apple.relations(include_inferred=False)
    all_edges = gq_apple.relations(include_inferred=True)
    assert len(extracted_only) <= len(all_edges)
    assert not any(e.is_inferred for e in extracted_only)


# ---------------------------------------------------------------------------
# Backend indexing
# ---------------------------------------------------------------------------


def test_in_memory_backend_neighbors(simple_kg: EnhancedKG):
    from drg.query import InMemoryBackend

    backend = InMemoryBackend(simple_kg)
    neighbors = backend.neighbors("Alice")
    assert "Acme Inc" in neighbors
    assert "Paris" in neighbors


def test_top_level_lazy_import():
    import drg

    gq_cls = drg.GraphQuery
    assert gq_cls is not None
