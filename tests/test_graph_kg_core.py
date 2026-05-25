"""Unit tests for :mod:`drg.graph.kg_core`.

These tests cover the public surface of the most-touched module in the
project (``EnhancedKG`` plus the ``KGNode`` / ``KGEdge`` / ``Cluster``
dataclasses). They run without any external dependencies — no LLM, no
network, no optional packages.

Coverage targets:
    - Dataclass validation in ``__post_init__`` (every guard clause)
    - ``to_dict`` / ``from_dict`` round-trip integrity for every dataclass
    - ``KGEdge.from_enriched_relationship`` adapter, including temporal
      metadata pulled from nested and flat shapes
    - ``EnhancedKG.add_node`` / ``add_edge`` / ``add_cluster`` happy paths
      *and* their referential integrity guards (orphan edge / orphan
      cluster member rejections)
    - All three exporters (``to_json`` / ``to_json_ld`` /
      ``to_enriched_format``) with full payload, partial payload, and
      empty-graph variants
    - File-writing wrappers (``save_json`` / ``save_json_ld`` /
      ``save_enriched_format``) using ``tmp_path``
    - ``EnhancedKG.from_enriched_relationships`` factory
    - ``EnhancedKG.add_entity_embeddings`` with a stub provider, including
      the default-identity-mapping path and the explicit-mapping path
"""

from __future__ import annotations

import json

import pytest

from drg.graph.kg_core import Cluster, EnhancedKG, KGEdge, KGNode
from drg.graph.relationship_model import EnrichedRelationship, RelationshipType

# ---------------------------------------------------------------------------
# KGNode
# ---------------------------------------------------------------------------


def test_kgnode_minimal_construction_is_valid():
    node = KGNode(id="alice")
    assert node.id == "alice"
    assert node.type is None
    assert node.properties == {}
    assert node.metadata == {}
    assert node.embedding is None


def test_kgnode_full_construction_preserves_all_fields():
    node = KGNode(
        id="alice",
        type="Person",
        properties={"age": 30},
        metadata={"source_ref": "doc_1"},
        embedding=[0.1, 0.2, 0.3],
    )
    assert node.type == "Person"
    assert node.properties == {"age": 30}
    assert node.metadata == {"source_ref": "doc_1"}
    assert node.embedding == [0.1, 0.2, 0.3]


def test_kgnode_rejects_empty_id():
    with pytest.raises(ValueError, match="Node id cannot be empty"):
        KGNode(id="")


def test_kgnode_to_dict_includes_only_populated_optionals():
    bare = KGNode(id="alice").to_dict()
    assert bare == {"id": "alice", "type": None}
    assert "properties" not in bare
    assert "metadata" not in bare
    assert "embedding" not in bare

    rich = KGNode(
        id="alice",
        type="Person",
        properties={"age": 30},
        metadata={"src": "doc"},
        embedding=[0.1],
    ).to_dict()
    assert rich["properties"] == {"age": 30}
    assert rich["metadata"] == {"src": "doc"}
    assert rich["embedding"] == [0.1]


def test_kgnode_round_trip_through_dict_preserves_equality():
    original = KGNode(
        id="alice",
        type="Person",
        properties={"age": 30},
        metadata={"src": "doc"},
        embedding=[0.1, 0.2],
    )
    restored = KGNode.from_dict(original.to_dict())
    assert restored == original


def test_kgnode_from_dict_handles_missing_optionals():
    node = KGNode.from_dict({"id": "alice"})
    assert node.id == "alice"
    assert node.type is None
    assert node.properties == {}
    assert node.metadata == {}
    assert node.embedding is None


# ---------------------------------------------------------------------------
# KGEdge — validation
# ---------------------------------------------------------------------------


def _edge(**overrides) -> KGEdge:
    """Helper to build a minimally-valid KGEdge with overrides."""
    defaults = {
        "source": "alice",
        "target": "bob",
        "relationship_type": "KNOWS",
        "relationship_detail": "alice knows bob",
        "metadata": {},
    }
    defaults.update(overrides)
    return KGEdge(**defaults)


def test_kgedge_minimal_construction_is_valid():
    edge = _edge()
    assert edge.source == "alice"
    assert edge.target == "bob"
    assert edge.confidence is None
    assert edge.is_negated is False


def test_kgedge_rejects_empty_source():
    with pytest.raises(ValueError, match="source and target cannot be empty"):
        _edge(source="")


def test_kgedge_rejects_empty_target():
    with pytest.raises(ValueError, match="source and target cannot be empty"):
        _edge(target="")


def test_kgedge_rejects_empty_relationship_type():
    with pytest.raises(ValueError, match="relationship_type and detail cannot be empty"):
        _edge(relationship_type="")


def test_kgedge_rejects_empty_relationship_detail():
    with pytest.raises(ValueError, match="relationship_type and detail cannot be empty"):
        _edge(relationship_detail="")


def test_kgedge_rejects_self_loop():
    with pytest.raises(ValueError, match="source and target cannot be the same"):
        _edge(target="alice")


@pytest.mark.parametrize("bad", [-0.01, 1.01, -1.0, 2.0])
def test_kgedge_rejects_out_of_range_confidence(bad: float):
    with pytest.raises(ValueError, match=r"Confidence score must be between 0\.0 and 1\.0"):
        _edge(confidence=bad)


@pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
def test_kgedge_accepts_boundary_confidence(ok: float):
    edge = _edge(confidence=ok)
    assert edge.confidence == ok


# ---------------------------------------------------------------------------
# KGEdge — serialisation
# ---------------------------------------------------------------------------


def test_kgedge_to_dict_includes_only_populated_optionals():
    bare = _edge().to_dict()
    assert bare["source"] == "alice"
    assert bare["target"] == "bob"
    assert bare["metadata"] == {}
    for optional in ("start_time", "end_time", "confidence", "is_negated"):
        assert optional not in bare


def test_kgedge_to_dict_emits_all_optionals_when_set():
    edge = _edge(
        start_time="2024-01-01",
        end_time="2024-12-31",
        confidence=0.8,
        is_negated=True,
    )
    d = edge.to_dict()
    assert d["start_time"] == "2024-01-01"
    assert d["end_time"] == "2024-12-31"
    assert d["confidence"] == 0.8
    assert d["is_negated"] is True


def test_kgedge_round_trip_preserves_all_fields():
    original = _edge(
        start_time="2024-01-01",
        end_time="2024-12-31",
        confidence=0.8,
        is_negated=True,
        metadata={"source_ref": "chunk_1"},
    )
    restored = KGEdge.from_dict(original.to_dict())
    assert restored == original


def test_kgedge_from_dict_handles_missing_optionals():
    edge = KGEdge.from_dict(
        {
            "source": "alice",
            "target": "bob",
            "relationship_type": "KNOWS",
            "relationship_detail": "...",
        }
    )
    assert edge.metadata == {}
    assert edge.is_negated is False
    assert edge.confidence is None


# ---------------------------------------------------------------------------
# KGEdge.from_enriched_relationship
# ---------------------------------------------------------------------------


def _enriched(**overrides) -> EnrichedRelationship:
    defaults = {
        "source": "alice",
        "target": "bob",
        "relationship_type": RelationshipType.CAUSES,
        "relationship_detail": "alice caused bob",
        "confidence": 0.9,
        "source_ref": "chunk_1",
    }
    defaults.update(overrides)
    return EnrichedRelationship(**defaults)


def test_from_enriched_relationship_carries_core_fields():
    rel = _enriched()
    edge = KGEdge.from_enriched_relationship(rel)
    assert edge.source == "alice"
    assert edge.target == "bob"
    assert edge.relationship_type == "causes"  # Enum value
    assert edge.relationship_detail == "alice caused bob"
    assert edge.confidence == 0.9
    # Confidence + source_ref bubble up into metadata for back-compat callers.
    assert edge.metadata["confidence"] == 0.9
    assert edge.metadata["source_ref"] == "chunk_1"


def test_from_enriched_relationship_omits_source_ref_when_none():
    rel = _enriched(source_ref=None)
    edge = KGEdge.from_enriched_relationship(rel)
    assert "source_ref" not in edge.metadata


def test_from_enriched_relationship_preserves_is_negated_when_present():
    # EnrichedRelationship doesn't define is_negated, so we test the
    # graceful getattr() fallback by attaching the attribute manually.
    rel = _enriched()
    rel.is_negated = True
    edge = KGEdge.from_enriched_relationship(rel)
    assert edge.is_negated is True


def test_from_enriched_relationship_temporal_nested_shape():
    rel = _enriched()
    # The adapter looks at `rel.metadata["temporal"]` first.
    rel.metadata = {"temporal": {"start": "2024-01-01", "end": "2024-12-31"}}
    edge = KGEdge.from_enriched_relationship(rel)
    assert edge.start_time == "2024-01-01"
    assert edge.end_time == "2024-12-31"


def test_from_enriched_relationship_temporal_flat_shape_backcompat():
    # Older shape: start_time / end_time at the top of metadata.
    rel = _enriched()
    rel.metadata = {"start_time": "2024-01-01", "end_time": "2024-12-31"}
    edge = KGEdge.from_enriched_relationship(rel)
    assert edge.start_time == "2024-01-01"
    assert edge.end_time == "2024-12-31"


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


def test_cluster_minimal_construction_is_valid():
    c = Cluster(id="c1", node_ids={"a", "b"})
    assert c.id == "c1"
    assert c.node_ids == {"a", "b"}
    assert c.metadata == {}


def test_cluster_rejects_empty_id():
    with pytest.raises(ValueError, match="Cluster id and node_ids cannot be empty"):
        Cluster(id="", node_ids={"a"})


def test_cluster_rejects_empty_node_ids():
    with pytest.raises(ValueError, match="Cluster id and node_ids cannot be empty"):
        Cluster(id="c1", node_ids=set())


def test_cluster_round_trip_preserves_equality():
    original = Cluster(id="c1", node_ids={"a", "b"}, metadata={"algo": "louvain"})
    restored = Cluster.from_dict(original.to_dict())
    assert restored == original


def test_cluster_to_dict_lists_node_ids_for_json_compatibility():
    d = Cluster(id="c1", node_ids={"a", "b"}).to_dict()
    # Sets aren't JSON-serialisable; the dict form must give us a list.
    assert isinstance(d["node_ids"], list)
    assert set(d["node_ids"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# EnhancedKG — node + edge + cluster registration
# ---------------------------------------------------------------------------


def test_enhanced_kg_empty_is_well_formed():
    kg = EnhancedKG()
    assert kg.nodes == {}
    assert kg.edges == []
    assert kg.clusters == {}


def test_enhanced_kg_add_node_and_lookup():
    kg = EnhancedKG()
    n = KGNode(id="alice", type="Person")
    kg.add_node(n)
    assert kg.get_node("alice") is n
    assert kg.get_node("missing") is None


def test_enhanced_kg_add_node_idempotent_on_same_id_overwrites():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice", type="Person"))
    kg.add_node(KGNode(id="alice", type="Customer"))  # same id, different type
    assert kg.get_node("alice").type == "Customer"
    assert len(kg.nodes) == 1


def test_enhanced_kg_add_edge_requires_endpoints_registered():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    edge = _edge()  # target=bob isn't registered
    with pytest.raises(ValueError, match="Source and target nodes must exist"):
        kg.add_edge(edge)


def test_enhanced_kg_add_edge_happy_path():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    kg.add_node(KGNode(id="bob"))
    edge = _edge()
    kg.add_edge(edge)
    assert kg.edges == [edge]


def test_enhanced_kg_add_cluster_rejects_orphan_members():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    with pytest.raises(ValueError, match="non-existent nodes"):
        kg.add_cluster(Cluster(id="c1", node_ids={"alice", "ghost"}))


def test_enhanced_kg_add_cluster_happy_path():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    kg.add_node(KGNode(id="bob"))
    cluster = Cluster(id="c1", node_ids={"alice", "bob"}, metadata={"algo": "louvain"})
    kg.add_cluster(cluster)
    assert kg.clusters == {"c1": cluster}


# ---------------------------------------------------------------------------
# EnhancedKG — exporters
# ---------------------------------------------------------------------------


def _populated_kg() -> EnhancedKG:
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice", type="Person", properties={"age": 30}))
    kg.add_node(KGNode(id="bob", type="Person"))
    kg.add_edge(
        KGEdge(
            source="alice",
            target="bob",
            relationship_type="KNOWS",
            relationship_detail="alice knows bob",
            metadata={"source_ref": "chunk_1"},
            confidence=0.9,
            start_time="2024-01-01",
            is_negated=False,
        )
    )
    kg.add_cluster(Cluster(id="c1", node_ids={"alice", "bob"}, metadata={"algo": "louvain"}))
    return kg


def test_to_json_emits_valid_json_with_three_top_level_keys():
    kg = _populated_kg()
    data = json.loads(kg.to_json())
    assert set(data.keys()) == {"nodes", "edges", "clusters"}
    assert {n["id"] for n in data["nodes"]} == {"alice", "bob"}
    assert data["edges"][0]["confidence"] == 0.9
    assert data["edges"][0]["start_time"] == "2024-01-01"
    assert data["clusters"][0]["id"] == "c1"


def test_to_json_for_empty_kg_returns_empty_arrays():
    data = json.loads(EnhancedKG().to_json())
    assert data == {"nodes": [], "edges": [], "clusters": []}


def test_to_json_ld_uses_kg_prefixes_and_context():
    kg = _populated_kg()
    data = json.loads(kg.to_json_ld())
    assert "@context" in data
    assert data["@context"]["@vocab"] == "https://schema.org/"
    assert data["nodes"][0]["@id"].startswith("kg:node/")
    assert data["edges"][0]["source"]["@id"].startswith("kg:node/")
    assert data["clusters"][0]["@id"].startswith("kg:cluster/")
    # Optional fields only appear when set.
    assert data["edges"][0]["start_time"] == "2024-01-01"
    assert data["edges"][0]["confidence"] == 0.9
    assert "is_negated" not in data["edges"][0]


def test_to_json_ld_emits_is_negated_only_when_true():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    kg.add_node(KGNode(id="bob"))
    kg.add_edge(
        KGEdge(
            source="alice",
            target="bob",
            relationship_type="KNOWS",
            relationship_detail="...",
            is_negated=True,
        )
    )
    data = json.loads(kg.to_json_ld())
    assert data["edges"][0]["is_negated"] is True


def test_to_enriched_format_top_level_keys_and_none_for_no_communities():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    kg.add_node(KGNode(id="bob"))
    kg.add_edge(_edge())
    data = json.loads(kg.to_enriched_format())
    assert set(data.keys()) == {"entities", "relationships", "communities"}
    assert data["communities"] is None  # explicit signal for "no clusters"


def test_to_enriched_format_backcompat_lifts_confidence_from_metadata():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    kg.add_node(KGNode(id="bob"))
    # Confidence stored only in metadata (older shape); enriched export
    # should still surface it at the top level.
    kg.add_edge(
        KGEdge(
            source="alice",
            target="bob",
            relationship_type="KNOWS",
            relationship_detail="...",
            metadata={"confidence": 0.42, "source_ref": "chunk_1"},
        )
    )
    data = json.loads(kg.to_enriched_format())
    edge = data["relationships"][0]
    assert edge["confidence"] == 0.42
    assert edge["source_ref"] == "chunk_1"


def test_to_enriched_format_emits_is_negated_only_when_true():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    kg.add_node(KGNode(id="bob"))
    kg.add_edge(
        KGEdge(
            source="alice",
            target="bob",
            relationship_type="KNOWS",
            relationship_detail="...",
            is_negated=True,
        )
    )
    data = json.loads(kg.to_enriched_format())
    assert data["relationships"][0]["is_negated"] is True


# ---------------------------------------------------------------------------
# EnhancedKG — save_* (file IO)
# ---------------------------------------------------------------------------


def test_save_json_writes_valid_json(tmp_path):
    kg = _populated_kg()
    target = tmp_path / "nested" / "out.json"
    kg.save_json(str(target))
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["nodes"][0]["id"] in {"alice", "bob"}


def test_save_json_ld_writes_valid_json_ld(tmp_path):
    kg = _populated_kg()
    target = tmp_path / "out.jsonld"
    kg.save_json_ld(str(target))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "@context" in data


def test_save_enriched_format_writes_valid_json(tmp_path):
    kg = _populated_kg()
    target = tmp_path / "out.enriched.json"
    kg.save_enriched_format(str(target))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"entities", "relationships", "communities"}


# ---------------------------------------------------------------------------
# EnhancedKG.from_enriched_relationships
# ---------------------------------------------------------------------------


def test_from_enriched_relationships_builds_complete_kg():
    nodes = [KGNode(id="alice", type="Person"), KGNode(id="bob", type="Person")]
    rels = [_enriched()]
    kg = EnhancedKG.from_enriched_relationships(nodes, rels)
    assert set(kg.nodes.keys()) == {"alice", "bob"}
    assert len(kg.edges) == 1
    assert kg.edges[0].source == "alice"
    assert kg.edges[0].target == "bob"


def test_from_enriched_relationships_empty_inputs_yield_empty_kg():
    kg = EnhancedKG.from_enriched_relationships([], [])
    assert kg.nodes == {}
    assert kg.edges == []


# ---------------------------------------------------------------------------
# EnhancedKG.add_entity_embeddings
# ---------------------------------------------------------------------------


class _StubEmbeddingProvider:
    """Minimal embedding provider stub.

    Returns a deterministic per-text vector so we can assert the exact
    embedding each node receives.
    """

    def __init__(self):
        self.last_call: list[str] | None = None

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.last_call = list(texts)
        # Use the text length as a fingerprint; trivially deterministic.
        return [[float(len(t)), 1.0] for t in texts]


def test_add_entity_embeddings_default_uses_node_ids_as_text():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    kg.add_node(KGNode(id="bob"))
    provider = _StubEmbeddingProvider()
    kg.add_entity_embeddings(provider)
    assert set(provider.last_call) == {"alice", "bob"}
    assert kg.get_node("alice").embedding == [5.0, 1.0]  # len("alice")
    assert kg.get_node("bob").embedding == [3.0, 1.0]  # len("bob")


def test_add_entity_embeddings_uses_explicit_text_mapping():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    kg.add_node(KGNode(id="bob"))
    provider = _StubEmbeddingProvider()
    kg.add_entity_embeddings(
        provider,
        entity_texts={"alice": "Alice Liddell", "bob": "Robert"},
    )
    assert set(provider.last_call) == {"Alice Liddell", "Robert"}


def test_add_entity_embeddings_no_op_on_empty_graph():
    kg = EnhancedKG()
    provider = _StubEmbeddingProvider()
    kg.add_entity_embeddings(provider)
    assert provider.last_call == []
