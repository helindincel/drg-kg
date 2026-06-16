"""Unit tests for drg.graph.kg_core — KGNode, KGEdge, Cluster, EnhancedKG."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from drg.graph.kg_core import Cluster, EnhancedKG, KGEdge, KGNode
from drg.graph.relationship_model._enriched import EnrichedRelationship
from drg.graph.relationship_model._types import RelationshipType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(id_: str, type_: str = "Entity") -> KGNode:
    return KGNode(id=id_, type=type_)


def _edge(src: str, tgt: str, rel: str = "related_to") -> KGEdge:
    return KGEdge(source=src, target=tgt, relationship_type=rel, relationship_detail=f"{src} {rel} {tgt}")


def _kg_with_two_nodes() -> EnhancedKG:
    kg = EnhancedKG()
    kg.add_node(_node("A"))
    kg.add_node(_node("B"))
    return kg


# ---------------------------------------------------------------------------
# KGNode
# ---------------------------------------------------------------------------


class TestKGNode:
    def test_minimal_creation(self):
        node = KGNode(id="Alice")
        assert node.id == "Alice"
        assert node.type is None

    def test_full_creation(self):
        node = KGNode(id="Alice", type="Person", properties={"age": 30}, metadata={"src": "doc1"})
        assert node.type == "Person"
        assert node.properties["age"] == 30
        assert node.metadata["src"] == "doc1"

    def test_empty_id_raises(self):
        with pytest.raises(ValueError):
            KGNode(id="")

    def test_to_dict_minimal(self):
        node = KGNode(id="X", type="Thing")
        d = node.to_dict()
        assert d["id"] == "X"
        assert d["type"] == "Thing"
        assert "properties" not in d
        assert "metadata" not in d

    def test_to_dict_includes_properties_when_set(self):
        node = KGNode(id="X", type="T", properties={"k": "v"})
        assert node.to_dict()["properties"] == {"k": "v"}

    def test_to_dict_includes_metadata_when_set(self):
        node = KGNode(id="X", type="T", metadata={"confidence": 0.9})
        assert node.to_dict()["metadata"] == {"confidence": 0.9}

    def test_to_dict_includes_embedding_when_set(self):
        node = KGNode(id="X", type="T", embedding=[0.1, 0.2, 0.3])
        assert node.to_dict()["embedding"] == [0.1, 0.2, 0.3]

    def test_from_dict_round_trip(self):
        node = KGNode(id="Bob", type="Person", properties={"role": "engineer"}, metadata={"src": "x"})
        restored = KGNode.from_dict(node.to_dict())
        assert restored.id == node.id
        assert restored.type == node.type
        assert restored.properties == node.properties

    def test_from_dict_without_type(self):
        node = KGNode.from_dict({"id": "Z"})
        assert node.id == "Z"
        assert node.type is None


# ---------------------------------------------------------------------------
# KGEdge
# ---------------------------------------------------------------------------


class TestKGEdge:
    def test_minimal_creation(self):
        edge = _edge("A", "B")
        assert edge.source == "A"
        assert edge.target == "B"

    def test_empty_source_raises(self):
        with pytest.raises(ValueError):
            KGEdge(source="", target="B", relationship_type="r", relationship_detail="d")

    def test_empty_target_raises(self):
        with pytest.raises(ValueError):
            KGEdge(source="A", target="", relationship_type="r", relationship_detail="d")

    def test_same_source_target_raises(self):
        with pytest.raises(ValueError):
            KGEdge(source="A", target="A", relationship_type="r", relationship_detail="d")

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            KGEdge(source="A", target="B", relationship_type="r", relationship_detail="d", confidence=1.5)

    def test_negative_confidence_raises(self):
        with pytest.raises(ValueError):
            KGEdge(source="A", target="B", relationship_type="r", relationship_detail="d", confidence=-0.1)

    def test_valid_confidence_accepted(self):
        edge = KGEdge(source="A", target="B", relationship_type="r", relationship_detail="d", confidence=0.75)
        assert edge.confidence == 0.75

    def test_to_dict_basic(self):
        edge = _edge("A", "B", "produces")
        d = edge.to_dict()
        assert d["source"] == "A"
        assert d["target"] == "B"
        assert d["relationship_type"] == "produces"

    def test_to_dict_excludes_optional_fields_when_none(self):
        edge = _edge("A", "B")
        d = edge.to_dict()
        assert "confidence" not in d
        assert "start_time" not in d
        assert "is_negated" not in d

    def test_to_dict_includes_temporal_when_set(self):
        edge = KGEdge(source="A", target="B", relationship_type="r",
                      relationship_detail="d", start_time="2020-01", end_time="2021-01")
        d = edge.to_dict()
        assert d["start_time"] == "2020-01"
        assert d["end_time"] == "2021-01"

    def test_to_dict_includes_is_negated_when_true(self):
        edge = KGEdge(source="A", target="B", relationship_type="r",
                      relationship_detail="d", is_negated=True)
        assert edge.to_dict()["is_negated"] is True

    def test_from_dict_round_trip(self):
        edge = KGEdge(source="A", target="B", relationship_type="produces",
                      relationship_detail="A produces B", confidence=0.8,
                      start_time="2020", is_negated=False)
        restored = KGEdge.from_dict(edge.to_dict())
        assert restored.source == "A"
        assert restored.confidence == 0.8
        assert restored.start_time == "2020"

    def test_from_enriched_relationship(self):
        rel = EnrichedRelationship(
            source="Apple", target="iPhone",
            relationship_type=RelationshipType.PRODUCES,
            relationship_detail="Apple manufactures iPhone",
            confidence=0.95,
        )
        edge = KGEdge.from_enriched_relationship(rel)
        assert edge.source == "Apple"
        assert edge.target == "iPhone"
        assert edge.confidence == 0.95
        assert edge.relationship_type == RelationshipType.PRODUCES.value


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


class TestCluster:
    def test_minimal_creation(self):
        c = Cluster(id="c1", node_ids={"A", "B"})
        assert c.id == "c1"
        assert "A" in c.node_ids

    def test_empty_id_raises(self):
        with pytest.raises(ValueError):
            Cluster(id="", node_ids={"A"})

    def test_empty_node_ids_raises(self):
        with pytest.raises(ValueError):
            Cluster(id="c1", node_ids=set())

    def test_to_dict(self):
        c = Cluster(id="c1", node_ids={"A", "B"}, metadata={"label": "tech"})
        d = c.to_dict()
        assert d["id"] == "c1"
        assert set(d["node_ids"]) == {"A", "B"}
        assert d["metadata"]["label"] == "tech"

    def test_from_dict_round_trip(self):
        c = Cluster(id="c2", node_ids={"X", "Y", "Z"})
        restored = Cluster.from_dict(c.to_dict())
        assert restored.id == "c2"
        assert restored.node_ids == {"X", "Y", "Z"}


# ---------------------------------------------------------------------------
# EnhancedKG — construction
# ---------------------------------------------------------------------------


class TestEnhancedKGConstruction:
    def test_empty_kg(self):
        kg = EnhancedKG()
        assert kg.nodes == {}
        assert kg.edges == []
        assert kg.clusters == {}

    def test_add_and_get_node(self):
        kg = EnhancedKG()
        kg.add_node(_node("Alice", "Person"))
        assert kg.get_node("Alice") is not None
        assert kg.get_node("Alice").type == "Person"  # type: ignore[union-attr]

    def test_get_nonexistent_node_returns_none(self):
        kg = EnhancedKG()
        assert kg.get_node("missing") is None

    def test_add_edge_with_existing_nodes(self):
        kg = _kg_with_two_nodes()
        kg.add_edge(_edge("A", "B"))
        assert len(kg.edges) == 1

    def test_add_edge_missing_source_raises(self):
        kg = EnhancedKG()
        kg.add_node(_node("B"))
        with pytest.raises(ValueError):
            kg.add_edge(_edge("A", "B"))

    def test_add_edge_missing_target_raises(self):
        kg = EnhancedKG()
        kg.add_node(_node("A"))
        with pytest.raises(ValueError):
            kg.add_edge(_edge("A", "B"))

    def test_add_cluster_with_valid_nodes(self):
        kg = _kg_with_two_nodes()
        c = Cluster(id="c1", node_ids={"A", "B"})
        kg.add_cluster(c)
        assert "c1" in kg.clusters

    def test_add_cluster_with_invalid_nodes_raises(self):
        kg = _kg_with_two_nodes()
        c = Cluster(id="c1", node_ids={"A", "MISSING"})
        with pytest.raises(ValueError, match="non-existent"):
            kg.add_cluster(c)


# ---------------------------------------------------------------------------
# EnhancedKG — serialization
# ---------------------------------------------------------------------------


class TestEnhancedKGSerialization:
    def _populated_kg(self) -> EnhancedKG:
        kg = _kg_with_two_nodes()
        kg.add_edge(_edge("A", "B", "produces"))
        kg.add_cluster(Cluster(id="c1", node_ids={"A", "B"}))
        return kg

    def test_to_json_is_valid_json(self):
        kg = self._populated_kg()
        parsed = json.loads(kg.to_json())
        assert "nodes" in parsed
        assert "edges" in parsed
        assert "clusters" in parsed

    def test_to_json_node_count(self):
        kg = self._populated_kg()
        parsed = json.loads(kg.to_json())
        assert len(parsed["nodes"]) == 2

    def test_to_json_edge_count(self):
        kg = self._populated_kg()
        parsed = json.loads(kg.to_json())
        assert len(parsed["edges"]) == 1

    def test_to_json_ld_has_context(self):
        kg = self._populated_kg()
        parsed = json.loads(kg.to_json_ld())
        assert "@context" in parsed

    def test_to_enriched_format_keys(self):
        kg = self._populated_kg()
        parsed = json.loads(kg.to_enriched_format())
        assert "entities" in parsed
        assert "relationships" in parsed
        assert "communities" in parsed

    def test_save_and_reload_json(self):
        kg = self._populated_kg()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "kg.json")
            kg.save_json(path)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert len(data["nodes"]) == 2

    def test_save_json_creates_parent_dirs(self):
        kg = _kg_with_two_nodes()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sub" / "dir" / "kg.json")
            kg.save_json(path)
            assert Path(path).exists()

    def test_empty_kg_to_json(self):
        kg = EnhancedKG()
        parsed = json.loads(kg.to_json())
        assert parsed["nodes"] == []
        assert parsed["edges"] == []
        assert parsed["clusters"] == []


# ---------------------------------------------------------------------------
# EnhancedKG — from_enriched_relationships
# ---------------------------------------------------------------------------


class TestEnhancedKGFromEnrichedRelationships:
    def test_builds_from_nodes_and_relationships(self):
        nodes = [KGNode(id="Apple", type="Company"), KGNode(id="iPhone", type="Product")]
        rels = [
            EnrichedRelationship(
                source="Apple", target="iPhone",
                relationship_type=RelationshipType.PRODUCES,
                relationship_detail="Apple makes iPhone",
            )
        ]
        kg = EnhancedKG.from_enriched_relationships(nodes, rels)
        assert "Apple" in kg.nodes
        assert len(kg.edges) == 1

    def test_empty_nodes_and_rels(self):
        kg = EnhancedKG.from_enriched_relationships([], [])
        assert kg.nodes == {}
        assert kg.edges == []
