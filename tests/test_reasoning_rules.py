"""Unit tests for built-in multi-document inference rules."""

from __future__ import annotations

from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode
from drg.reasoning._rules import (
    CompositionRule,
    InverseRule,
    PathBridgeRule,
    SymmetricRule,
    TransitiveRule,
)


def _edge(
    source: str,
    target: str,
    relationship_type: str,
    *,
    confidence: float | None = None,
    metadata: dict | None = None,
) -> KGEdge:
    return KGEdge(
        source=source,
        target=target,
        relationship_type=relationship_type,
        relationship_detail=relationship_type,
        confidence=confidence,
        metadata=metadata or {},
    )


def _kg_with_edges(*edges: KGEdge) -> EnhancedKG:
    kg = EnhancedKG()
    for edge in edges:
        if edge.source not in kg.nodes:
            kg.add_node(KGNode(id=edge.source))
        if edge.target not in kg.nodes:
            kg.add_node(KGNode(id=edge.target))
        kg.add_edge(edge)
    return kg


def test_inverse_rule_infers_child_of_from_parent_of():
    kg = _kg_with_edges(_edge("Alice", "Bob", "parent_of", confidence=0.9))
    inferred = InverseRule().apply(kg)
    assert len(inferred) == 1
    assert inferred[0].relationship_type == "child_of"
    assert inferred[0].source == "Bob"
    assert inferred[0].target == "Alice"


def test_inverse_rule_skips_when_inverse_already_present():
    kg = _kg_with_edges(
        _edge("Alice", "Bob", "parent_of"),
        _edge("Bob", "Alice", "child_of"),
    )
    assert InverseRule().apply(kg) == []


def test_symmetric_rule_infers_reverse_related_to():
    kg = _kg_with_edges(_edge("A", "B", "related_to", confidence=0.8))
    inferred = SymmetricRule().apply(kg)
    assert len(inferred) == 1
    assert inferred[0].source == "B"
    assert inferred[0].target == "A"


def test_transitive_rule_infers_part_of_chain():
    kg = _kg_with_edges(
        _edge("Wheel", "Car", "part_of"),
        _edge("Car", "Fleet", "part_of"),
    )
    inferred = TransitiveRule().apply(kg)
    assert any(
        edge.source == "Wheel" and edge.target == "Fleet" and edge.relationship_type == "part_of"
        for edge in inferred
    )


def test_composition_rule_operates_in():
    kg = _kg_with_edges(
        _edge("Acme", "Factory", "owns"),
        _edge("Factory", "Berlin", "located_in"),
    )
    inferred = CompositionRule().apply(kg)
    assert any(edge.relationship_type == "operates_in" for edge in inferred)


def test_path_bridge_rule_connects_cross_document_hops():
    kg = _kg_with_edges(
        _edge(
            "Alice",
            "Acme",
            "works_at",
            metadata={"provenance": {"document_id": "doc_1"}},
        ),
        _edge(
            "Acme",
            "Widget",
            "produces",
            metadata={"provenance": {"document_id": "doc_2"}},
        ),
    )
    inferred = PathBridgeRule().apply(kg)
    assert any(
        edge.relationship_type == "connected_via"
        and edge.source == "Alice"
        and edge.target == "Widget"
        for edge in inferred
    )
