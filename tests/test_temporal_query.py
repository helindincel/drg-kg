"""Integration tests for temporal GraphQuery methods."""

from __future__ import annotations

from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode
from drg.query import GraphQuery


def _apple_ceo_kg() -> EnhancedKG:
    kg = EnhancedKG()
    for name, etype in [
        ("Apple", "Company"),
        ("Steve Jobs", "Person"),
        ("Tim Cook", "Person"),
        ("Microsoft", "Company"),
        ("LinkedIn", "Company"),
    ]:
        kg.add_node(KGNode(id=name, type=etype))

    kg.add_edge(
        KGEdge(
            source="Steve Jobs",
            target="Apple",
            relationship_type="CEO_OF",
            relationship_detail="Steve Jobs was CEO of Apple",
            start_time="1997",
            end_time="2011",
        )
    )
    kg.add_edge(
        KGEdge(
            source="Tim Cook",
            target="Apple",
            relationship_type="CEO_OF",
            relationship_detail="Tim Cook is CEO of Apple",
            start_time="2011",
            end_time=None,
        )
    )
    kg.add_edge(
        KGEdge(
            source="Microsoft",
            target="LinkedIn",
            relationship_type="acquired",
            relationship_detail="Microsoft acquired LinkedIn",
            start_time="2016",
            end_time=None,
        )
    )
    return kg


def test_role_holders_at_2008():
    gq = GraphQuery(_apple_ceo_kg())
    holders = gq.role_holders_at("Apple", "CEO_OF", "2008")
    assert len(holders) == 1
    assert holders[0].source == "Steve Jobs"


def test_role_holders_at_2015():
    gq = GraphQuery(_apple_ceo_kg())
    holders = gq.role_holders_at("Apple", "CEO_OF", "2015")
    assert len(holders) == 1
    assert holders[0].source == "Tim Cook"


def test_relations_active_at_acquisitions_2020():
    gq = GraphQuery(_apple_ceo_kg())
    active = gq.relations_active_at(
        "2020",
        source="Microsoft",
        relationship_type="acquired",
    )
    assert len(active) == 1
    assert active[0].target == "LinkedIn"


def test_temporal_timeline_for_apple_ceo():
    gq = GraphQuery(_apple_ceo_kg())
    timeline = gq.temporal_timeline(target="Apple", relationship_type="CEO_OF")
    assert len(timeline.entries) == 2


def test_temporal_conflicts_clean_for_sequential_ceos():
    gq = GraphQuery(_apple_ceo_kg())
    assert gq.temporal_conflicts(relationship_type="CEO_OF", target="Apple") == []


def test_edge_view_includes_valid_from():
    gq = GraphQuery(_apple_ceo_kg())
    edges = gq.relations(source="Tim Cook", relationship_type="CEO_OF")
    assert edges[0].valid_from == "2011"
