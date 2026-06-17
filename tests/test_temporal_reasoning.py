"""Unit tests for temporal reasoning (overlap, conflict, timeline)."""

from __future__ import annotations

from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode
from drg.temporal import (
    build_timeline,
    changes_between,
    detect_conflicts,
    detect_overlaps,
    entity_state_transitions,
)


def _ceo_edge(source: str, valid_from: str, valid_to: str | None) -> KGEdge:
    return KGEdge(
        source=source,
        target="Apple",
        relationship_type="CEO_OF",
        relationship_detail=f"{source} CEO of Apple",
        start_time=valid_from,
        end_time=valid_to,
        confidence=0.9,
    )


def _apple_ceo_kg() -> EnhancedKG:
    kg = EnhancedKG()
    for name, etype in [("Apple", "Company"), ("Steve Jobs", "Person"), ("Tim Cook", "Person")]:
        kg.add_node(KGNode(id=name, type=etype))
    kg.add_edge(_ceo_edge("Steve Jobs", "1997", "2011"))
    kg.add_edge(_ceo_edge("Tim Cook", "2011", None))
    return kg


def test_detect_conflicts_no_conflict_for_sequential_ceos():
    kg = _apple_ceo_kg()
    conflicts = detect_conflicts(kg.edges, relationship_type="CEO_OF", target="Apple")
    assert conflicts == []


def test_detect_conflicts_concurrent_holders():
    kg = _apple_ceo_kg()
    kg.add_node(KGNode(id="John Doe", type="Person"))
    kg.add_edge(_ceo_edge("John Doe", "2008", "2010"))
    conflicts = detect_conflicts(kg.edges, relationship_type="CEO_OF", target="Apple")
    assert len(conflicts) >= 1
    assert conflicts[0].conflict_type == "concurrent_role_holders"


def test_detect_overlaps_same_triple():
    e1 = _ceo_edge("Steve Jobs", "2005", "2010")
    e2 = _ceo_edge("Steve Jobs", "2008", "2012")
    overlaps = detect_overlaps([e1, e2])
    assert len(overlaps) == 1


def test_build_timeline_ordered():
    kg = _apple_ceo_kg()
    timeline = build_timeline(kg.edges, target="Apple", relationship_type="CEO_OF")
    assert timeline.subject == "Apple"
    assert len(timeline.entries) == 2
    assert timeline.entries[0].source == "Steve Jobs"
    assert timeline.entries[1].source == "Tim Cook"


def test_entity_state_transitions():
    kg = _apple_ceo_kg()
    timeline = entity_state_transitions(kg.edges, "Steve Jobs", "CEO_OF")
    assert len(timeline.entries) == 1
    assert timeline.entries[0].target == "Apple"


def test_changes_between_detects_role_start():
    kg = _apple_ceo_kg()
    report = changes_between(kg.edges, "2010", "2012", relationship_type="CEO_OF")
    started_sources = {e.source for e in report.started}
    assert "Tim Cook" in started_sources
