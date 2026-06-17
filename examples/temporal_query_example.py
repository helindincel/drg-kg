#!/usr/bin/env python3
"""Temporal knowledge graph query examples (Apple CEO succession)."""

from __future__ import annotations

from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode
from drg.query import GraphQuery


def build_apple_ceo_kg() -> EnhancedKG:
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
            relationship_detail="Steve Jobs was CEO of Apple from 1997 to 2011.",
            start_time="1997",
            end_time="2011",
            confidence=0.95,
            metadata={"evidence": "Steve Jobs was CEO of Apple from 1997 to 2011."},
        )
    )
    kg.add_edge(
        KGEdge(
            source="Tim Cook",
            target="Apple",
            relationship_type="CEO_OF",
            relationship_detail="Tim Cook became CEO of Apple in 2011.",
            start_time="2011",
            end_time=None,
            confidence=0.95,
            metadata={"evidence": "Tim Cook became CEO of Apple in 2011."},
        )
    )
    kg.add_edge(
        KGEdge(
            source="Microsoft",
            target="LinkedIn",
            relationship_type="acquired",
            relationship_detail="Microsoft acquired LinkedIn in 2016.",
            start_time="2016",
            end_time=None,
            confidence=0.9,
        )
    )
    return kg


def main() -> None:
    gq = GraphQuery(build_apple_ceo_kg())

    print("=== Who was CEO of Apple in 2008? ===")
    for edge in gq.role_holders_at("Apple", "CEO_OF", "2008"):
        print(f"  {edge.source} (valid_from={edge.valid_from}, confidence={edge.confidence})")

    print("\n=== Who was CEO of Apple in 2015? ===")
    for edge in gq.role_holders_at("Apple", "CEO_OF", "2015"):
        print(f"  {edge.source}")

    print("\n=== What did Microsoft acquire (active in 2020)? ===")
    for edge in gq.relations_active_at("2020", source="Microsoft", relationship_type="acquired"):
        print(f"  {edge.source} —[{edge.relationship_type}]→ {edge.target}")

    print("\n=== CEO timeline for Apple ===")
    timeline = gq.temporal_timeline(target="Apple", relationship_type="CEO_OF")
    for entry in timeline.entries:
        temporal = entry.temporal.to_dict() if entry.temporal else {}
        print(f"  {entry.source}: {temporal.get('valid_from')} → {temporal.get('valid_to')}")

    print("\n=== Changes between 2010 and 2012 ===")
    report = gq.changes_between("2010", "2012", relationship_type="CEO_OF")
    for entry in report.started:
        print(f"  Started: {entry.source} → {entry.target}")
    for entry in report.ended:
        print(f"  Ended: {entry.source} → {entry.target}")

    print("\n=== Temporal conflicts (should be empty for clean succession) ===")
    conflicts = gq.temporal_conflicts(relationship_type="CEO_OF", target="Apple")
    print(f"  {len(conflicts)} conflict(s)")


if __name__ == "__main__":
    main()
