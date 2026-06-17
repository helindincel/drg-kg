#!/usr/bin/env python3
"""Query & Reasoning Layer — end-to-end demo (no LLM required).

Builds the same Apple / Beats / Jimmy Iovine multi-document KG used in
``multi_document_reasoning_example.py``, runs inference, then demonstrates
the :class:`drg.query.GraphQuery` API:

- entity lookup & fuzzy search
- relationship & evidence retrieval
- multi-hop neighborhoods
- path finding & explanation
- community exploration
- related-entity ranking

Run::

    python examples/query_layer_example.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drg.graph import EnhancedKG, GraphMerger
from drg.graph.builders import build_enhanced_kg
from drg.graph.kg_core import Cluster
from drg.query import GraphQuery
from drg.reasoning import MultiDocumentReasoner, ReasoningConfig


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _build_kg() -> EnhancedKG:
    base = EnhancedKG()
    merger = GraphMerger()

    merger.merge(
        base,
        build_enhanced_kg(
            entities_typed=[("Apple", "Company"), ("Beats", "Company")],
            triples=[("Apple", "ACQUIRED", "Beats")],
            document_id="doc_A_apple_news",
        ),
        document_id="doc_A_apple_news",
    )
    merger.merge(
        base,
        build_enhanced_kg(
            entities_typed=[("Jimmy Iovine", "Person"), ("Beats", "Company")],
            triples=[("Jimmy Iovine", "FOUNDED", "Beats")],
            document_id="doc_B_iovine_bio",
        ),
        document_id="doc_B_iovine_bio",
    )
    merger.merge(
        base,
        build_enhanced_kg(
            entities_typed=[
                ("Beats", "Company"),
                ("Culver City", "Place"),
            ],
            triples=[("Beats", "HEADQUARTERED_IN", "Culver City")],
            document_id="doc_C_location",
        ),
        document_id="doc_C_location",
    )

    base.add_cluster(
        Cluster(
            id="music_tech",
            node_ids={"Apple", "Beats", "Jimmy Iovine"},
            metadata={"theme": "music technology"},
        )
    )

    MultiDocumentReasoner(config=ReasoningConfig(min_confidence=0.5)).reason(
        base,
        record_history=False,
    )
    return base


def main() -> int:
    kg = _build_kg()
    gq = GraphQuery(kg)

    _section("1. Entity lookup")
    apple = gq.entity("Apple")
    print(f"  Apple type={apple.type!r} cluster={apple.cluster_id!r}")

    matches = gq.find_entities("beats", entity_type="Company")
    for m in matches[:3]:
        print(f"  match: {m.entity.id} score={m.score:.2f} reasons={m.match_reasons}")

    _section("2. Relationships & evidence")
    acquired = gq.relations(source="Apple", relationship_type="ACQUIRED")
    for e in acquired:
        print(f"  {e.source} --[{e.relationship_type}]--> {e.target}")
        print(f"    docs: {e.provenance.source_documents}")

    bundle = gq.evidence_for("Apple", "ACQUIRED", "Beats")
    print(f"  evidence summary: {bundle.summary}")

    _section("3. Multi-hop neighborhood (Apple, 2 hops)")
    hood = gq.neighbors("Apple", hops=2, max_edges=20)
    print(f"  entities ({len(hood.entities)}): {[e.id for e in hood.entities]}")
    print(f"  edges: {len(hood.edges)}")

    _section("4. Path finding: Apple → Jimmy Iovine")
    paths = gq.find_paths("Apple", "Jimmy Iovine", max_hops=3, max_paths=3)
    for i, path in enumerate(paths, 1):
        via = " → ".join(path.nodes)
        print(f"  path {i}: {via}  (hops={path.hop_count}, conf={path.confidence})")

    _section("5. Explain connection")
    exp = gq.explain("Apple", "Jimmy Iovine", max_hops=3)
    print(f"  connected: {exp.connected}")
    print(f"  summary: {exp.summary}")
    print(f"  evidence items: {len(exp.evidence)}")

    _section("6. Related companies")
    related = gq.related_entities(
        "Apple",
        mode="shortest_path",
        entity_type="Company",
        hops=3,
        limit=5,
    )
    for r in related:
        print(f"  {r.entity.id}  score={r.score:.3f}  mode={r.relation_mode}")

    _section("7. Community exploration")
    comm = gq.community_of("Apple")
    if comm:
        print(f"  cluster={comm.cluster_id} members={list(comm.node_ids)}")
    print(f"  community neighbors: {gq.community_neighbors('Apple')}")

    _section("8. Free-text search")
    answer = gq.search("Apple (ACQUIRED)")
    print(f"  answer: {answer.answer}")
    print(f"  seed entities: {answer.seed_entities}")
    print(f"  edges with provenance: {len(answer.edges)}")

    _section("9. EnhancedKG.query() convenience")
    via_kg = kg.query().entity("Beats")
    print(f"  kg.query().entity('Beats').type = {via_kg.type!r}")

    _section("10. Serialize explanation to JSON")
    print(json.dumps(exp.to_dict(), indent=2)[:800] + "\n  ...")

    print()
    print("OK — query layer demo finished without an LLM call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
