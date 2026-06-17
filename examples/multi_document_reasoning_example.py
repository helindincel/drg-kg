#!/usr/bin/env python3
"""Multi-Document Reasoning — End-to-end demo (no LLM required).

This example mirrors :mod:`examples.incremental_update_example` but
goes one step further: after merging three small per-document graphs
into a single knowledge graph, it runs the
:class:`drg.reasoning.MultiDocumentReasoner` and shows the inferred
edges that **only become visible once you reason across documents**.

The headline scenario is the one from the design doc:

- Document A says ``Apple ACQUIRED Beats``.
- Document B says ``Jimmy Iovine FOUNDED Beats``.
- Document C says ``Beats is HEADQUARTERED_IN Culver City``.

After extraction + merge + reasoning the KG contains, alongside the
extracted edges:

- ``Beats acquired_by Apple`` (InverseRule)
- ``Beats founded_by Jimmy Iovine`` (InverseRule)
- ``Apple connected_via_beats Jimmy Iovine`` (PathBridgeRule)
- ``Apple connected_via_beats Culver City`` (PathBridgeRule)
- ``Jimmy Iovine connected_via_beats Culver City`` (PathBridgeRule)

Each inferred edge carries full provenance under
``metadata.inference``: the rule that fired, the evidence chain (with
the source documents), a human-readable explanation, and a confidence
score derived from the evidence.

Run::

    python examples/multi_document_reasoning_example.py

The script reuses the same patterns demonstrated in
``incremental_update_example.py`` — no LLM, no API key required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drg.graph import (
    EnhancedKG,
    GraphMerger,
)
from drg.graph.builders import build_enhanced_kg
from drg.reasoning import (
    MultiDocumentReasoner,
    ReasoningConfig,
)


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _kg_summary(kg: EnhancedKG, label: str) -> None:
    extracted = [e for e in kg.edges if not e.metadata.get("inferred")]
    inferred = [e for e in kg.edges if e.metadata.get("inferred")]
    print(
        f"  [{label}] nodes={len(kg.nodes)}  "
        f"edges={len(kg.edges)} "
        f"(extracted={len(extracted)}, inferred={len(inferred)})  "
        f"version={kg.metadata.get('version', '-')}"
    )


# ---------------------------------------------------------------------------
# Per-document fake "extractions"
#
# In a real pipeline these come from `drg.extract.extract_typed(...)` ->
# `build_enhanced_kg(...)`. We hand-build the (entities, triples) here
# so the demo is reproducible without an LLM.
# ---------------------------------------------------------------------------


def doc_a_kg() -> EnhancedKG:
    """Doc A — corporate news: 'Apple acquired Beats Electronics.'"""
    return build_enhanced_kg(
        entities_typed=[
            ("Apple", "Company"),
            ("Beats", "Company"),
        ],
        triples=[("Apple", "ACQUIRED", "Beats")],
        document_id="doc_A_apple_news",
    )


def doc_b_kg() -> EnhancedKG:
    """Doc B — biography: 'Jimmy Iovine founded Beats Electronics.'"""
    return build_enhanced_kg(
        entities_typed=[
            ("Jimmy Iovine", "Person"),
            ("Beats", "Company"),
        ],
        triples=[("Jimmy Iovine", "FOUNDED", "Beats")],
        document_id="doc_B_iovine_bio",
    )


def doc_c_kg() -> EnhancedKG:
    """Doc C — location facts: 'Beats is headquartered in Culver City,
    California, USA. Culver City is located in California. California
    is part of United States.'"""
    return build_enhanced_kg(
        entities_typed=[
            ("Beats", "Company"),
            ("Culver City", "Place"),
            ("California", "Place"),
            ("United States", "Place"),
        ],
        triples=[
            ("Beats", "HEADQUARTERED_IN", "Culver City"),
            ("Culver City", "part_of", "California"),
            ("California", "part_of", "United States"),
        ],
        document_id="doc_C_location_facts",
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> int:
    out_path = Path(__file__).with_suffix(".kg.json")
    if out_path.exists():
        out_path.unlink()

    _print_section("1. Bootstrap an empty global KG")
    print(f"  Output file: {out_path}")
    base = EnhancedKG()
    _kg_summary(base, "empty")

    merger = GraphMerger()

    _print_section("2. Ingest doc_A — 'Apple acquired Beats.'")
    merger.merge(base, doc_a_kg(), document_id="doc_A_apple_news")
    _kg_summary(base, "after doc_A")

    _print_section("3. Ingest doc_B — 'Jimmy Iovine founded Beats.'")
    merger.merge(base, doc_b_kg(), document_id="doc_B_iovine_bio")
    _kg_summary(base, "after doc_B")

    _print_section("4. Ingest doc_C — 'Beats HQ in Culver City, …'")
    merger.merge(base, doc_c_kg(), document_id="doc_C_location_facts")
    _kg_summary(base, "after doc_C")

    _print_section("5. Run multi-document reasoning")
    print(
        "  Built-in rules: inverse, symmetric, transitive, composition, path_bridge\n"
        "  All rules are pure functions of the merged graph — no LLM, no fabrication."
    )
    config = ReasoningConfig(min_confidence=0.5)
    report = MultiDocumentReasoner(config=config).reason(
        base,
        document_id="post_merge_reasoning",
    )
    print(f"  Report: {json.dumps(report.summary(), indent=2)}")
    _kg_summary(base, "after reasoning")

    _print_section("6. Inferred edges (with provenance)")
    inferred = [e for e in base.edges if e.metadata.get("inferred")]
    for e in inferred:
        inf = e.metadata["inference"]
        print(
            f"\n  {e.source}  --[{e.relationship_type}]-->  {e.target}"
            f"   (confidence={e.confidence:.2f})"
        )
        print(f"    rule:              {inf['rule']}")
        if inf.get("bridge_entity"):
            print(f"    bridge entity:     {inf['bridge_entity']}")
        print(f"    source documents:  {', '.join(inf['source_documents']) or '(unknown)'}")
        print("    evidence chain:")
        for link in inf["evidence_chain"]:
            src, rel, tgt = link["triple"]
            ref = link.get("source_ref", "(unknown)")
            print(f"      - {src} --[{rel}]--> {tgt}   ({ref})")
        print(f"    explanation:       {inf['explanation']}")

    _print_section("7. Find the 'Apple → Jimmy Iovine' bridge edge")
    bridge_edges = [
        e
        for e in inferred
        if e.metadata["inference"]["rule"] == "path_bridge"
        and {e.source, e.target} == {"Apple", "Jimmy Iovine"}
    ]
    if bridge_edges:
        print(
            "  Yes — the system independently inferred a connection between "
            "Apple and Jimmy Iovine, bridged by Beats, with evidence drawn "
            "from two different documents:"
        )
        edge = bridge_edges[0]
        for link in edge.metadata["inference"]["evidence_chain"]:
            src, rel, tgt = link["triple"]
            ref = link.get("source_ref", "(unknown)")
            print(f"    {src} {rel} {tgt}   ({ref})")
    else:
        print("  No bridge edge found — check the input documents and config.")

    _print_section("8. Persist and round-trip the KG")
    base.save_json(str(out_path))
    print(f"  Wrote: {out_path}")

    reloaded = EnhancedKG.load_json(str(out_path))
    reloaded_inferred = [e for e in reloaded.edges if e.metadata.get("inferred")]
    print(f"  Reloaded {len(reloaded.edges)} edges total")
    print(f"  Inferred edges preserved: {len(reloaded_inferred)}")
    print(f"  History entries: {len(reloaded.metadata.get('history', []))}")

    print()
    print("OK — multi-document reasoning demo finished without an LLM call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
