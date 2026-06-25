#!/usr/bin/env python3
"""Multi-document reasoning example — no LLM required.

Demonstrates :class:`drg.reasoning.MultiDocumentReasoner` over a small
hand-crafted knowledge graph spanning two documents.  Inference rules
derive new edges that were not directly extracted from the text.

Run::

    python examples/multi_document_reasoning_example.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drg.graph import EnhancedKG, GraphMerger
from drg.graph.builders import build_enhanced_kg
from drg.reasoning import (
    MultiDocumentReasoner,
    ReasoningConfig,
)


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Build a small multi-document KG (no LLM needed)
# ---------------------------------------------------------------------------


def build_multi_doc_kg() -> EnhancedKG:
    """Construct a KG from two fictional documents about Apple / Beats."""
    base = EnhancedKG()
    merger = GraphMerger()

    # Document 1: Apple acquires Beats; Beats is located in California
    merger.merge(
        base,
        build_enhanced_kg(
            entities_typed=[
                ("Apple Inc.", "Company"),
                ("Beats Electronics", "Company"),
                ("California", "Place"),
            ],
            triples=[
                ("Apple Inc.", "acquired", "Beats Electronics"),
                ("Beats Electronics", "located_in", "California"),
            ],
            document_id="doc_1_apple_news",
        ),
        document_id="doc_1_apple_news",
    )

    # Document 2: Jimmy Iovine and Dr. Dre founded Beats
    merger.merge(
        base,
        build_enhanced_kg(
            entities_typed=[
                ("Jimmy Iovine", "Person"),
                ("Dr. Dre", "Person"),
                ("Beats Electronics", "Company"),
            ],
            triples=[
                ("Jimmy Iovine", "founded", "Beats Electronics"),
                ("Dr. Dre", "founded", "Beats Electronics"),
                ("Jimmy Iovine", "collaborated_with", "Dr. Dre"),
            ],
            document_id="doc_2_founders_bio",
        ),
        document_id="doc_2_founders_bio",
    )

    return base


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main() -> None:
    _section("Building multi-document KG")
    kg = build_multi_doc_kg()
    print(f"  nodes : {len(kg.nodes)}")
    print(f"  edges : {len(kg.edges)}")

    # Print edges before reasoning
    _section("Directly extracted edges")
    for edge in kg.edges:
        print(f"  {edge.source} --[{edge.relationship_type}]--> {edge.target}")

    # ---------------------------------------------------------------------------
    # Run reasoning with default rules
    # ---------------------------------------------------------------------------
    _section("Running MultiDocumentReasoner (all rules)")
    cfg = ReasoningConfig(
        min_confidence=0.25,
        # Optionally disable specific rules:
        # disabled_rules=frozenset({"path_bridge"}),
    )
    reasoner = MultiDocumentReasoner(config=cfg)
    report = reasoner.reason(kg, document_id="doc_2_founders_bio")

    print(f"  Rules applied       : {report.rules_applied}")
    print(f"  Candidate edges     : {report.edges_inferred}")
    print(f"  Edges added to KG   : {report.edges_added}")
    print(f"  Skipped (low conf)  : {report.skipped_low_confidence}")

    # ---------------------------------------------------------------------------
    # Inspect inferred edges
    # ---------------------------------------------------------------------------
    _section("Inferred edges")
    for inferred in report.inferred_edges:
        print(
            f"  [{inferred.rule_name}] {inferred.source} "
            f"--[{inferred.relationship_type}]--> {inferred.target} "
            f"(conf={inferred.confidence:.3f})"
        )

    _section("Full edges in KG after reasoning")
    for edge in kg.edges:
        tag = " [INFERRED]" if edge.metadata.get("inferred") else ""
        print(f"  {edge.source} --[{edge.relationship_type}]--> {edge.target}{tag}")

    # ---------------------------------------------------------------------------
    # Dry-run mode (inspect without modifying KG)
    # ---------------------------------------------------------------------------
    _section("Dry-run mode (no mutations)")
    kg2 = build_multi_doc_kg()
    dry_report = reasoner.reason(kg2, dry_run=True)
    print(f"  Edges inferred (dry): {dry_report.edges_inferred}")
    print(f"  Edges added  (dry)  : {dry_report.edges_added}")
    assert dry_report.edges_added == 0, "dry_run should not write edges"
    print("  Assertion passed: KG unchanged in dry-run mode")

    # ---------------------------------------------------------------------------
    # Reasoning history
    # ---------------------------------------------------------------------------
    history = (kg.metadata or {}).get("reasoning_history", [])
    _section(f"Reasoning history ({len(history)} pass(es))")
    for entry in history:
        print(f"  run: {entry.get('document_id')} — rules: {entry.get('rules_applied')}")

    # ---------------------------------------------------------------------------
    # Export inferred edges as JSON
    # ---------------------------------------------------------------------------
    _section("Inferred edge report (JSON excerpt)")
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()
