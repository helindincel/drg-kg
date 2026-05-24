#!/usr/bin/env python3
"""
Quickstart 3 — Biomedical abstract -> Knowledge Graph
======================================================

Showcases DRG on a research-style domain: extracting Drugs, Diseases, Genes,
and their causal / therapeutic relationships from a short biomedical
paragraph.

What this script does
---------------------
1. Defines an `EnhancedDRGSchema` for biomedical entity / relation types
   (`treats`, `inhibits`, `expressed_in`, `causes`, ...).
2. Runs `extract_typed()` over a sample paragraph synthesized in the style of
   a biomedical abstract.
3. Persists the resulting KG as JSON next to this file.

Why this matters
----------------
Biomedical literature is one of the canonical KG extraction targets
(BioGRID, Hetionet, PrimeKG style graphs). DRG's declarative schema lets a
researcher iterate on the *ontology* (what counts as a `Drug`, what relations
exist between a `Gene` and a `Disease`) without touching prompts.

Prerequisites
-------------
- An LLM provider key in the environment (default: `OPENAI_API_KEY`).
- For sensitive content you probably want a stronger model — set
  `DRG_MODEL=openai/gpt-4o` (or similar) before running.

Run
---
    python examples/quickstarts/03_biomedical.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drg import (
    KG,
    EnhancedDRGSchema,
    EntityType,
    Relation,
    RelationGroup,
    extract_typed,
)

SAMPLE_TEXT = """
Metformin is a first-line oral medication used to treat type 2 diabetes
mellitus. It primarily works by inhibiting hepatic glucose production
through activation of the AMPK pathway, which is regulated in part by the
LKB1 gene. Recent studies suggest that metformin may also reduce the risk
of colorectal cancer, although the underlying mechanism is not fully
understood. The drug is generally well tolerated, but it can cause
gastrointestinal side effects and, rarely, lactic acidosis.
""".strip()


def build_schema() -> EnhancedDRGSchema:
    """Schema for biomedical entity / relation extraction."""
    return EnhancedDRGSchema(
        entity_types=[
            EntityType(
                name="Drug",
                description="A pharmaceutical compound or medication",
                examples=["Metformin"],
            ),
            EntityType(
                name="Disease",
                description="A medical condition or disease entity",
                examples=["type 2 diabetes mellitus", "colorectal cancer"],
            ),
            EntityType(
                name="Gene",
                description="A gene or gene product",
                examples=["LKB1"],
            ),
            EntityType(
                name="Pathway",
                description="A biological signaling or metabolic pathway",
                examples=["AMPK pathway"],
            ),
            EntityType(
                name="SideEffect",
                description="An adverse effect attributed to a drug",
                examples=["gastrointestinal side effects", "lactic acidosis"],
            ),
        ],
        relation_groups=[
            RelationGroup(
                name="therapeutic",
                description="Drug -> disease therapeutic relationships",
                relations=[
                    Relation("treats", "Drug", "Disease"),
                    Relation("reduces_risk_of", "Drug", "Disease"),
                ],
            ),
            RelationGroup(
                name="mechanism",
                description="How drugs act biologically",
                relations=[
                    Relation("inhibits", "Drug", "Pathway"),
                    Relation("activates", "Drug", "Pathway"),
                    Relation("regulates", "Gene", "Pathway"),
                ],
            ),
            RelationGroup(
                name="safety",
                description="Adverse reactions and tolerability",
                relations=[
                    Relation("causes", "Drug", "SideEffect"),
                ],
            ),
        ],
        auto_discovery=False,
    )


def main() -> int:
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print(
            "No LLM API key detected. Set OPENAI_API_KEY (default) or use a "
            "different provider via DRG_MODEL + the matching key.",
            file=sys.stderr,
        )
        return 1

    schema = build_schema()
    print(
        f"Schema: {len(schema.entity_types)} entity types, "
        f"{sum(len(rg.relations) for rg in schema.relation_groups)} relations"
    )
    print(f"\nInput text ({len(SAMPLE_TEXT)} chars):\n{SAMPLE_TEXT}\n")

    entities, triples = extract_typed(SAMPLE_TEXT, schema)
    kg = KG.from_typed(entities, triples)

    print(f"Extracted {len(entities)} entities and {len(triples)} triples.\n")

    print("Entities:")
    for name, etype in entities:
        print(f"  - {name}  [{etype}]")

    print("\nTriples:")
    for src, rel, dst in triples:
        print(f"  ({src})  --{rel}-->  ({dst})")

    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(kg.to_json(indent=2), encoding="utf-8")
    print(f"\nKG written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
