#!/usr/bin/env python3
"""
Quickstart 1 — Wikipedia-style biographical article -> Knowledge Graph
=======================================================================

Showcases the most generic DRG use case: turning an encyclopedic paragraph
about a person/event into a structured KG via a hand-defined `EnhancedDRGSchema`.

What this script does
---------------------
1. Defines an `EnhancedDRGSchema` for biographical content (Person, Place,
   Organization, Work) with three relation groups (affiliation, location,
   creation).
2. Runs `extract_typed()` over a short sample paragraph (Marie Curie).
3. Builds a `KG` (legacy lightweight graph) and prints a human-readable
   summary plus a JSON dump.

Prerequisites
-------------
- An LLM provider key in the environment (default: `OPENAI_API_KEY`).
- Override the model with `DRG_MODEL`, e.g.:

    DRG_MODEL=gemini/gemini-2.0-flash-exp GEMINI_API_KEY=... \
        python examples/quickstarts/01_wikipedia_article.py

Run
---
    python examples/quickstarts/01_wikipedia_article.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running directly from the repo root without `pip install -e .`
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
Marie Curie was a Polish-born physicist and chemist who later became a French
citizen. She conducted pioneering research on radioactivity at the University
of Paris. Curie discovered the elements polonium and radium and was awarded
the Nobel Prize in Physics in 1903, shared with her husband Pierre Curie and
Henri Becquerel. In 1911 she received a second Nobel Prize, this time in
Chemistry, becoming the first person to win the award in two different
scientific fields.
""".strip()


def build_schema() -> EnhancedDRGSchema:
    """Hand-crafted schema for biographical content."""
    return EnhancedDRGSchema(
        entity_types=[
            EntityType(
                name="Person",
                description="A real individual mentioned by name",
                examples=["Marie Curie", "Pierre Curie", "Henri Becquerel"],
            ),
            EntityType(
                name="Organization",
                description="A university, company, or institution",
                examples=["University of Paris"],
            ),
            EntityType(
                name="Place",
                description="A country, city, or geographic location",
                examples=["Poland", "France"],
            ),
            EntityType(
                name="Award",
                description="A formal prize or honour granted to a person",
                examples=["Nobel Prize in Physics", "Nobel Prize in Chemistry"],
            ),
            EntityType(
                name="Discovery",
                description="A scientific element, theory, or finding",
                examples=["polonium", "radium"],
            ),
        ],
        relation_groups=[
            RelationGroup(
                name="affiliation",
                description="How people connect to organizations or places",
                relations=[
                    Relation("worked_at", "Person", "Organization"),
                    Relation("born_in", "Person", "Place"),
                    Relation("citizen_of", "Person", "Place"),
                ],
            ),
            RelationGroup(
                name="achievement",
                description="Awards and discoveries attributed to a person",
                relations=[
                    Relation("received", "Person", "Award"),
                    Relation("discovered", "Person", "Discovery"),
                ],
            ),
            RelationGroup(
                name="relation_to_others",
                description="Personal relationships between people",
                relations=[
                    Relation("married_to", "Person", "Person"),
                    Relation("collaborated_with", "Person", "Person"),
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
