#!/usr/bin/env python3
"""
Quickstart 2 — Financial / corporate news -> Knowledge Graph
=============================================================

Showcases DRG on business-domain text: corporate actions (acquisitions,
investments, product launches) condensed into a KG of Companies, People,
Products, and Money flows.

What this script does
---------------------
1. Defines an `EnhancedDRGSchema` aimed at M&A / corporate-news text.
2. Runs `extract_typed()` on a short multi-fact paragraph.
3. Persists the resulting KG as JSON next to this file.

Why this matters
----------------
Most market-intelligence / news-aggregation work needs **structured triples**
out of free-form prose (acquirer -> acquired, investor -> startup, company ->
product). Define the schema once, point DRG at the next earnings report or
press release, and the same code keeps working.

Prerequisites
-------------
- An LLM provider key in the environment (default: `OPENAI_API_KEY`).

Run
---
    python examples/quickstarts/02_financial_news.py
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
Microsoft announced on Tuesday that it had completed its acquisition of
Activision Blizzard for $68.7 billion, the largest deal in the gaming
industry's history. Phil Spencer, head of Microsoft Gaming, said the
combined company will continue to operate the Call of Duty franchise as a
multi-platform release. Separately, Sequoia Capital led a $40 million
Series B round in the Berlin-based AI startup Niko Labs, with participation
from Index Ventures.
""".strip()


def build_schema() -> EnhancedDRGSchema:
    """Schema for corporate / financial news extraction."""
    return EnhancedDRGSchema(
        entity_types=[
            EntityType(
                name="Company",
                description="A corporation, startup, or business entity",
                examples=["Microsoft", "Activision Blizzard", "Niko Labs"],
            ),
            EntityType(
                name="Investor",
                description="A venture-capital firm or institutional investor",
                examples=["Sequoia Capital", "Index Ventures"],
            ),
            EntityType(
                name="Person",
                description="A named executive or individual",
                examples=["Phil Spencer"],
            ),
            EntityType(
                name="Product",
                description="A named product, franchise, or service",
                examples=["Call of Duty"],
            ),
            EntityType(
                name="MoneyAmount",
                description="A monetary figure tied to a deal or round",
                examples=["$68.7 billion", "$40 million"],
            ),
            EntityType(
                name="Place",
                description="A geographic location (city, country, region)",
                examples=["Berlin"],
            ),
        ],
        relation_groups=[
            RelationGroup(
                name="corporate_actions",
                description="M&A and ownership relations",
                relations=[
                    Relation("acquired", "Company", "Company"),
                    Relation("paid_amount", "Company", "MoneyAmount"),
                    Relation("operates", "Company", "Product"),
                ],
            ),
            RelationGroup(
                name="investment",
                description="Funding-round style relationships",
                relations=[
                    Relation("led_round_in", "Investor", "Company"),
                    Relation("participated_in", "Investor", "Company"),
                    Relation("raised_amount", "Company", "MoneyAmount"),
                ],
            ),
            RelationGroup(
                name="people",
                description="People and their corporate affiliations",
                relations=[
                    Relation("executive_of", "Person", "Company"),
                    Relation("headquartered_in", "Company", "Place"),
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
