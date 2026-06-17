#!/usr/bin/env python3
"""End-to-end demo of event extraction on real-world snippets.

Runs the standard entity + relation pipeline first, then the event
extractor (using the bundled example registry), and prints both views.

Without an LM configured the example short-circuits to empty events
(same convention as the entity / relation pipeline) — the structural
parts of the demo (post-processing, graph mapping, JSON output)
still run via the manual fallback at the bottom of the file so the
output is meaningful even in a pure-offline run.

Usage::

    python examples/event_extraction_example.py
    DRG_MODEL=openai/gpt-4o-mini OPENAI_API_KEY=... \\
        python examples/event_extraction_example.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from drg.events import (
    Event,
    EventProvenance,
    EventTimestamp,
    EventTypeRegistry,
    TextSpan,
    example_event_registry,
    extract_events,
    make_event_id,
)
from drg.extract import extract_typed
from drg.graph.builders import build_enhanced_kg
from drg.schema import (
    EnhancedDRGSchema,
    EntityType,
    Relation,
    RelationGroup,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


DOCS = [
    {
        "id": "apple-beats-2014",
        "text": (
            "On May 28, 2014, Apple Inc. announced that it would acquire "
            "Beats Electronics, the headphone and music-streaming company "
            "co-founded by Dr. Dre and Jimmy Iovine, for $3 billion. The "
            "deal closed in August 2014 and was Apple's largest "
            "acquisition to date."
        ),
    },
    {
        "id": "anthropic-amazon-2023",
        "text": (
            "In September 2023, Amazon announced a strategic investment "
            "of up to $4 billion in Anthropic, the AI safety startup "
            "behind the Claude family of models."
        ),
    },
    {
        "id": "apple-ceo-2011",
        "text": (
            "In August 2011, Steve Jobs resigned as CEO of Apple. "
            "He was succeeded by Tim Cook, who had previously served "
            "as Chief Operating Officer."
        ),
    },
    {
        "id": "epic-apple-2020",
        "text": (
            "Epic Games filed a lawsuit against Apple in August 2020 in "
            "the U.S. District Court for the Northern District of "
            "California, challenging App Store policies."
        ),
    },
]


def _make_schema() -> EnhancedDRGSchema:
    return EnhancedDRGSchema(
        entity_types=[
            EntityType(name="Company", description="A business organization"),
            EntityType(name="Organization", description="Any organization"),
            EntityType(name="Person", description="A human individual"),
            EntityType(name="Product", description="A product or service"),
            EntityType(name="Location", description="Geographic location"),
        ],
        relation_groups=[
            RelationGroup(
                name="business",
                description="Business relationships",
                relations=[
                    Relation(
                        name="acquired",
                        src="Company",
                        dst="Company",
                        description="Company acquires another company",
                    ),
                    Relation(
                        name="invested_in",
                        src="Company",
                        dst="Company",
                        description="Company invests in another",
                    ),
                    Relation(
                        name="filed_lawsuit_against",
                        src="Organization",
                        dst="Organization",
                        description="Organization files lawsuit",
                    ),
                ],
            ),
            RelationGroup(
                name="people",
                description="People-organization relationships",
                relations=[
                    Relation(
                        name="ceo_of",
                        src="Person",
                        dst="Company",
                        description="Person is CEO of company",
                    ),
                    Relation(
                        name="works_at",
                        src="Person",
                        dst="Company",
                        description="Person works at company",
                    ),
                ],
            ),
        ],
    )


def _print_events(doc_id: str, events: list[Event]) -> None:
    print(f"\n--- Events from {doc_id} ({len(events)} extracted) ---")
    for ev in events:
        print(f"  [{ev.event_type}]  id={ev.id}")
        for role, members in ev.participants.items():
            print(f"    {role}: {', '.join(members)}")
        if ev.timestamp is not None and not ev.timestamp.is_empty():
            print(f"    when: {ev.timestamp.start} (precision={ev.timestamp.precision})")
        if ev.location:
            print(f"    where: {ev.location}")
        if ev.properties:
            print(f"    props: {ev.properties}")
        if ev.provenance.text_spans:
            snippet = ev.provenance.text_spans[0].text
            if len(snippet) > 90:
                snippet = snippet[:87] + "..."
            print(f"    evidence: {snippet}")
        print(f"    confidence: {ev.confidence:.2f}")


def _manual_fallback_events(doc_id: str, text: str) -> list[Event]:
    """Hand-crafted events used when no LM is configured.

    Lets the example produce a non-empty graph for offline demos
    without pretending the extractor ran.
    """
    if doc_id == "apple-beats-2014":
        return [
            Event(
                id=make_event_id(
                    "Acquisition",
                    {"acquirer": ["Apple"], "acquired": ["Beats"]},
                    EventTimestamp(start="2014-05-28", precision="day"),
                    text,
                ),
                event_type="Acquisition",
                participants={"acquirer": ["Apple"], "acquired": ["Beats"]},
                timestamp=EventTimestamp(
                    start="2014-05-28",
                    end="2014-08-31",
                    precision="day",
                    raw_text="May 28, 2014 – August 2014",
                ),
                properties={"deal_value": "$3B", "currency": "USD"},
                provenance=EventProvenance(
                    document_id=doc_id,
                    text_spans=[TextSpan(text=text)],
                    extraction_method="manual",
                    confidence=0.95,
                ),
            )
        ]
    if doc_id == "anthropic-amazon-2023":
        return [
            Event(
                id=make_event_id(
                    "Investment",
                    {"investor": ["Amazon"], "target": ["Anthropic"]},
                    EventTimestamp(start="2023-09", precision="month"),
                    text,
                ),
                event_type="Investment",
                participants={"investor": ["Amazon"], "target": ["Anthropic"]},
                timestamp=EventTimestamp(start="2023-09", precision="month"),
                properties={"amount": "$4B", "currency": "USD"},
                provenance=EventProvenance(
                    document_id=doc_id,
                    text_spans=[TextSpan(text=text)],
                    extraction_method="manual",
                    confidence=0.93,
                ),
            )
        ]
    if doc_id == "apple-ceo-2011":
        return [
            Event(
                id=make_event_id(
                    "LeadershipChange",
                    {
                        "organization": ["Apple"],
                        "predecessor": ["Steve Jobs"],
                        "successor": ["Tim Cook"],
                    },
                    EventTimestamp(start="2011-08", precision="month"),
                    text,
                ),
                event_type="LeadershipChange",
                participants={
                    "organization": ["Apple"],
                    "predecessor": ["Steve Jobs"],
                    "successor": ["Tim Cook"],
                },
                timestamp=EventTimestamp(start="2011-08", precision="month"),
                properties={"role_title": "CEO"},
                provenance=EventProvenance(
                    document_id=doc_id,
                    text_spans=[TextSpan(text=text)],
                    extraction_method="manual",
                    confidence=0.92,
                ),
            )
        ]
    if doc_id == "epic-apple-2020":
        return [
            Event(
                id=make_event_id(
                    "Lawsuit",
                    {"plaintiff": ["Epic Games"], "defendant": ["Apple"]},
                    EventTimestamp(start="2020-08", precision="month"),
                    text,
                ),
                event_type="Lawsuit",
                participants={
                    "plaintiff": ["Epic Games"],
                    "defendant": ["Apple"],
                },
                timestamp=EventTimestamp(start="2020-08", precision="month"),
                properties={"cause_of_action": "App Store policies"},
                provenance=EventProvenance(
                    document_id=doc_id,
                    text_spans=[TextSpan(text=text)],
                    extraction_method="manual",
                    confidence=0.90,
                ),
            )
        ]
    return []


def _entities_for(doc_id: str) -> list[tuple[str, str]]:
    """Stable entity sets for the offline fallback path."""
    table = {
        "apple-beats-2014": [("Apple", "Company"), ("Beats", "Company")],
        "anthropic-amazon-2023": [
            ("Amazon", "Company"),
            ("Anthropic", "Company"),
        ],
        "apple-ceo-2011": [
            ("Apple", "Company"),
            ("Steve Jobs", "Person"),
            ("Tim Cook", "Person"),
        ],
        "epic-apple-2020": [
            ("Apple", "Company"),
            ("Epic Games", "Company"),
        ],
    }
    return table.get(doc_id, [])


def main() -> None:
    schema = _make_schema()
    registry: EventTypeRegistry = example_event_registry()

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_events_payload: list[dict] = []

    for doc in DOCS:
        doc_id = doc["id"]
        text = doc["text"]
        print(f"\n=== {doc_id} ===")
        print(text)

        try:
            entities, triples = extract_typed(text, schema)
        except Exception as exc:
            print(f"(entity/relation extraction skipped: {exc})")
            entities, triples = [], []

        if not entities:
            entities = _entities_for(doc_id)

        print(f"Entities ({len(entities)}): {entities}")
        print(f"Triples ({len(triples)}): {triples}")

        try:
            events = extract_events(
                text=text,
                entities_typed=entities,
                registry=registry,
                document_id=doc_id,
            )
        except Exception as exc:
            print(f"(event extraction skipped: {exc})")
            events = []

        if not events:
            events = _manual_fallback_events(doc_id, text)

        _print_events(doc_id, events)

        kg = build_enhanced_kg(
            entities_typed=entities,
            triples=triples,
            schema=schema,
            source_text=text,
            events=events,
            document_id=doc_id,
        )
        out_path = output_dir / f"{doc_id}.kg.json"
        kg.save_json(str(out_path))
        print(f"Saved: {out_path}")

        all_events_payload.extend(ev.to_dict() for ev in events)

    summary_path = output_dir / "events_summary.json"
    summary_path.write_text(
        json.dumps({"events": all_events_payload}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nAll events: {summary_path} ({len(all_events_payload)} total)")


if __name__ == "__main__":
    main()
