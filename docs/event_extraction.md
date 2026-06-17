# Event Extraction

> Status: feature module (`drg.events`).
> Opt-in; the legacy entity + relation pipeline is unchanged.

## Why events?

A flat triple model (`Apple --acquired--> Beats`) cannot express n-ary
facts cleanly. "Apple acquired Beats in May 2014 for $3 billion" has
four dimensions — acquirer, acquired, time, value — and shoving them
into separate triples loses the connection between them.

`drg.events` introduces events as **first-class graph objects** with
typed roles, ISO 8601 timestamps, locations, properties, and full
provenance. Events live alongside the existing entity / relation
extraction; nothing in the legacy pipeline changes.

## Concepts

- **`Event`** — a single n-ary fact (an "occurrence"). Has a stable
  id, an event type (e.g. `"Acquisition"`), a dict of typed
  participants, an optional timestamp, an optional location, free-form
  properties, and an `EventProvenance`.
- **`EventTypeDefinition`** — declarative schema for one event type
  (its roles + property bag). Pure data, no code.
- **`EventTypeRegistry`** — ordered, name-indexed catalogue of event
  types. The default is **empty**; provide your own or use the bundled
  `example_event_registry()`.
- **`EventProvenance`** — `document_id`, `chunk_ids`, `text_spans`,
  `extracted_at`, `extractor_version`, `extraction_method`,
  `confidence`. Every event carries one.

## Quick start

```python
from drg.events import (
    EventRole,
    EventTypeDefinition,
    EventTypeRegistry,
    extract_events,
)
from drg.extract import extract_typed
from drg.graph.builders import build_enhanced_kg
from drg.schema import EnhancedDRGSchema, EntityType, Relation, RelationGroup

# 1. Standard entity / relation pipeline (unchanged).
text = "Apple announced in May 2014 that it acquired Beats Electronics for $3 billion."
schema = EnhancedDRGSchema(
    entity_types=[
        EntityType(name="Company", description="Business organization"),
    ],
    relation_groups=[
        RelationGroup(
            name="business",
            description="Business relationships",
            relations=[Relation("acquired", "Company", "Company", description="...")],
        ),
    ],
)
entities, triples = extract_typed(text, schema)

# 2. Define an event-type registry. Use `example_event_registry()` for
#    a curated set of common business / political event types.
registry = EventTypeRegistry()
registry.register(
    EventTypeDefinition(
        name="Acquisition",
        description="One organization acquires another.",
        roles=[
            EventRole(name="acquirer", entity_types=("Company",), required=True),
            EventRole(name="acquired", entity_types=("Company",), required=True),
        ],
        properties={"deal_value": "monetary amount"},
    )
)

# 3. Extract events.
events = extract_events(text, entities, registry, document_id="doc1")

# 4. Build an EnhancedKG that contains BOTH the legacy graph AND the events.
kg = build_enhanced_kg(
    entities_typed=entities,
    triples=triples,
    schema=schema,
    source_text=text,
    events=events,
    document_id="doc1",
)

print(kg.to_json())
```

## Storage model

An `Event` is projected onto the existing `KGNode` / `KGEdge` storage:

- One `KGNode` per event with `type="Event:<event_type>"`. The full
  event payload (participants, timestamp, location, provenance) lives
  in `metadata`. The node prefix lets visualizers / exporters treat
  events distinctly without inventing a new node class.
- One `KGEdge` per `(role, participant)` pair with
  `relationship_type="role:<role_name>"` and
  `source=<event_id>, target=<entity_id>`. Role edges carry the event
  timestamp on `start_time` / `end_time` so consumers can filter
  participants by event date without joining tables.
- An optional `occurred_at` edge linking the event node to its
  location (when known).

Because events use the same storage primitives as everything else,
**`GraphMerger`, `Neo4jExporter`, the visualization adapters, and the
reasoning engine work on event-augmented graphs without modification**.

## Serialization

`EnhancedKG.to_json()`, `to_json_ld()`, and `to_enriched_format()`
emit an additional top-level `events` key **only when events are
present**. Legacy graphs serialize byte-identically.

```json
{
  "nodes": [...],
  "edges": [...],
  "clusters": [],
  "events": [
    {
      "id": "event:Acquisition:abc123def4",
      "event_type": "Acquisition",
      "participants": {"acquirer": ["Apple"], "acquired": ["Beats"]},
      "timestamp": {"start": "2014-05", "end": "2014-05", "precision": "month"},
      "properties": {"deal_value": "$3B"},
      "provenance": {
        "document_id": "doc1",
        "text_spans": [{"text": "Apple ... acquired Beats Electronics for $3 billion."}],
        "extracted_at": "2026-06-07T12:00:00+00:00",
        "extractor_version": "0.1.0a1",
        "extraction_method": "llm",
        "confidence": 0.92
      }
    }
  ]
}
```

## Extending the registry

Adding a new event type is pure data — no code edits anywhere:

```python
from drg.events import EventRole, EventTypeDefinition, EventTypeRegistry

reg = EventTypeRegistry()
reg.register(EventTypeDefinition(
    name="ResearchPublication",
    description="A paper is published by one or more authors at a venue.",
    roles=[
        EventRole(name="authors", cardinality="many",
                  entity_types=("Person",), required=True),
        EventRole(name="venue",
                  entity_types=("Conference", "Journal"), required=False),
    ],
    properties={"title": "string", "doi": "string"},
))
reg.save_json("my_event_registry.json")
```

…then load it from the CLI with `--extract-events
--events-registry my_event_registry.json`.

## CLI

```bash
# Use the bundled example registry (10 common event types).
drg extract input.txt -o out.kg.json \
  --auto-schema \
  --extract-events --events-use-example

# Use your own registry.
drg extract input.txt -o out.kg.json \
  --auto-schema \
  --extract-events --events-registry my_event_registry.json
```

The CLI is opt-in: without `--extract-events` nothing changes.

## Provenance contract

Every event preserves:

| Field | Source |
|-------|--------|
| `provenance.document_id` | `extract_events(..., document_id=...)` |
| `provenance.chunk_ids` | `extract_events(..., chunk_id=...)` (one or many depending on caller) |
| `provenance.text_spans` | LLM-emitted `evidence_text`, falling back to a deterministic windowed search around participant mentions |
| `provenance.extracted_at` | UTC ISO 8601 timestamp at extraction time |
| `provenance.extractor_version` | `drg.__version__` |
| `provenance.extraction_method` | `"llm" \| "rule" \| "merged" \| "manual"` |
| `provenance.confidence` | LLM self-rating clamped to `[0, 1]` if provided, else a coverage-based heuristic |

## Backward compatibility

- `extract_typed` / `extract_from_chunks` outputs are byte-identical.
- `EnhancedKG` JSON shape is byte-identical for graphs without events.
- All 662 pre-existing tests pass unmodified.

## Limitations

- LLM-driven; needs a configured DSPy LM. Mock-mode short-circuits to
  `[]` (same convention as the entity / relation pipeline).
- Cross-chunk event reconciliation is not yet implemented — events
  extracted from different chunks can produce duplicates if their
  surface forms differ. The deterministic event id mitigates exact
  duplicates; semantic dedup is a future iteration.
