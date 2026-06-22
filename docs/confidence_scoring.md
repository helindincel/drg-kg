# Confidence Scoring Framework

This document describes the confidence-scoring framework introduced in
`drg/confidence/` and its integration with the rest of the knowledge-graph
pipeline.

## Why confidence?

A knowledge graph extracted from unstructured text is inherently uncertain.
Users need to know **how much to trust each piece of information** in order
to:

- Filter low-confidence assertions before using them downstream.
- Rank graph-query and audit results.
- Audit the pipeline (e.g. spot systematically weak relations).
- Calibrate the extractor over time.

DRG already had `confidence` on the `KGEdge` and `EnrichedRelationship`
types but no equivalent on `KGNode` and no end-to-end scoring strategy.
This framework fills that gap with a small, modular layer.

## High-level architecture

```
            ┌────────────────────┐        ┌──────────────────────┐
text ─────► │   drg.extract.*    │ ─────► │   build_enhanced_kg  │ ─────► EnhancedKG
            │  (KGExtractor,     │        │   (drg.graph.builders)│
            │   extract_typed)   │        └──────────┬───────────┘
            └────────────────────┘                   │
                                                     │ uses
                                                     ▼
                                          ┌──────────────────────┐
                                          │  drg.confidence.*    │
                                          │  (ConfidenceStrategy │
                                          │   + DefaultStrategy) │
                                          └──────────────────────┘
```

- **Data model layer** (`drg/graph/kg_core.py`): `KGNode.confidence` and
  `KGEdge.confidence` are first-class `float | None` fields. They are the
  single source of truth and survive every serialisation path
  (`to_dict`, `to_json`, `to_json_ld`, `to_enriched_format`, Neo4j).
- **Strategy layer** (`drg/confidence/`): pluggable scorers that turn
  extraction signals into per-entity / per-relation scores. The default
  strategy is a deterministic, schema-aware heuristic placeholder. Custom
  strategies (LLM self-rating, ensemble disagreement, calibrated probs)
  drop in at the same seam.
- **Builder layer** (`drg/graph/builders.py`): orchestrates the strategy
  and stamps the resulting scores onto `KGNode`/`KGEdge` instances. This
  is the single entry point that ties extraction signals, the strategy,
  and the graph data model together.

## Quickstart

### Default behaviour (auto-scoring)

`build_enhanced_kg` enables the default strategy automatically — every
node and edge gets a confidence score in `[0.0, 1.0]`:

```python
from drg.extract import extract_typed
from drg.graph.builders import build_enhanced_kg

entities, triples, enriched = extract_typed(
    text=my_text,
    schema=my_schema,
    return_enriched=True,
)

kg = build_enhanced_kg(
    entities_typed=entities,
    triples=triples,
    schema=my_schema,
    source_text=my_text,
    enriched_relations=enriched,  # so upstream temporal/negation cues feed scoring
)

for node in kg.nodes.values():
    print(node.id, node.confidence)
for edge in kg.edges:
    print(edge.source, edge.relationship_type, edge.target, edge.confidence)
```

### Opt out (legacy mode)

```python
kg = build_enhanced_kg(
    entities_typed=entities,
    triples=triples,
    confidence_strategy=None,  # do not compute or attach confidence
)
```

### Custom strategy

```python
from drg.confidence import ConfidenceScore, ConfidenceStrategy

class MyStrategy(ConfidenceStrategy):
    name = "my-llm-critic"

    def score_entities(self, entities, *, context=None):
        return {name: ConfidenceScore(value=..., method=self.name) for name, _ in entities}

    def score_relations(self, relations, *, enriched_relations=None, context=None):
        return {triple: ConfidenceScore(value=..., method=self.name) for triple in relations}

kg = build_enhanced_kg(
    entities_typed=entities,
    triples=triples,
    confidence_strategy=MyStrategy(),
)
```

### Explicit overrides

When you already computed scores upstream (e.g. ensembled across multiple
extractors), bypass the strategy:

```python
kg = build_enhanced_kg(
    entities_typed=entities,
    triples=triples,
    entity_confidences={"Apple": 0.93, "iPhone": 0.88},
    relation_confidences={("Apple", "produces", "iPhone"): 0.91},
)
```

Overrides take precedence over strategy output for matching entries; the
strategy still scores anything not explicitly overridden.

## Default strategy specification

`drg.confidence.DefaultConfidenceStrategy` is a deterministic placeholder
that produces sensible scores without requiring labelled data. It is **not
calibrated** — the coefficients are starting points. The class attributes
exposing the coefficients (e.g. `BASE_ENTITY_SCORE`, `BOOST_SCHEMA_VALID`)
are subclass-friendly for easy tuning.

### Entity scoring

| Signal             | Contribution | Notes                                              |
|--------------------|--------------|----------------------------------------------------|
| Base               | +0.60        | LLM-extracted but otherwise unverified             |
| `type_in_schema`   | +0.15        | Entity type appears in the supplied schema         |
| `name_in_text`     | +0.10        | Whole-word match against the source text           |
| `multi_word`       | +0.05        | Entity name has 2+ tokens (specificity heuristic)  |
| Floor              | clamp to 0.05 | Avoid zeroing out legitimate entities             |
| Ceiling            | clamp to 1.00 |                                                   |

### Relation scoring

| Signal              | Contribution  | Notes                                               |
|---------------------|---------------|-----------------------------------------------------|
| Upstream `confidence`| pass-through | If `enriched_relations[i].confidence` is numeric, the strategy honours it verbatim (forward-compatibility for LLM self-rating) |
| Base                | +0.50         | Default for LLM-extracted relations                 |
| `both_typed`        | +0.10         | Both endpoints have known entity types              |
| `schema_valid`      | +0.20         | Schema allows `(s_type, relation, o_type)`          |
| `temporal`          | +0.05         | Deterministic temporal cue was attached             |
| `is_negated`        | −0.30         | Negation penalty (kept low instead of dropped)      |
| Floor / ceiling     | clamp to [0.05, 1.00] |                                            |

## Where confidence "lives" in the data model

| Object                    | Field          | Where it persists                                            |
|---------------------------|----------------|--------------------------------------------------------------|
| `KGNode`                  | `confidence`   | `to_dict`, `to_json`, `to_json_ld`, `to_enriched_format`, Neo4j |
| `KGEdge`                  | `confidence`   | (existing) all the same paths                                |
| `EnrichedRelationship`    | `confidence`   | (existing) `to_dict`, `to_enriched_format`                   |
| `ConfidenceScore`         | `value` + `signals` | Strategy-internal; only the scalar reaches the data model |

The breakdown (`signals`, `method`) on `ConfidenceScore` is not persisted
on the node/edge by default — it stays inside the strategy layer for
auditing. Persist it explicitly via `node.metadata["confidence_breakdown"]
= sc.to_dict()` if you need it on disk.

## Backward compatibility

- `KGNode.confidence` defaults to `None` — bare nodes serialise byte-for-byte
  the same as before.
- `build_enhanced_kg` keeps its existing positional/keyword surface intact.
  All new parameters (`enriched_relations`, `confidence_strategy`,
  `entity_confidences`, `relation_confidences`) are optional and default to
  values that preserve legacy behaviour OR the new safe default
  (`confidence_strategy="default"` for auto-scoring).
- `confidence_strategy=None` opts out completely — useful for tests and
  any caller that wants the pre-feature output.
- The Neo4j exporter keeps existing `weight` semantics; it now additionally
  surfaces `confidence` as a first-class property when present.

## Future improvements

The default strategy is intentionally a placeholder. Promising upgrades:

1. **LLM self-rating**: ask the extractor to emit a confidence score
   alongside each entity/relation. The strategy already honours upstream
   numeric `confidence` in `enriched_relations`, so this slots in without
   API changes.
2. **Ensemble disagreement**: run extraction multiple times (different
   prompts / temperatures / models) and use agreement as a confidence
   signal. Implementable as a new `ConfidenceStrategy` that consumes a
   list of extraction outputs through `context`.
3. **Calibrated probabilities**: collect labelled data and fit a calibration
   curve (Platt scaling, isotonic regression) on top of the current
   heuristic signals. The signal breakdown on `ConfidenceScore` already
   exposes the inputs needed.
4. **Embedding-based plausibility**: compare an entity's embedding to its
   declared type centroid; near-prototype entities score higher.
5. **Source/provenance weighting**: down-weight assertions extracted from
   noisy chunks (low chunk quality, OCR confidence, etc.) when chunk-level
   scores are available. The `context` dict already lets us pipe this
   through.
6. **Per-relation-type baselines**: the current base score is the same
   for every relation. Different relation types have different precision
   floors empirically — a small lookup table would help.
7. **Confidence propagation in clustering**: pass node/edge confidence into
   `CommunityReportGenerator` to weight top-actor / top-relation
   selection. The community-report module already iterates edges, so
   this is a small change.
8. **Persist the `signals` breakdown** in node/edge metadata when the
   pipeline is run in `--debug`/audit mode.

## Testing

Tests for the framework live in `tests/test_confidence.py`. They cover:

- `clamp_confidence` / `ConfidenceScore` value semantics.
- `DefaultConfidenceStrategy` — entity & relation scoring under all signal
  combinations.
- `KGNode.confidence` validation and JSON / JSON-LD / enriched-format
  round-trips.
- `build_enhanced_kg` end-to-end integration: default mode, opt-out,
  explicit overrides, custom strategies, legacy `DRGSchema` compatibility.
- Backward compatibility: bare nodes/edges still serialise without
  `confidence` keys.

Run them with:

```bash
pytest tests/test_confidence.py -v
```
