# Temporal Knowledge Graph

DRG-KG represents facts that change over time using optional temporal metadata on
entities, relationships, and events. Existing graphs without temporal fields
continue to work unchanged.

## Model

### TemporalScope

The canonical temporal block (`drg.temporal.TemporalScope`) carries:

| Field | Meaning |
|-------|---------|
| `valid_from` | When the fact became true (partial ISO date) |
| `valid_to` | When the fact stopped being true (`null` = still active) |
| `created_at` | Extraction / ingestion timestamp |
| `updated_at` | Last merge or update timestamp |
| `precision_from` / `precision_to` | `year`, `month`, `day`, or `instant` |
| `raw_text` | Surface form from source text |

Partial dates are stored as-is:

- `2014`
- `2014-06`
- `2014-06-15`

### Relationships (`KGEdge`)

Edges expose both legacy and semantic names:

- `start_time` ↔ `valid_from`
- `end_time` ↔ `valid_to`
- `created_at`, `updated_at` (optional)

Temporal metadata is also mirrored in `metadata.temporal` for Neo4j and JSON consumers.

### Entities (`KGNode`)

Entity validity windows live in `metadata.temporal` (or top-level `temporal` on
serialization). Use `node.apply_temporal_scope()` / `node.get_temporal_scope()`.

### Events

Events already use `EventTimestamp` with `start`, `end`, and `precision`. Event
role edges copy timestamps to `start_time` / `end_time`.

## Example: Apple CEOs

```python
from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode
from drg.query import GraphQuery

kg = EnhancedKG()
for name, etype in [("Apple", "Company"), ("Steve Jobs", "Person"), ("Tim Cook", "Person")]:
    kg.add_node(KGNode(id=name, type=etype))

kg.add_edge(KGEdge(
    source="Steve Jobs", target="Apple",
    relationship_type="CEO_OF",
    relationship_detail="Steve Jobs was CEO of Apple",
    start_time="1997", end_time="2011",
))
kg.add_edge(KGEdge(
    source="Tim Cook", target="Apple",
    relationship_type="CEO_OF",
    relationship_detail="Tim Cook is CEO of Apple",
    start_time="2011", end_time=None,
))

gq = GraphQuery(kg)
print(gq.role_holders_at("Apple", "CEO_OF", "2008")[0].source)  # Steve Jobs
print(gq.role_holders_at("Apple", "CEO_OF", "2015")[0].source)  # Tim Cook
print(gq.temporal_query("Apple CEO in 2008")[0].source)         # Steve Jobs
```

## Temporal queries

`GraphQuery` adds:

| Method | Question answered |
|--------|-------------------|
| `role_holders_at(target, rel, as_of)` | Who held role X at date Y? |
| `temporal_query(text)` | Compact natural temporal lookup such as `Apple CEO in 2008` |
| `relations_active_at(as_of, ...)` | What relationships were active? |
| `temporal_timeline(...)` | Chronological view of facts |
| `changes_between(from, to)` | What started or ended? |
| `temporal_overlaps()` | Overlapping duplicate edges |
| `temporal_conflicts(...)` | Concurrent role holders |
| `entity_transitions(entity, rel)` | How one entity's role evolved |

## Temporal reasoning

```python
from drg.temporal import detect_conflicts, build_timeline, is_active_at, TemporalScope

scope = TemporalScope(valid_from="2011")
assert is_active_at(scope, "2011-06")

timeline = gq.temporal_timeline(target="Apple", relationship_type="CEO_OF")
conflicts = gq.temporal_conflicts(relationship_type="CEO_OF", target="Apple")
```

## Extraction pipeline

`extract_typed(..., return_enriched=True)` attaches per-relation `temporal` metadata.
`build_enhanced_kg(..., enriched_relations=...)` now persists this to edge
`start_time` / `end_time` and `metadata.temporal`.

Year heuristics return partial dates (`"2014"`, not `"2014-01-01"`).

## Provenance

Temporal facts retain existing provenance:

- `metadata.source_ref` / `metadata.source_documents`
- `metadata.evidence` (text snippets)
- `confidence` on nodes and edges
- `EventProvenance` for events (`extracted_at`, `document_id`, …)

## Migration strategy

Existing graphs need no changes. To normalise legacy edges:

```python
from drg.graph.kg_core import EnhancedKG
from drg.temporal import migrate_edge_dict, migrate_node_dict

kg = EnhancedKG.load_json("old_graph.json")
data = kg.to_dict()  # parse JSON if needed
data["edges"] = [migrate_edge_dict(e) for e in data["edges"]]
data["nodes"] = [migrate_node_dict(n) for n in data["nodes"]]
```

Rules:

1. `start_time` → `valid_from` (alias emitted on save)
2. `end_time` → `valid_to`
3. Nested `metadata.temporal` block created when missing
4. Atemporal edges remain atemporal (`is_active_at` returns `True`)

## Neo4j / future backends

`QueryBackend` is backend-agnostic. Temporal filters operate on `KGEdge` fields
today; a Neo4j backend can push `valid_from` / `valid_to` into Cypher range
predicates using the same `TemporalScope` bounds helpers.
