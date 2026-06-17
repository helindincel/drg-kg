# Query & Reasoning Layer

> **Status:** Stable, opt-in. Lives in `drg.query`. Pure stdlib +
> `drg.graph` — **no LLM calls**, **no fabricated answers**.
>
> **Goal:** Turn DRG from a KG generation system into a KG exploration
> platform where every answer is traceable back to graph nodes, edges,
> events, or source documents.

---

## 1. Architecture

```
EnhancedKG  ──► InMemoryBackend (indexed adjacency)
                      ▲
                      │ QueryBackend Protocol
                      ▼
                GraphQuery (facade)
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   entity lookup   traversal    explanation
   relations       path find    evidence
   search          community    events
```

### Separation of concerns

| Layer | Responsibility |
|---|---|
| `drg.extract` + `drg.graph.builders` | **Generate** the graph from text |
| `drg.reasoning` | **Infer** new edges with provenance (post-merge) |
| `drg.query` | **Query** the graph — read-only, evidence-first |
| `drg.graph.query_engine` | Legacy UI search (unchanged; wrapped by `GraphQuery.search`) |

Graph generation workflows are **not modified**. Querying is a separate
opt-in import.

### Backend abstraction

`QueryBackend` is a `typing.Protocol`. The default implementation is
`InMemoryBackend`, which indexes an in-memory `EnhancedKG`.

A future `Neo4jBackend` can implement the same protocol:

```python
class Neo4jBackend:
    def get_node(self, node_id: str) -> KGNode | None: ...
    def edges_incident(self, node_id: str, ...) -> list[KGEdge]: ...
    # ... remaining protocol methods
```

`GraphQuery` accepts any backend:

```python
GraphQuery(kg, backend=Neo4jBackend(driver))
```

---

## 2. Quick start

```python
from drg.query import GraphQuery

# From an in-memory EnhancedKG
gq = GraphQuery(kg)

# Or from a persisted JSON file
gq = GraphQuery.from_json("outputs/global_kg.json")

# Or via the convenience method on EnhancedKG
gq = kg.query()

# Entity lookup
apple = gq.entity("Apple")
matches = gq.find_entities("openai", entity_type="Company")

# Relationships
edges = gq.relations(source="Apple", relationship_type="acquired")
bundle = gq.evidence_for("Apple", "ACQUIRED", "Beats")

# Multi-hop neighborhood
hood = gq.neighbors("OpenAI", hops=2)

# Path finding
paths = gq.find_paths("Apple", "Jimmy Iovine", max_hops=3)
path = gq.shortest_path("Entity A", "Entity B")

# Explainable reasoning
exp = gq.explain("Apple", "Jimmy Iovine")
print(exp.summary)
for item in exp.evidence:
    print(item.triple, item.source_ref)

# Community & related entities
comm = gq.community_of("Apple")
related = gq.related_entities("Apple", entity_type="Company", mode="shortest_path")

# Free-text search (deterministic, no LLM)
answer = gq.search("companies related to Apple")
```

Runnable demo: `python examples/query_layer_example.py`

Hybrid graph + vector retrieval is available through `gq.hybrid_search(...)`.
See [`docs/hybrid_retrieval.md`](hybrid_retrieval.md) and
`python examples/hybrid_retrieval_example.py`.

---

## 3. API catalog

### `GraphQuery`

| Method | Description |
|---|---|
| `entity(id)` | Exact entity lookup → `EntityView` |
| `find_entities(query, entity_type=, limit=)` | Fuzzy ranked search → `list[EntityMatch]` |
| `relations(source=, target=, relationship_type=, include_inferred=)` | Filter edges → `list[EdgeView]` |
| `evidence_for(source, rel, target)` | All evidence for a triple → `EvidenceBundle` |
| `neighbors(id, hops=, direction=, relationship_type=)` | Multi-hop BFS → `NeighborhoodView` |
| `find_paths(src, dst, max_hops=, max_paths=)` | Simple path enumeration → `list[GraphPath]` |
| `shortest_path(src, dst, ...)` | Shortest path → `GraphPath \| None` |
| `explain(src, dst, max_hops=)` | Why are A and B connected? → `Explanation` |
| `events_for(id, event_types=)` | Incident event nodes → `list[EventView]` |
| `community_of(id)` | Cluster membership → `CommunityView \| None` |
| `community_neighbors(id)` | Same-cluster entities → `list[str]` |
| `related_entities(id, mode=, entity_type=, hops=)` | Ranked related entities → `list[RelatedEntityMatch]` |
| `search(query)` / `query(query)` | Free-text deterministic search → `QueryAnswer` |

### `related_entities` modes

| Mode | Ranking criterion |
|---|---|
| `shared_neighbors` (default) | Count of shared 1-hop neighbors |
| `shortest_path` | Inverse path length × path confidence |
| `same_community` | Entities in the same cluster |
| `degree` | Node degree among candidates |

---

## 4. Provenance schema

Every result type exposes `.to_dict()` for JSON export. Evidence-bearing
results include:

```json
{
  "triple": ["Apple", "ACQUIRED", "Beats"],
  "source_ref": "doc_A_apple_news",
  "snippet": "Apple acquired Beats Electronics.",
  "confidence": 0.85,
  "is_inferred": false
}
```

Inferred edges additionally carry:

```json
{
  "is_inferred": true,
  "inference": {
    "rule": "path_bridge",
    "evidence_chain": [...],
    "source_documents": ["doc_A", "doc_B"],
    "explanation": "Apple and Jimmy Iovine are both connected to Beats. ..."
  }
}
```

`Explanation.summary` is **template-based** (no LLM). When no path exists
within the hop limit, the layer reports that fact — it never fabricates
a connection.

---

## 5. Example queries

| Natural language | API call |
|---|---|
| Find all companies related to Apple | `gq.related_entities("Apple", entity_type="Company")` |
| Entities connected to OpenAI within 2 hops | `gq.neighbors("OpenAI", hops=2)` |
| Why is Entity A connected to Entity B? | `gq.explain("Entity A", "Entity B")` |
| Show all events involving Company X | `gq.events_for("Company X")` |
| Show evidence for Relationship Y | `gq.evidence_for(src, rel, tgt)` |

---

## 6. Integration with reasoning

Run multi-document reasoning **before** querying so inferred edges are
available:

```python
from drg.reasoning import MultiDocumentReasoner

MultiDocumentReasoner().reason(kg)
gq = GraphQuery(kg)
exp = gq.explain("Apple", "Jimmy Iovine", include_inferred=True)
```

Pass `include_inferred=False` to restrict answers to extracted edges only.

---

## 7. Future: Neo4j backend

`Neo4jExporter` already writes `EnhancedKG` to Neo4j. A `Neo4jBackend`
would implement `QueryBackend` using Cypher:

```cypher
MATCH (a {id: $source})-[r*1..3]-(b {id: $target})
RETURN r
```

The `GraphQuery` facade would remain unchanged; only the backend swaps.

---

## 8. Relationship to `drg.graph.query_engine`

`drg.graph.query_engine.execute_query` powers the UI query box and remains
**unchanged**. `GraphQuery.search()` wraps it and enriches results with
full `EdgeView` objects and provenance. Existing API server endpoints
continue to work without modification.
