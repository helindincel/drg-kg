# Multi-Document Reasoning

> **Status:** Stable, opt-in. Lives in `drg.reasoning`. Pure stdlib +
> the existing `drg.graph` / `drg.utils` modules — **no new heavy
> dependencies**, **no LLM calls**, **no fabricated relationships**.
>
> **Goal:** Discover and surface meaningful connections **between
> information originating from different documents** — without breaking
> the existing extraction pipeline and without speculating.

---

## 1. Why this exists

The original DRG pipeline is one document at a time:

1. `drg.extract.extract_typed(text, schema)` → `(entities, triples)`
2. `drg.graph.builders.build_enhanced_kg(entities, triples)` →
   `EnhancedKG`
3. `drg.graph.GraphMerger.merge(base_kg, new_kg)` → merge into a
   long-lived KG, with entity dedup + edge dedup + version history.

That stack is excellent at **merging** what's already extracted. It is
**not** designed to **reason across documents**. Concretely, given:

- Document A: *"Apple acquired Beats."* → `Apple --ACQUIRED--> Beats`
- Document B: *"Jimmy Iovine founded Beats."* → `Jimmy Iovine --FOUNDED--> Beats`

after merging you get a graph with three nodes and two edges. The
graph *knows* that Apple and Jimmy Iovine are both connected to
Beats, but no edge directly expresses that connection. Anyone querying
the graph for "Is Apple linked to Jimmy Iovine?" gets back nothing
unless they manually traverse two hops and split the result over the
two source documents.

`drg.reasoning` closes that gap. It runs **after** extraction and
merging, walks the graph, applies a small set of deterministic
inference rules, and adds **new inferred edges** that carry full
provenance back to the extracted edges (and the documents) that
license them.

---

## 2. Design principles

| Principle | What it means in practice |
|---|---|
| **No LLM** | All inference is rule-based + deterministic. No prompts, no temperature, no hallucination risk. |
| **No fabricated entities** | Rules can only emit edges between nodes that already exist in the graph. Endpoints created by extraction or merge, never by the reasoner. |
| **Evidence chains are mandatory** | Every inferred edge requires at least one `EvidenceLink` citing extracted edges. Edges without evidence cannot be constructed (validated in `InferredEdge.__post_init__`). |
| **Extracted vs. inferred is always distinguishable** | Inferred edges carry `metadata.inferred = True` plus a `metadata.inference = {...}` provenance bag. UI / query / export layers can filter or style on this flag. |
| **Conservative by default** | Confidence floors, per-bridge fan-out caps, and a "require distinct relations on bridge edges" gate all default to the strict side. |
| **Idempotent** | Running the reasoner twice on the same graph produces no new edges the second time. |
| **Existing pipeline untouched** | `extract`, `chunking`, `coreference_resolution`, `entity_resolution`, `clustering` — none of these change. Their tests stay green. |

---

## 3. Mental model

```
                    extract            build_enhanced_kg
text  ────────────►  per doc  ─────►  per doc EnhancedKG  ──┐
                                                            │
                                                            ▼
                                              ┌──────────────────────┐
                                              │     GraphMerger      │
                                  base KG ──► │  .merge(base, doc)   │ ──► merged KG
                                              └──────────────────────┘
                                                            │
                                                            ▼
                                              ┌──────────────────────┐
                                              │ MultiDocumentReasoner│  ──► merged KG
                                              │      .reason(kg)     │      + inferred
                                              └──────────────────────┘        edges
```

The reasoner is the **last** pass: it sees the already-merged graph
with `source_ref` and `source_documents` provenance baked in by the
builders / merger.

---

## 4. The provenance schema (cross-cutting)

For multi-document reasoning to be meaningful, the system has to know
which edge came from which document. Two small, fully back-compatible
additions to the pipeline make this possible:

### 4.1 `build_enhanced_kg(document_id=…)`

When you pass a `document_id` to `build_enhanced_kg`, every edge it
builds gets `metadata['source_ref'] = document_id` (existing
`source_ref` values from `enriched_relations` win), and every node
gets `metadata['source_documents'] = [document_id]`. Without the
argument, the legacy behaviour is byte-for-byte preserved.

```python
from drg.graph.builders import build_enhanced_kg

kg = build_enhanced_kg(
    entities_typed=[("Apple", "Company"), ("Beats", "Company")],
    triples=[("Apple", "ACQUIRED", "Beats")],
    document_id="doc_A_apple_news",
)
assert kg.edges[0].metadata["source_ref"] == "doc_A_apple_news"
assert kg.nodes["Apple"].metadata["source_documents"] == ["doc_A_apple_news"]
```

### 4.2 `GraphMerger.merge(..., document_id=…)`

The merger already accepts `document_id`; on top of the existing
history entry, it now also:

- Stamps `source_ref = document_id` on any **incoming edge** that
  doesn't already carry one (so even hand-built per-document KGs gain
  provenance "for free" when merged).
- Extends `metadata['source_documents']` on each matched / added node
  with the active `document_id`.

Behaviour for edges and nodes that *already* carry `source_ref` or
`source_documents` is preserved verbatim.

### 4.3 What the reasoner reads

The reasoning rules consult **only**:

- `KGEdge.metadata['source_ref']` (string) — for cross-document
  detection in `PathBridgeRule`.
- `KGEdge.metadata['inferred']` (bool) — to avoid chaining off
  previously-inferred edges (configurable).
- `KGEdge.confidence` (float | None) — folded into inferred-edge
  confidence.

If `source_ref` is missing on an edge, the rule treats that edge's
document of origin as **unknown** and the bridge rule abstains rather
than guessing.

---

## 5. Quick start (Python)

```python
from drg.graph import EnhancedKG, GraphMerger
from drg.graph.builders import build_enhanced_kg
from drg.reasoning import MultiDocumentReasoner

base = EnhancedKG()
merger = GraphMerger()

# Per-document extraction (replace with your real extraction calls)
for doc_id, entities, triples in your_documents:
    new_kg = build_enhanced_kg(
        entities_typed=entities,
        triples=triples,
        document_id=doc_id,            # ← stamps source_ref on edges
    )
    merger.merge(base, new_kg, document_id=doc_id)

# Multi-document reasoning
report = MultiDocumentReasoner().reason(base)
print(report.summary())
# {'added_edges': 5, 'per_rule_counts': {'path_bridge': 1, 'inverse': 2, ...}, ...}

# Tell extracted vs. inferred edges apart
extracted = [e for e in base.edges if not e.metadata.get("inferred")]
inferred  = [e for e in base.edges if e.metadata.get("inferred")]
```

---

## 6. Quick start (CLI)

```bash
# Per-document ingestion + reasoning after every merge
drg extract docs/apple_news.txt   --auto-schema --update outputs/global_kg.json \
    --update-document-id doc_A_apple_news   --infer

drg extract docs/iovine_bio.txt   --auto-schema --update outputs/global_kg.json \
    --update-document-id doc_B_iovine_bio   --infer
```

| Flag | Default | Effect |
|---|---|---|
| `--infer` | off | Run `MultiDocumentReasoner` after extraction (and after the merge, when combined with `--update`). |
| `--infer-min-confidence FLOAT` | `0.5` | Drop inferred edges below this confidence. |
| `--infer-disable-rule NAME` | (none) | Disable a built-in rule by its `name` (`path_bridge`, `inverse`, `symmetric`, `transitive`, `composition`). Pass multiple times for several. |
| `--update-document-id ID` | input filename | Used by the merger for the history entry **and** stamped as the `source_ref` on every edge built from this document. |

You can also run the reasoner standalone over an existing KG file via
the Python API (no CLI shortcut yet — see [§8](#8-public-api)).

---

## 7. Built-in rule catalog

All rules return `InferredEdge` candidates. The engine validates each
candidate (endpoints exist, no self-loop, no duplicate, passes the
confidence floor) before persisting it as a real `KGEdge`.

### 7.1 `PathBridgeRule` — the multi-document workhorse

**Triggers:** two evidence edges share a "bridge" node, and the two
edges have **different** `source_ref` values (cross-document).

**Emits:** `outer_a --connected_via_<bridge_slug>--> outer_b`. The
endpoint order is the lexicographic-smaller-first, deterministic
across runs.

**Conservatism gates (defaults in [§9](#9-reasoning-config)):**

- Same-document evidence is skipped (`source_ref` equality check).
- Edges without `source_ref` are skipped (no guessing the document).
- When `require_distinct_bridge_relations=True` (default), the two
  evidence edges must have different relation types — prevents a fact
  re-observed in two documents from triggering a spurious connection.
- Per-bridge candidate cap (`max_bridge_candidates_per_node`, default
  32) keeps hub bridges from exploding the output.

**Confidence:** `max(bridge_confidence_floor, min(1.0, c1 * c2))`
where `c1`, `c2` are the evidence edges' confidences (defaulting to
0.8 if missing). The floor (default 0.6) keeps cross-document
inferences on a slightly stricter leash than the other rules.

### 7.2 `InverseRule`

**Triggers:** edge `A --r--> B` exists, `r` is in
`INVERSE_RELATION_PAIRS`, and the inverse direction `B --r'--> A` is
not already present.

**Emits:** `B --r'--> A` with confidence equal to the source edge.

**Default inverse pairs:** `founded ↔ founded_by`, `owns ↔ owned_by`,
`acquired ↔ acquired_by`, `produces ↔ produced_by`, `manufactures ↔
manufactured_by`, `employs ↔ employed_by`, `parent_of ↔ child_of`,
`contains ↔ part_of`, `has_part → part_of`.

### 7.3 `SymmetricRule`

**Triggers:** edge `A --r--> B` exists, `r` is in
`SYMMETRIC_RELATIONS`, and the back-edge is missing.

**Emits:** `B --r--> A` with confidence equal to the source edge.

**Default symmetric relations:** `works_with`, `collaborates_with`,
`married_to`, `sibling_of`, `similar_to`, `related_to`, `near`,
`communicates_with`.

### 7.4 `TransitiveRule`

**Triggers:** `A --r--> B` and `B --r--> C` both exist, `r` is in
`TRANSITIVE_RELATIONS`, and `A --r--> C` is missing.

**Emits:** one transitive hop — `A --r--> C`. To reach the full
closure on a longer chain, call `reason()` again; the engine's
idempotency check makes repeated calls safe. By default the rule
**does not** chain off previously-inferred edges (toggle via
`allow_inferred_in_evidence`).

**Default transitive relations:** `part_of`, `subclass_of`,
`located_in`, `contains`, `ancestor_of`, `descendant_of`.

### 7.5 `CompositionRule`

**Triggers:** `A --owns--> B` and `B --located_in--> L` (or any
combination of ownership/location synonyms) and `A --operates_in--> L`
is missing.

**Emits:** `A --operates_in--> L`. This is the **graph-only**
complement to the existing extract-time
`_infer_implicit_relations` heuristic, which requires the source text
to contain "operation cues". The graph-only version naturally bridges
two documents that contributed the two evidence edges separately.

---

## 8. Public API

```python
from drg.reasoning import (
    MultiDocumentReasoner,    # the engine
    ReasoningConfig,          # knobs
    InferenceRule,            # base class for custom rules
    InferredEdge,             # rule output type
    EvidenceLink,             # one cited extracted edge
    InferenceReport,          # returned by reason()
    PathBridgeRule,
    InverseRule,
    SymmetricRule,
    TransitiveRule,
    CompositionRule,
    default_rules,            # the bundled built-ins as a list
    reason_over_graph,        # one-call convenience wrapper
)
```

Lazy top-level imports also work:

```python
from drg import MultiDocumentReasoner, ReasoningConfig
```

### 8.1 `MultiDocumentReasoner(rules=None, config=None)`

- `rules`: list of `InferenceRule` instances. Defaults to
  `default_rules()`. Pass an empty list to turn the reasoner into a
  no-op; pass `[MyCustomRule(), ...]` to use only your own rules.
- `config`: `ReasoningConfig`. Defaults to a conservative profile.

### 8.2 `reason(kg, *, document_id=None, record_history=True, dry_run=False)`

Mutates `kg` in place and returns an `InferenceReport`. `dry_run=True`
validates and reports candidates without mutating the graph.
`record_history=True` appends a `reasoning` entry to
`kg.metadata['history']` (mirroring the merger's bookkeeping).

### 8.3 `InferenceReport`

Typed report of a single run:

```python
report.added_edges                # list of (s, r, t) added
report.skipped_existing           # candidates dropped as duplicates
report.skipped_low_confidence     # candidates dropped by min_confidence
report.skipped_self_loop          # endpoints collapsed
report.skipped_missing_endpoint   # rule emitted unknown endpoint
report.per_rule_counts            # {'path_bridge': 2, 'inverse': 3, ...}

report.summary()                  # compact stats dict
report.is_empty()                 # True iff the run was a no-op
report.to_dict()                  # JSON-serialisable
```

---

## 9. Reasoning config

`ReasoningConfig` is a frozen dataclass:

```python
from drg.reasoning import ReasoningConfig

ReasoningConfig(
    min_confidence=0.5,                       # global floor on inferred-edge confidence
    max_inferences_per_run=None,              # hard cap on total inferred edges per run
    disabled_rules=frozenset({"path_bridge"}),# disable rules by name
    max_bridge_candidates_per_node=32,        # per-bridge fan-out cap (PathBridgeRule)
    bridge_confidence_floor=0.6,              # lower bound applied to bridge confidences
    require_distinct_bridge_relations=True,   # bridge rule needs r1 != r2
    allow_inferred_in_evidence=False,         # rules cannot chain off inferred edges
)
```

---

## 10. The inferred-edge metadata schema

Every inferred edge gets two metadata keys:

```python
edge.metadata = {
    "inferred": True,
    "inference": {
        "rule": "path_bridge",
        "evidence_chain": [
            {"triple": ["Apple", "ACQUIRED", "Beats"],
             "source_ref": "doc_A_apple_news", "confidence": 0.95},
            {"triple": ["Jimmy Iovine", "FOUNDED", "Beats"],
             "source_ref": "doc_B_iovine_bio", "confidence": 0.90},
        ],
        "source_documents": ["doc_A_apple_news", "doc_B_iovine_bio"],
        "explanation": "Apple and Jimmy Iovine are both connected to Beats. "
                       "Evidence: Apple —[ACQUIRED]→ Beats (source: doc_A_apple_news); "
                       "Jimmy Iovine —[FOUNDED]→ Beats (source: doc_B_iovine_bio).",
        "confidence": 0.855,
        "bridge_entity": "Beats",            # optional: PathBridgeRule only
        "extra": {"intermediate": "Beats"},  # optional: rule-specific bag
    },
}
```

Plus the edge itself carries `confidence` and `relationship_detail`
(set to the explanation). The whole bag is `json.dumps`-safe and
round-trips through `EnhancedKG.save_json` / `load_json`.

---

## 11. Writing a custom rule

Subclass `InferenceRule`, give it a stable `name`, and implement
`apply`:

```python
from drg.reasoning import InferenceRule, InferredEdge, EvidenceLink, ReasoningConfig

class CitationRule(InferenceRule):
    name = "citation"

    def apply(self, kg, config: ReasoningConfig):
        out = []
        for e in kg.edges:
            if e.metadata.get("inferred"):
                continue   # don't chain off inferences
            if e.relationship_type.lower() != "cites":
                continue
            # ... produce InferredEdge(s) with full provenance ...
        return out
```

Then plug it into the reasoner:

```python
from drg.reasoning import MultiDocumentReasoner, default_rules

reasoner = MultiDocumentReasoner(rules=default_rules() + [CitationRule()])
```

The engine validates every `InferredEdge` you return (endpoints
exist, no self-loop, no duplicate, passes the confidence floor), so
your rule only needs to express the **inference logic**, not the
plumbing.

---

## 12. Limitations / non-goals

- **No transitive closure beyond the whitelisted relations.** Adding
  e.g. `causes` to `TRANSITIVE_RELATIONS` is a one-line change in
  `drg/reasoning/_rules.py` — but is left to the schema author because
  transitivity is not universally safe (`causes` chains can compound
  uncertainty).
- **No probabilistic / belief-propagation reasoning.** Confidence
  scores are deterministic functions of the evidence; there's no
  Bayesian update.
- **Bridge rule only emits 2-hop bridges.** Longer chains are out of
  scope by design — every additional hop weakens the evidence.
- **No cross-document entity resolution beyond what the merger does.**
  Fuzzy entity merges are handled by `drg.entity_resolution`
  (intra-document) and `drg.graph.GraphMerger` (cross-document); the
  reasoner inherits whatever decisions those layers made.

---

## 13. See also

- [`examples/multi_document_reasoning_example.py`](../examples/multi_document_reasoning_example.py)
  — runnable, no-LLM end-to-end demo (the Apple/Beats/Jimmy Iovine
  scenario from this doc).
- [`tests/test_reasoning.py`](../tests/test_reasoning.py) — 53 tests
  covering every rule, the engine invariants, JSON round-trip, and
  the headline scenario.
- [`docs/incremental_updates.md`](incremental_updates.md) — the
  `GraphMerger` layer this module sits on top of.
- [`drg/reasoning/`](../drg/reasoning) — source of truth.
