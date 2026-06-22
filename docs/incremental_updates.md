# Incremental Knowledge Graph Updates

> **Status:** Stable, opt-in. Lives in `drg.graph.incremental`. No new heavy
> dependencies; only stdlib + the existing `drg.entity_resolution` package.
>
> **Goal:** Add new documents to an existing KG **without rebuilding from
> scratch** — reusing entities/relationships that already exist, adding
> only what's new, and keeping a per-update audit trail.

## 1. When you want this

Use the incremental layer when any of the following is true:

- You're ingesting documents over time (news feeds, paper drops,
  internal docs) and don't want to re-LLM the whole archive on every
  run.
- You want a single, long-lived KG file in `outputs/global_kg.json`
  rather than per-document `outputs/<doc>_kg.json` files.
- You need `version` / `history` metadata for audit / debugging
  (e.g. "which document introduced this edge?").
- You want strong duplicate-entity / duplicate-edge guarantees that go
  beyond the in-chunk dedup the extraction pipeline already does.

If your workflow is "build once, throw away" — keep using the existing
`build_enhanced_kg` + `kg.save_json` flow. The incremental layer is
fully opt-in and does **not** change that path.

## 2. Mental model

```
                     ┌─────────────────────┐
   doc_1 ─► extract ─►  build_enhanced_kg  │──┐
                     └─────────────────────┘  │
                                              ▼
                                     ┌─────────────────┐
                                     │   GraphMerger   │
                  base KG (on disk) ─►  .merge(base,   │─► updated base + KGDiff
                                     │     incoming)   │
                                     └─────────────────┘
                                              ▲
                     ┌─────────────────────┐  │
   doc_2 ─► extract ─►  build_enhanced_kg  │──┘
                     └─────────────────────┘
```

The merger:

1. Looks for matching nodes by (a) exact id, then (b) normalized name +
   type.
2. Folds incoming node data into matched nodes per the configured
   policy.
3. Re-routes incoming edges through the resulting id remap.
4. Skips edges whose canonical triple already exists (dedup).
5. Bumps the graph's `metadata.version` and appends a
   `metadata.history` entry that captures the diff.

Every call returns a [`KGDiff`](#kgdiff) for callers to inspect.

## 3. Quick start (Python)

```python
from drg.graph import EnhancedKG, GraphMerger
from drg.graph.builders import build_enhanced_kg

# 1) Re-hydrate the persisted KG (or start empty on first run).
kg_path = "outputs/global_kg.json"
try:
    base = EnhancedKG.load_json(kg_path)
except FileNotFoundError:
    base = EnhancedKG()

# 2) Run the existing extraction pipeline on the new document.
new_kg = build_enhanced_kg(
    entities_typed=entities,
    triples=triples,
    source_text=text,
    schema=schema,
)

# 3) Merge — entity matching, dedup, version bump all happen here.
diff = GraphMerger().merge(base, new_kg, document_id="doc_2026_06_06")
print(diff.summary())
# {'added_nodes': 4, 'merged_nodes': 12, 'added_edges': 7, 'skipped_edges': 3, ...}

# 4) Persist; the file now carries a fresh version + history entry.
base.save_json(kg_path)
```

That's the whole loop. Subsequent calls just reload, merge, save.

## 4. Quick start (CLI)

```bash
# First run — creates outputs/global_kg.json from scratch.
drg extract docs/article_1.txt --auto-schema -o outputs/global_kg.json

# Every subsequent run — appends to the same KG file.
drg extract docs/article_2.txt --auto-schema --update outputs/global_kg.json
drg extract docs/article_3.txt --auto-schema --update outputs/global_kg.json

# Validate and compare persisted snapshots.
drg validate outputs/global_kg.json
drg diff outputs/global_kg.before.json outputs/global_kg.json --json

# Persist the per-merge audit diff while updating.
drg extract docs/article_4.txt --auto-schema --update outputs/global_kg.json \
  --diff-output outputs/article_4_merge_diff.json

# Inspect version snapshots created by incremental updates.
drg versions list outputs/global_kg.json
drg versions diff outputs/global_kg.json v1 v2 --json
```

CLI flags:

| Flag | Default | Meaning |
|---|---|---|
| `--update PATH` | unset | Enable incremental ingestion. Loads PATH (or starts empty if missing), merges the freshly extracted graph into it, writes back. |
| `--update-strategy {prefer_existing, prefer_new, union}` | `prefer_existing` | Node merge policy. See [§ 6](#6-merge-strategies). |
| `--update-document-id` | input filename | Identifier recorded in the history entry. |
| `--diff-output PATH` | unset | Write the merge-time `KGDiff` audit report as JSON. |

When `--update` is set:

- `--output-format` is forced to `enhancedkg` (legacy KG class has no
  metadata surface).
- If `-o` is omitted, output is written back to the `--update` path.

`drg diff OLD NEW` is a separate snapshot diff command. It compares two saved
KG files after validation and is intended for CI/regression gates. Merge diffs
come from `--diff-output`; snapshot diffs come from `drg diff`.

Every `--update` run also creates a graph version snapshot in a manifest
directory next to the updated KG file. See `docs/graph_versioning.md` for
rollback and version-to-version diff commands.

Merged nodes and edges preserve source metadata under both legacy keys
(`source_ref`, `source_documents`, `evidence`) and the structured
`metadata.provenance` block. See `docs/provenance_tracking.md` for the full
contract.

## 5. The public API

```python
from drg.graph import (
    EnhancedKG,           # gained: load_json(), from_dict(), metadata field
    GraphMerger,          # the merger
    KGDiff,               # report returned by merger.merge()
    MergeStrategy,        # bundle of policies / parameters
    NodeMergePolicy,      # PREFER_EXISTING | PREFER_NEW | UNION
    EdgeMergePolicy,      # SKIP | APPEND_EVIDENCE | MAX_CONFIDENCE
    merge_graphs,         # one-call convenience over GraphMerger
)
```

Top-level lazy import works too:

```python
from drg import EnhancedKG, GraphMerger, MergeStrategy
```

### `EnhancedKG.load_json(path)` / `EnhancedKG.from_dict(data)`

Symmetrical with `kg.save_json(path)` and the existing JSON shape. Loads:

- nodes / edges / clusters with full attributes;
- optional `metadata` key (silently empty for legacy files written
  before incremental was introduced).

Backward-compatible: a graph saved without `metadata` still loads
cleanly, and `kg.save_json()` for a graph with empty metadata still
produces the legacy `{nodes, edges, clusters}` shape byte-for-byte.

### `GraphMerger(strategy=None).merge(base, incoming, *, document_id=None, record_history=True)`

Mutates `base` in place and returns a `KGDiff`. Pass `document_id` to
get it embedded in the history entry. Pass `record_history=False` to
silently merge without touching `base.metadata` (useful for tests).

### `merge_graphs(base, incoming, *, strategy=None, document_id=None, record_history=True)`

One-call convenience around `GraphMerger`. Identical behaviour.

### `KGDiff`

Structured, JSON-serialisable report of a single merge:

```python
diff.added_nodes        # ['Steve Jobs', ...]
diff.merged_nodes       # [('Apple Inc', 'apple inc'), ...]
diff.skipped_nodes      # [('mercury', 'type_mismatch'), ...]
diff.added_edges        # [('Steve Jobs', 'FOUNDED', 'Apple Inc'), ...]
diff.skipped_edges      # duplicates that the dedup logic dropped
diff.rewritten_edges    # endpoints rewritten through the id remap
diff.added_clusters     # cluster ids copied from the incoming graph
diff.skipped_clusters   # cluster ids dropped (id collision / no valid members)

diff.is_empty()         # True iff the merge was a no-op
diff.summary()          # {'added_nodes': 4, 'merged_nodes': 12, ...}
diff.to_dict()          # JSON-serialisable form (used in history entries)
```

## 6. Merge strategies

`MergeStrategy` is a frozen dataclass; pick a single field or several:

```python
from drg.graph import MergeStrategy, NodeMergePolicy, EdgeMergePolicy

MergeStrategy(
    node_policy=NodeMergePolicy.UNION,
    edge_policy=EdgeMergePolicy.MAX_CONFIDENCE,
    require_type_match=True,        # default
    use_normalized_match=True,      # default
    case_insensitive_relation=True, # default
)
```

### Node policies

| Policy | Behaviour |
|---|---|
| `PREFER_EXISTING` (**default**) | Existing node wins. The incoming node is recorded under `metadata.merged_from` for provenance. Safest. |
| `PREFER_NEW` | The incoming node overwrites the existing node's mutable fields (`type`, `properties`, `metadata`, `embedding`, `confidence`). Use when the new document is more authoritative. |
| `UNION` | Take the union of `properties` and `metadata` (incoming wins on overlapping keys). Embeddings are averaged element-wise iff dimensions match. Higher `confidence` is kept. |

### Edge policies

| Policy | Behaviour |
|---|---|
| `SKIP` (**default**) | Drop the duplicate edge. Smallest graph, no fabricated evidence. |
| `APPEND_EVIDENCE` | Keep the original edge but extend its `metadata.evidence_refs` with the duplicate's `source_ref` / `evidence` / `confidence`. Useful for "how many documents independently observed this fact?" queries. |
| `MAX_CONFIDENCE` | Like `APPEND_EVIDENCE`, but additionally bumps the existing edge's confidence to the duplicate's value when strictly higher. The previous confidence is remembered in `metadata.alt_confidences`. |

### Matching parameters

| Parameter | Default | Effect |
|---|---|---|
| `require_type_match` | `True` | Two nodes only match if their `type` agrees. Setting to `False` lets nodes survive a schema-time type rename — at the cost of occasionally collapsing different entities that share a surface form. |
| `use_normalized_match` | `True` | Apply `drg.entity_resolution.normalize_entity_name` before key comparison (handles case, honorifics, whitespace). When `False`, only **byte-exact** id matches are recognised. |
| `case_insensitive_relation` | `True` | `RUNS_ON` and `runs_on` are treated as the same edge for dedup. The original surface form on the existing edge is preserved. |

## 7. Versioning + history

After a successful merge, `base.metadata` looks like:

```json
{
  "version": 3,
  "created_at": "2026-06-06T22:00:00+00:00",
  "updated_at": "2026-06-06T23:00:00+00:00",
  "history": [
    {
      "version": 1,
      "operation": "merge",
      "timestamp": "2026-06-06T22:00:00+00:00",
      "document_id": "doc_1",
      "added_nodes": 4, "merged_nodes": 0, "skipped_nodes": 0,
      "added_edges": 3, "skipped_edges": 0, "rewritten_edges": 0,
      "added_clusters": 0, "skipped_clusters": 0
    },
    { "version": 2, "operation": "merge", "document_id": "doc_2", ... },
    { "version": 3, "operation": "merge", "document_id": "doc_3", ... }
  ]
}
```

Properties:

- `version` is a **monotonically increasing integer**, starting at 1
  on the first incremental touch.
- `created_at` is set once on first touch; `updated_at` is set on
  every touch.
- `history` is **append-only** within the merger — the merger never
  rewrites earlier entries. (Callers can prune it manually if they
  want a bounded log; it lives in plain `metadata`.)
- Pass `record_history=False` to merge silently for tests / hot paths
  that maintain their own version metadata.

## 8. Conservative defaults — what the merger refuses

Following the same philosophy as `drg.entity_resolution.EntityResolver`,
the merger refuses to merge when in doubt:

- **Type mismatch.** With `require_type_match=True` (default), a
  `Person` named "Mercury" never collapses into an `Element` named
  "Mercury". The skip is reported in `diff.skipped_nodes` so callers
  can investigate.
- **Self-loop after remap.** When two endpoints would resolve to the
  same canonical id post-merge, the edge is dropped (it would have
  failed `KGEdge.__post_init__` validation). Reported in
  `diff.skipped_edges`.
- **Cluster id collision.** Existing cluster keeps its membership;
  incoming cluster with the same id is skipped (cluster renaming is
  out of scope — clusters are typically derived per-graph by
  Louvain/Leiden anyway).

## 9. Demo without LLM

[`examples/incremental_update_example.py`](../examples/incremental_update_example.py)
runs three synthetic documents through the merger end-to-end without
any LLM call. Output includes the diff per document, the final KG, and
the history audit trail. Use it as a copy-paste starting point.

```bash
python examples/incremental_update_example.py
```

## 10. Limitations / non-goals

- **No fuzzy entity matching beyond `normalize_entity_name`.** The
  merger does not invoke embedding similarity by itself — that's
  `drg.entity_resolution.EntityResolver`'s job, used inside the
  extraction pipeline. If you want fuzzy graph-level merges, run the
  resolver against the concatenated entity list before the merge call;
  the public API of that package is designed for this.
- **Clusters are not re-clustered.** When you merge two graphs that
  each have their own Louvain clusters, you keep both sets of
  clusters (id-collisions skipped). To produce fresh clusters across
  the merged graph, re-run your clustering pass after the merge.
- **No transactionality on disk.** `kg.save_json()` overwrites the
  target file. Callers that need atomic writes should write to a
  temp file and `os.replace()` (one-line wrapper around the existing
  helper).

## 11. See also

- [`drg.entity_resolution`](../drg/entity_resolution) — the in-document
  entity merging that runs *before* the merger ever sees a graph.
- [`drg.graph.kg_core`](../drg/graph/kg_core.py) — `EnhancedKG`,
  `KGNode`, `KGEdge`, `Cluster`.
- [`drg.graph.builders`](../drg/graph/builders.py) — how the existing
  pipeline produces an `EnhancedKG` from `(entities, triples)`.
