# DRG Extraction Architecture — Implementation Report

> **Phase**: P0 + P1 implementation following `VERIFIED_FINDINGS.md`  
> **Constraint**: DRG is a schema-first, modular-monolith knowledge-graph extraction library. It is NOT GraphRAG, hybrid search, generic RAG, or a vector database framework. All changes are backward-compatible.

---

## 1. Summary

Six P0 bugs and three P1 architectural gaps were implemented across two source files and one new test file. All changes are strictly limited to the scope described in `VERIFIED_FINDINGS.md`. No refactoring, API removals, or speculative improvements were made.

**Test outcome**: 202 extraction-related tests pass. 0 regressions in extraction path. 17/17 new signature metadata tests pass.

---

## 2. Files Modified

### `drg/extract/_signatures.py`

This file contains the DSPy signature factory functions that define what the LLM sees for each extraction task.

**Before** (pre-session state):
- `_relation_schema_for()` returned `{"name", "source_type", "target_type"}` only; all semantic fields discarded.
- `_create_entity_signature()` built `entity_types_list = [et.name for et in schema.entity_types]` — flat `list[str]`, descriptions and examples discarded.
- All five signature `OutputField` annotations were `list[dict]` — generic object array, no field structure visible to LLM.
- All five signature docstrings contained one uninstructive sentence.

**Changes applied**:

| Change | Addresses | Detail |
|---|---|---|
| Added `_entity_schema_for(schema)` helper | VF-1, VF-7, VF-10 | Returns `list[dict]` with `name`, `description`, `examples[:5]`, and optionally `group`/`group_description` from `EntityGroup` membership |
| Updated `_relation_schema_for()` | VF-1, VF-6, P1-1, P1-3 | EnhancedDRGSchema now includes `description`, `example` (from `Relation.detail`), `group`, `group_description` (from `RelationGroup`); legacy `DRGSchema` now includes `description` and `example` |
| Changed `_create_entity_signature` input field | VF-10 | `entity_types: list[dict]` (was `list[str]`); populated from `_entity_schema_for(schema)` |
| Changed all five `OutputField` annotations | VF-2, VF-4 | `entities: list[EntityMention]`, `relations: list[ExtractedRelation]`, `resolved_relations: list[ExtractedRelation]` |
| Added behavioral docstrings to all five signatures | VF-9, P0-5 | `EntityExtraction`: instructs on evidence and alias population. `RelationExtraction`/`DocumentRelationExtraction`/`ImplicitRelationExtraction`: instructs on `is_negated`, `confidence`, `evidence`, `temporal`. `CoreferenceResolution`: instructs on canonical name resolution. |

**Unchanged**: The `KGExtractor._entity_types` line in `__init__.py` (`getattr(EntitySig, "_entity_types", [])`) now naturally stores `list[dict]` because `EntityExtraction._entity_types` is set to the `list[dict]` output of `_entity_schema_for`. No change required in that file for VF-10.

---

### `drg/extract/__init__.py`

This file contains the main extraction pipeline: `extract_typed`, `extract_from_chunks`, and `KGExtractor`.

**Three targeted changes were applied** (no structural refactoring):

#### Change 1 — Wire `context_entities` in single-pass loop (VF-3, P0-1)

**Location**: Single-pass extraction loop inside `extract_from_chunks`.

**Before**: `result = extractor(text=chunk_text)` — `context_entities` was accumulated but never passed.

**After**:
```python
result = extractor(
    text=augmented_text,
    context_entities=(
        context_entities
        if enable_cross_chunk_relationships and context_entities
        else None
    ),
)
```

Additionally, the single-pass loop now maintains a running `sp_entity_to_chunks` index and injects cross-chunk context snippets via `_select_anchor_entities` + `_build_cross_chunk_context_snippets` from previously-processed chunks before each extractor call (when `enable_cross_chunk_context_snippets=True`).

**Effect**: Cross-chunk entity references are now visible to the relation extractor in single-pass mode. Previously, relation extraction could only see entities in the current chunk.

---

#### Change 2 — Wire cross-chunk context snippets into two-pass per-chunk path (VF-4, P0-2)

**Location**: Two-pass Pass 2 per-chunk loop inside `extract_from_chunks`.

**Before**: `_build_cross_chunk_context_snippets` and `_select_anchor_entities` were imported and their controlling parameters were accepted, but neither function was ever called. The parameter `enable_cross_chunk_context_snippets=True` (default) had zero effect.

**After**: After Pass 1 entity resolution, an entity-to-chunks index is built mapping each canonical entity name to the list of chunk indices where it appears. Before each chunk is processed in Pass 2, anchor entities are selected via `_select_anchor_entities` and context snippets are retrieved via `_build_cross_chunk_context_snippets`. When snippets are available, the chunk text is augmented:

```python
augmented_text = chunk_text + "\n\n[Cross-document context]\n" + context_block
```

**Effect**: The LLM now receives coreference-disambiguating context from other chunks when extracting relations within a chunk. The mechanism was fully implemented and tested in isolation (`tests/test_cross_chunk_snippets.py`, `tests/test_anchor_selection.py`); this change activates it.

---

#### Change 3 — Schema validation gate in `extract_from_chunks` (VF-8, P0-6)

**Location**: Post-deduplication block in `extract_from_chunks`.

**Before**: `extract_typed` called `_filter_against_schema()` to remove entities with invalid types and relations not present in the schema. `extract_from_chunks` had no equivalent call, so schema-violating triples were returned unchecked.

**After**:
```python
try:
    all_entities, all_triples = _filter_against_schema(schema, all_entities, all_triples)
    # Remap enriched relations after filtering
    ...
except Exception as exc:
    if strict:
        raise
    logger.warning("Schema validation gate failed: %s", exc)
```

**Effect**: Output from `extract_from_chunks` now has parity with `extract_typed` for schema validation. Strict mode re-raises; non-strict logs and continues.

---

## 3. File Added

### `tests/test_signature_metadata.py`

New test file with 17 tests across four test classes.

| Class | Tests | What is verified |
|---|---|---|
| `TestEntitySchemaFor` | 3 | `_entity_schema_for()` returns `list[dict]` with `name`/`description`/`examples`; includes `group`/`group_description` from `EntityGroup` |
| `TestRelationSchemaFor` | 4 | `_relation_schema_for()` includes `description`, `detail`, `group`, `group_description` for `EnhancedDRGSchema`; `description` and `example` for legacy schema; no group keys in legacy schema |
| `TestEntitySignatureTypes` | 3 | `_create_entity_signature` stores `_entity_types` as `list[dict]`; each dict contains a `description` key |
| `TestRelationSignatureSchema` | 1 | Relation schema dicts include `description` |
| `TestSignatureDocstrings` | 6 | Behavioral keywords (`evidence`, `negat`/`is_negated`, `confidence`, `temporal`, `canonical`) exist in `_signatures.py` source; verified via source-file read to avoid MagicMock metaclass interference |

**Implementation note on docstring tests**: When `dspy.Signature` is replaced with `MagicMock()` in the test environment (standard practice for offline tests), Python's `class Inner(MagicMock):` invocation causes the inner class's `__doc__` to be absorbed or inaccessible via normal attribute lookup. The six docstring tests verify the source file `_signatures.py` directly using `pathlib.Path.read_text()`, which is mock-safe and reads the canonical source of truth.

---

## 4. Rejected Findings — Not Implemented

The following items from the architecture review were explicitly NOT implemented:

| Finding | Decision | Reason |
|---|---|---|
| "DefaultConfidenceStrategy not wired" | ❌ Rejected | `DefaultConfidenceStrategy` IS wired correctly in `drg/graph/builders.py` via `build_enhanced_kg()`. This is the correct integration point — confidence is a graph-level concern. |
| Two-pass asymmetry decoupling | ⚠️ Not implemented | The asymmetric design is intentional (Pass 1 = entity-focused, Pass 2 = relation-focused). Decoupling would be a major refactor with unclear benefit and no correctness impact. |
| Reverse-relation lookup table as architecture gap | ⚠️ Not implemented | The table is opt-in (`enable_reverse_relation_fallback=False` by default). Not a correctness bug in the default path. |
| Length-based canonicalization accuracy | ⚠️ Not implemented | Requires benchmarking against labeled data before any change. |

---

## 5. Remaining Architectural Gaps (P2 / P3)

These items were identified in `VERIFIED_FINDINGS.md` but are out of scope for this session:

### P2 — Schema Discovery

| ID | Description |
|---|---|
| P2-1 | Schema completeness validation: detect orphan entity types, disconnected relations |
| P2-2 | Hybrid content-aware sampling for schema generation |
| P2-3 | Iterative schema coverage loop |

### P3 — Future Research

| ID | Description |
|---|---|
| P3-1 | Add span/chunk provenance to `EntityMention` and `ExtractedRelation` (breaking API change — requires migration plan) |
| P3-2 | Calibrate entity resolution thresholds against labeled benchmark data |
| P3-3 | Make `SchemaGenerator` a proper `dspy.Module` for DSPy optimizer integration |
| P3-4 | Bitemporal documentation for `TemporalInfo` (valid-time vs. transaction-time disambiguation) |
| P3-5 | `PropertyGroup` extraction signature (new extraction task required) |

### P1-2 Partial — `EntityGroup` context

`EntityGroup` context (`group_name`, `group_description`) is now included in `_entity_schema_for()` output when an entity type belongs to a group. This covers the information-passing aspect. What remains (P2 scope): restructuring the extraction loop so that entity types within the same group are processed together, allowing the LLM to use intra-group contrast as disambiguation signal.

---

## 6. Test Regression Summary

All extraction-related tests were run after each change. Results at session end:

```
tests/test_extract_core.py         PASS
tests/test_extract_mock.py         PASS
tests/test_cross_chunk_snippets.py PASS
tests/test_anchor_selection.py     PASS
tests/test_extract_relations.py    PASS
tests/test_extract_parsing.py      PASS
tests/test_extract_schema_gen.py   PASS
tests/test_implicit_relations.py   PASS
tests/test_schema_parsing.py       PASS
tests/test_di_and_protocols.py     PASS
tests/test_entity_resolution_package.py PASS
tests/test_entity_resolution_safety.py  PASS
tests/test_graph_builders.py       PASS
tests/test_relation_metadata_heuristics.py PASS
tests/test_basic.py                PASS
tests/test_integration.py          PASS (2 skipped — require DRG_RUN_INTEGRATION=1)
tests/test_top_level_api.py        PASS
tests/test_signature_metadata.py   PASS (17/17)
--------------------------------------------------------------
Total: 202 passed, 2 skipped, 0 failed
```

**Pre-existing failures** (unrelated to this session): Tests in `test_confidence.py`, `test_events_*.py`, `test_kg_core.py`, `test_query_layer.py`, `test_reasoning.py`, `test_provenance_versioning.py`, and `test_mcp_api_contract.py` fail with `ModuleNotFoundError: No module named 'drg._version'`. This is a packaging issue — `drg._version` is generated by the build system (`hatch-vcs`) and is absent in the development checkout. These failures predate this session.

---

## 7. VF-to-Implementation Mapping

| Verified Finding | P-level | Status | Implementation |
|---|---|---|---|
| VF-1: Schema metadata not passed to LLM | P0-3 | ✅ Done | `_entity_schema_for()`, `_relation_schema_for()` updated |
| VF-2: OutputField annotations are `list[dict]` | P0-4 | ✅ Done | All 5 signatures now use typed Pydantic annotations |
| VF-3: `context_entities` never passed to extractor | P0-1 | ✅ Done | Single-pass loop now passes `context_entities` |
| VF-4: Cross-chunk snippets never called | P0-2 | ✅ Done | Two-pass and single-pass paths now call snippet functions |
| VF-5: DefaultConfidenceStrategy not wired | — | ❌ Rejected | Already correctly wired in `graph/builders.py` |
| VF-6: `RelationGroup` metadata discarded | P1-1 | ✅ Done | `group`, `group_description` added to relation schema dicts |
| VF-7: `EntityGroup` unused | P1-2 (partial) | ✅ Partial | Group context added to entity schema dicts; loop restructuring is P2 |
| VF-8: Schema validation gate missing from `extract_from_chunks` | P0-6 | ✅ Done | `_filter_against_schema` called after deduplication |
| VF-9: Signature docstrings provide no behavioral guidance | P0-5 | ✅ Done | All 5 signatures have behavioral instruction docstrings |
| VF-10: `_entity_types` is `list[str]` | P0-3 / P1-2 | ✅ Done | `_entity_schema_for()` returns `list[dict]`; KGExtractor stores automatically |
