# DRG Extraction Architecture — Verified Findings

> **Methodology**: Every finding below was verified directly against the source code with exact file/line references. Findings from the architecture review that could not be confirmed, or that were partially wrong, are explicitly rejected with justification.

---

## Part 1 — Verified Findings

---

### VF-1 ✅ CONFIRMED — Schema metadata (descriptions, examples) not passed to LLM

**File**: `drg/extract/_signatures.py`, lines 19–55

**Evidence**:

`_relation_schema_for()` (lines 19–30) builds relation entries with only three keys:
```python
{"name": r.name, "source_type": r.src, "target_type": r.dst}
```
`Relation.description` and `Relation.detail` are discarded. `RelationGroup.name` and `RelationGroup.description` are discarded.

`_create_entity_signature()` (lines 32–55) builds:
```python
entity_types_list = [et.name for et in schema.entity_types]
```
`EntityType.description` and `EntityType.examples` are discarded. The DSPy input field is declared as `list[str]`.

**Root cause**: Factories were written for minimal API surface. Schema authors fill in descriptions and examples, but those fields never reach the LLM prompt.

**Severity**: High. The LLM has no basis for distinguishing between entity types with similar names, and no example instances to calibrate extraction. Relation extraction has no description of what each relation means.

**Implementation risk**: Low. Changes are contained to `_signatures.py`. Backward compatible (input data shape grows; no API removed).

---

### VF-2 ✅ CONFIRMED — Signature `OutputField` type annotations are `list[dict]`; `TypedPredictor` output_type is Pydantic

**File**: `drg/extract/_signatures.py`, lines 50–70

**Evidence**:

```python
# All five signatures declare:
entities: list[dict] = dspy.OutputField(desc="Extracted entity mentions")
relations: list[dict] = dspy.OutputField(desc="Extracted relations")
```

`KGExtractor.__init__` (line ~430) wraps these with:
```python
self.entity_extractor = dspy.TypedPredictor(EntitySig, output_type=EntityList)
self.relation_extractor = dspy.TypedPredictor(RelationSig, output_type=RelationList)
```

`EntityList` contains `entities: list[EntityMention]`; `RelationList` contains `relations: list[ExtractedRelation]`. The `ExtractedRelation` model defines `confidence`, `evidence`, `is_negated`, `temporal` — none of which appear in the `list[dict]` annotation used in the signature.

**Root cause**: Signatures were written before the Pydantic output models were finalized. The annotation and the `output_type` were never reconciled.

**Impact**: DSPy generates the JSON schema for the output field from the *signature annotation* (`list[dict]` = generic object array), not from `output_type`. This means the LLM is not prompted with the structure of `ExtractedRelation`. Fields like `confidence`, `evidence`, `is_negated`, and `temporal` are invisible to the LLM and default to `None`.

**Severity**: High. All rich relation metadata fields are structurally hidden from the LLM.

**Implementation risk**: Low. Requires adding one import to `_signatures.py` and changing `list[dict]` to the Pydantic type. No coercion layer changes needed.

---

### VF-3 ✅ CONFIRMED — `context_entities` is built in single-pass loop but never passed to extractor

**File**: `drg/extract/__init__.py`, lines 831–858

**Evidence**:

```python
context_entities: list[tuple[str, str]] = []
for i, chunk in enumerate(chunks):
    ...
    result = extractor(text=chunk_text)         # ← context_entities NOT passed
    chunk_entities = ...
    if enable_cross_chunk_relationships:
        ...
        context_entities.append((name, etype))  # ← accumulates but is never used
```

`KGExtractor.forward()` (line 453) accepts `context_entities: list[tuple[str, str]] | None = None` and merges them into the relation extractor's entity list (lines 480–489). The infrastructure is complete; the call site just doesn't pass the argument.

**Root cause**: The `context_entities` mechanism was designed and implemented in `KGExtractor` but the integration with the main loop was never completed.

**Severity**: High for multi-chunk documents in single-pass mode. Cross-chunk relations are missed because the relation extractor only sees entities extracted from the current chunk.

**Implementation risk**: Very low. One-line fix: `result = extractor(text=chunk_text, context_entities=context_entities if enable_cross_chunk_relationships and context_entities else None)`.

---

### VF-4 ✅ CONFIRMED — Cross-chunk context snippets imported and parameterized but never called

**File**: `drg/extract/__init__.py`, lines 35–36 (imports), line 694 (parameter)

**Evidence**:

The functions are imported:
```python
from ._chunk_context import (
    _build_cross_chunk_context_snippets,
    _select_anchor_entities,
)
```

The controlling parameter exists:
```python
def extract_from_chunks(
    ...
    enable_cross_chunk_context_snippets: bool = True,
    max_cross_chunk_context_chunks: int = 3,
    cross_chunk_snippet_chars: int = 350,
    max_cross_chunk_context_chars: int = 1200,
    ...
```

Search across the entire `extract_from_chunks` body confirms: `_build_cross_chunk_context_snippets` and `_select_anchor_entities` are never called. The parameter `enable_cross_chunk_context_snippets` is accepted but has no effect.

The functions are tested in isolation (`tests/test_cross_chunk_snippets.py`, `tests/test_anchor_selection.py`) and work correctly.

**Root cause**: The snippet mechanism was implemented and tested at the unit level but never wired into the pipeline. The parameter was added as a forward-declaration of intent.

**Severity**: High. The most scalable mechanism for cross-chunk evidence injection is completely inactive. Users who set `enable_cross_chunk_context_snippets=True` (the default) get no benefit.

**Implementation risk**: Medium. Requires building the entity-to-chunks index after Pass 1, then injecting snippets into text before extraction in Pass 2 (per-chunk path) and single-pass path.

---

### VF-5 ❌ PARTIALLY REJECTED — "DefaultConfidenceStrategy not wired to pipeline"

**Review claim**: `DefaultConfidenceStrategy` is not connected to the extraction pipeline.

**Verdict**: This finding is **partially incorrect**. `DefaultConfidenceStrategy` IS used — in `drg/graph/builders.py` line 16 and in `build_enhanced_kg()`. Every call to `build_enhanced_kg()` with `confidence_strategy="default"` (the default value) applies `DefaultConfidenceStrategy`. This is the correct integration point: confidence is a graph-level concern, not an extraction-level concern.

**What IS confirmed**: Confidence scores from `enriched_relations` produced by `extract_from_chunks` are `None` until either (a) the LLM provides them or (b) `build_enhanced_kg` applies the strategy. Since `extract_from_chunks` does not call `build_enhanced_kg`, callers who use the raw tuple output and skip the graph builder do not get confidence-scored data. This is a documentation gap, not an architecture bug.

**Severity**: Low. The architecture is correct. The gap is that `extract_from_chunks` with `return_enriched=True` returns `confidence=None` for all relations, which callers not using `build_enhanced_kg` must handle.

**Action**: Document the intended pipeline. No code change required.

---

### VF-6 ✅ CONFIRMED — `RelationGroup.description` and `RelationGroup.examples` never passed to LLM

**File**: `drg/extract/_signatures.py`, lines 19–30

**Evidence**:

```python
def _relation_schema_for(schema: DRGSchema | EnhancedDRGSchema) -> list[dict[str, str]]:
    if isinstance(schema, EnhancedDRGSchema):
        return [
            {"name": r.name, "source_type": r.src, "target_type": r.dst}  # group info dropped
            for rg in schema.relation_groups
            for r in rg.relations
        ]
```

`rg.name`, `rg.description`, `rg.examples` are silently discarded.

**Severity**: Medium-High. `RelationGroup.examples` are the highest-signal schema data — they provide concrete text-level evidence for what a relation looks like. Discarding them removes the most useful grounding information for the LLM.

**Implementation risk**: Low. Add fields to the relation dict in `_relation_schema_for()`.

---

### VF-7 ✅ CONFIRMED — `EntityGroup` and `PropertyGroup` are declared but entirely unused in extraction

**File**: `drg/schema.py` (definitions), `drg/extract/_signatures.py` and `drg/extract/__init__.py` (never referenced)

**Evidence**:

`EnhancedDRGSchema` stores `entity_groups: list[EntityGroup]` and `property_groups: list[PropertyGroup]`. Neither attribute is referenced in any extraction signature factory or in the extraction pipeline.

`PropertyGroup.properties` is never extracted. No extraction signature declares a property extraction task.

**Severity**: Medium. Property extraction is a documented DRG capability (present in schema, absent in extraction). `EntityGroup` provides semantic grouping that could guide the LLM when choosing between similar entity types — this information is currently wasted.

**Implementation risk for P1**: Low for `RelationGroup` (add to existing dict). Medium for `EntityGroup` (would require restructuring how entity types are passed). High for `PropertyGroup` (requires new signature). Recommend phased approach.

---

### VF-8 ✅ CONFIRMED — Schema relation validation gate missing from `extract_from_chunks`

**File**: `drg/extract/__init__.py`

**Evidence**:

`extract_typed` calls `_filter_against_schema()` at line 1175 — this removes entities with invalid types and relations not in the schema.

`extract_from_chunks` has no equivalent call. All extracted entities and relations, including those violating the schema, flow through unfiltered.

**Severity**: Medium. `extract_from_chunks` (the recommended multi-chunk API) produces schema-violating output that `extract_typed` (single-chunk API) would reject. Users who migrate from single-chunk to multi-chunk mode encounter different output quality.

**Implementation risk**: Low. `_filter_against_schema` is already written. Add one call after the deduplication step in `extract_from_chunks`.

---

### VF-9 ✅ CONFIRMED — Signature docstrings provide no behavioral guidance to the LLM

**File**: `drg/extract/_signatures.py`, lines 46, 62, 77, 91, 107

**Evidence**:

```python
class EntityExtraction(dspy.Signature):
    """Extract entities from text according to the schema."""   # minimal

class RelationExtraction(dspy.Signature):
    """Extract relationships between provided entities under the schema."""  # minimal
```

The comment in `_create_relation_signature` explicitly states:
```python
# Minimal signature: we do NOT request extra metadata fields from the LLM
# (e.g., confidence/temporal/negation). Those are computed by deterministic
# post-processing in `_heuristics`.
```

This explains the design choice: heuristics in `_heuristics.py` were intended to compensate for the LLM not providing these fields. However, this design means `confidence`, `is_negated`, and `temporal` are only populated via a conservative English-only heuristic — not by the LLM's reading of the text.

**Severity**: High. `_heuristics.py` can only detect 17 hardcoded English negation cues; `confidence` always defaults to `None`; temporal extraction is year-only. The LLM has significantly more capability for all three when asked.

**Implementation risk**: Low. Adding behavioral instructions to the signature `__doc__` string and to `OutputField` `desc` strings is non-breaking.

---

### VF-10 ✅ CONFIRMED — `KGExtractor._entity_types` is `list[str]` (names only)

**File**: `drg/extract/__init__.py`, line ~431  
**File**: `drg/extract/_signatures.py`, line 52

**Evidence**:

In `__init__`:
```python
self._entity_types = list(getattr(EntitySig, "_entity_types", []))
```

In `_signatures.py`:
```python
EntityExtraction._entity_types = entity_types_list
# where entity_types_list = [et.name for et in schema.entity_types]
```

`self._entity_types` is passed as the `entity_types` argument:
```python
entity_result = self.entity_extractor(text=text, entity_types=self._entity_types)
```

The LLM receives a list of strings like `["Researcher", "Institution", "Technology"]` with no distinguishing information.

**Severity**: High (same as VF-1, different layer). Confirmed at two levels: schema factory and extractor storage.

---

## Part 2 — Rejected / Qualified Findings

| Finding from Review | Status | Reason |
|---|---|---|
| "DefaultConfidenceStrategy not wired" | ❌ Rejected | It IS wired in `graph/builders.py` via `build_enhanced_kg()`. The architecture is correct. |
| "Two-pass asymmetry requires decoupling" | ⚠️ Qualified | The current design is intentional. Decoupling would be a major refactor with uncertain benefit. Not a correctness bug. |
| "Length-based canonicalization is wrong 5-10% of the time" | ⚠️ Qualified | True, but not calibrated against data. Requires benchmarking before any change. P3 at earliest. |
| "SchemaGeneration not in dspy.Module" | ⚠️ Qualified | True, but the optimizer impact requires measurement. Low-urgency. |
| "REVERSE_RELATION_PATTERNS lookup table is domain-specific" | ⚠️ Qualified | True, but table is opt-in (`enable_reverse_relation_fallback=False` by default). Not a correctness bug in the default path. |
| "Missing span/chunk provenance in output types" | ⚠️ Qualified | Correct architectural observation but requires changes to public API (`ExtractionResult`). Migration risk is high. P3. |

---

## Part 3 — Prioritized Refactor Plan

### P0 — Extraction Correctness Bugs

These are bugs where existing, shipped functionality either silently does nothing or produces lower-quality output than the architecture intended.

| ID | Description | Impact | Complexity | Affected Modules | Migration Risk |
|---|---|---|---|---|---|
| P0-1 | Wire `context_entities` in single-pass extraction loop | High — cross-chunk relations missed | Very Low | `extract/__init__.py` line ~841 | None |
| P0-2 | Wire cross-chunk context snippets (`_build_cross_chunk_context_snippets`) into extraction | High — parameter `enable_cross_chunk_context_snippets` has zero effect | Medium | `extract/__init__.py` | None |
| P0-3 | Pass schema metadata (descriptions, examples) to LLM via signatures | High — LLM has no type/relation guidance | Low | `extract/_signatures.py` | None — input data shape grows |
| P0-4 | Fix `OutputField` type annotations from `list[dict]` to typed Pydantic models | High — LLM never sees `ExtractedRelation` field structure | Low | `extract/_signatures.py` | None |
| P0-5 | Add behavioral instructions to signature docstrings for evidence/negation/confidence | High — heuristics compensate for what LLM could provide directly | Low | `extract/_signatures.py` | None |
| P0-6 | Add schema validation gate to `extract_from_chunks` (parity with `extract_typed`) | Medium — multi-chunk output contains schema-violating triples | Low | `extract/__init__.py` | None |

### P1 — Declarative Architecture Gaps

These are cases where schema constructs exist in `EnhancedDRGSchema` but are not connected to extraction.

| ID | Description | Impact | Complexity | Affected Modules | Migration Risk |
|---|---|---|---|---|---|
| P1-1 | Pass `RelationGroup.description` and `RelationGroup.examples` to relation signatures | High — highest-signal schema data wasted | Low | `extract/_signatures.py` | None |
| P1-2 | Pass `EntityGroup` context to entity extraction (group name + description) | Medium — helps LLM distinguish semantically grouped types | Medium | `extract/_signatures.py`, `extract/__init__.py` | Low |
| P1-3 | Pass `Relation.description` and `Relation.detail` in all relation schemas | High — relation semantics completely absent from LLM prompt | Low | `extract/_signatures.py` | None |

### P2 — Schema Discovery Improvements

| ID | Description | Impact | Complexity | Affected Modules |
|---|---|---|---|---|
| P2-1 | Add schema completeness validation (orphan entity types, disconnected relations) | Medium | Low | `schema.py` |
| P2-2 | Hybrid content-aware sampling for schema generation | Medium | Medium | `extract/_schema_gen.py` |
| P2-3 | Iterative schema coverage loop | High | High | `extract/_schema_gen.py` |

### P3 — Repository Cleanup / Future Research

| ID | Description |
|---|---|
| P3-1 | Add span/chunk provenance to `EntityMention` and `ExtractedRelation` (breaking API change) |
| P3-2 | Calibrate entity resolution thresholds against labeled data |
| P3-3 | Make `SchemaGenerator` a proper `dspy.Module` for optimizer integration |
| P3-4 | Bitemporal documentation for `TemporalInfo` |
| P3-5 | `PropertyGroup` extraction signature |

---

## Part 4 — Implementation Scope for P0 + P1

The following changes will be made in Phase 3:

**`drg/extract/_signatures.py`**
- Add `EntityMention`, `ExtractedRelation` imports from `._types`
- `_relation_schema_for()`: include `description`, `detail` from `Relation`; include `group`, `group_description` from `RelationGroup` (EnhancedDRGSchema only)
- `_create_entity_signature()`: change to `list[dict]` with `name/description/examples`; change output to `list[EntityMention]`; add behavioral instruction to docstring
- All relation signatures: change output to `list[ExtractedRelation]`; add behavioral instruction to docstring

**`drg/extract/__init__.py`**
- Single-pass loop: pass `context_entities` to extractor call when `enable_cross_chunk_relationships=True`
- Build entity-to-chunks index after Pass 1 in two-pass mode (when `enable_cross_chunk_context_snippets=True`)
- Wire snippet injection into Pass 2 per-chunk path and single-pass path
- Add `_filter_against_schema` call in `extract_from_chunks` after deduplication

**Tests to add**:
- `tests/test_signature_metadata.py`: verify signatures carry schema descriptions
- Extend `tests/test_extract_core.py`: verify `enable_cross_chunk_context_snippets=True` uses `_build_cross_chunk_context_snippets`
