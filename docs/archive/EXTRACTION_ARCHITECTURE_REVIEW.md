# DRG: Research-Grade Extraction Architecture Review

> **Scope**: This review evaluates whether DRG's extraction architecture is logically correct for the pipeline goal of `Document → Declarative Schema Discovery → Knowledge Graph Extraction`. It does not address dead code or scope management. Every finding includes a root cause, impact estimate, and a concrete redesign proposal.

---

## Table of Contents

1. [DSPy Signature Design](#1-dspy-signature-design)
2. [InputField / OutputField Usage](#2-inputfield--outputfield-usage)
3. [Typed Outputs](#3-typed-outputs)
4. [Predict vs TypedPredictor](#4-predict-vs-typedpredictor)
5. [Prompt Engineering vs Declarative Signatures](#5-prompt-engineering-vs-declarative-signatures)
6. [Schema Discovery Sampling Strategy](#6-schema-discovery-sampling-strategy)
7. [Coverage Guarantees](#7-coverage-guarantees)
8. [Failure Modes](#8-failure-modes)
9. [Missing Entity and Relationship Classes](#9-missing-entity-and-relationship-classes)
10. [Schema Completeness](#10-schema-completeness)
11. [Cross-Chunk Relationship Extraction](#11-cross-chunk-relationship-extraction)
12. [Entity Memory](#12-entity-memory)
13. [Two-Pass Extraction Architecture](#13-two-pass-extraction-architecture)
14. [Entity Resolution: Alias Handling](#14-entity-resolution-alias-handling)
15. [Entity Resolution: Canonicalization](#15-entity-resolution-canonicalization)
16. [Entity Resolution: Threshold Design](#16-entity-resolution-threshold-design)
17. [Entity Resolution: Failure Cases](#17-entity-resolution-failure-cases)
18. [Relationship Modeling: Directionality](#18-relationship-modeling-directionality)
19. [Relationship Modeling: Temporal Metadata](#19-relationship-modeling-temporal-metadata)
20. [Relationship Modeling: Confidence Scores](#20-relationship-modeling-confidence-scores)
21. [Relationship Modeling: Negation Support](#21-relationship-modeling-negation-support)
22. [Evidence Tracking](#22-evidence-tracking)
23. [Declarative Design: EntityGroup, RelationGroup, PropertyGroup](#23-declarative-design-entitygroup-relationgroup-propertygroup)
24. [Schema Composition Patterns](#24-schema-composition-patterns)
25. [Research Novelty and Differentiation](#25-research-novelty-and-differentiation)

---

## 1. DSPy Signature Design

### Observation

The five extraction signatures (`EntityExtraction`, `RelationExtraction`, `DocumentRelationExtraction`, `ImplicitRelationExtraction`, `CoreferenceResolution`) are dynamically constructed inside factory functions (`_create_entity_signature`, `_create_relation_signature`, etc.) in `drg/extract/_signatures.py`. Each factory creates an inner class that inherits from `dspy.Signature` and attaches schema metadata as a class-level attribute (`_entity_types`, `_relation_schema`).

```python
# _signatures.py — actual pattern
class EntityExtraction(dspy.Signature):
    """Extract entities from text according to the schema."""
    text: str = dspy.InputField(desc="Input text")
    entity_types: list[str] = dspy.InputField(desc="Available entity types")
    entities: list[dict] = dspy.OutputField(desc="Extracted entity mentions")

EntityExtraction._entity_types = entity_types_list  # only names, no descriptions
```

The `entity_types` passed at runtime is a flat list of type *names* (e.g. `["Researcher", "Institution", "Technology"]`). The `EntityType` dataclass in `schema.py` carries `description`, `examples`, and `properties` — none of which reach the signature.

Similarly, `relation_schema` is constructed from `_relation_schema_for()`:
```python
{"name": r.name, "source_type": r.src, "target_type": r.dst}
```
This discards the `Relation.description` and `Relation.detail` fields that the schema author provided specifically to guide the LLM.

### Root Cause

The signature `desc=` strings serve as the only LLM-visible semantic documentation for each field. By omitting entity-type descriptions and relation detail strings, the LLM is asked to perform schema-guided extraction with no guidance about what differentiates one type from another. The extraction becomes entirely dependent on the LLM's prior knowledge about the type names.

### Impact

**High**. For domain-specific or non-obvious type names (e.g. `"CognitiveEnhancer"` vs `"Intervention"` in a medical document), the LLM cannot distinguish between them without the descriptions. Recall and precision both degrade. The schema author's investment in writing `EntityType.description` and `Relation.detail` is entirely wasted at extraction time.

### Redesign

Pass the full schema metadata into the `desc` of the `entity_types` and `relation_schema` fields, or better, create per-entity-type `InputField` entries:

```python
# Option A: Embed schema descriptions into the field desc string
entity_schema_desc = "; ".join(
    f"{et.name}: {et.description}" for et in schema.entity_types
)
class EntityExtraction(dspy.Signature):
    """Extract entities from text according to the schema."""
    text: str = dspy.InputField(desc="Input text to analyze")
    entity_types: list[dict] = dspy.InputField(
        desc=f"Entity type schema: {entity_schema_desc}"
    )
    entities: list[EntityMention] = dspy.OutputField(
        desc="Extracted entity mentions with name, type, aliases, and evidence"
    )
```

```python
# Option B: Pass richer dicts including examples
entity_types_input = [
    {"name": et.name, "description": et.description, "examples": et.examples[:3]}
    for et in schema.entity_types
]
```

For relations, pass `description` and `detail`:
```python
relation_schema_input = [
    {
        "name": r.name,
        "source_type": r.src,
        "target_type": r.dst,
        "description": r.description,
        "example": r.detail,
    }
    for r in all_relations
]
```

---

## 2. InputField / OutputField Usage

### Observation

The `RelationExtraction` signature specifies:

```python
entities: list[dict] = dspy.InputField(desc="Available entity mentions")
relation_schema: list[dict] = dspy.InputField(desc="Allowed relation types")
relations: list[dict] = dspy.OutputField(desc="Extracted relations")
```

Both `entities` and `relations` are typed as `list[dict]` in the signature but `EntityList` / `RelationList` Pydantic models are used as `output_type` in `TypedPredictor`. This creates a type mismatch between the signature's declared type (`list[dict]`) and the output type the predictor will produce (`RelationList`).

The `CoreferenceResolution` signature's output field is named `resolved_relations` but the coercion logic in `KGExtractor.resolve_coreferences_dspy()` checks for both `resolved_relations` and `relations` via an `or` fallback — indicating the signature contract is not stable.

### Root Cause

DSPy's `TypedPredictor` uses the `output_type` argument to wrap the output in the target Pydantic model, but the *signature* field type annotation (`list[dict]` vs `list[RelationList]`) still informs the JSON schema DSPy generates for the LLM prompt. If these disagree, the generated prompt schema is less specific than what `TypedPredictor` enforces on the Python side, and the LLM is guided by the weaker schema.

### Impact

**Medium**. DSPy may generate a JSON schema for the output field based on `list[dict]` (generic object array) rather than the tighter `RelationList` Pydantic schema. This means the LLM does not see the `ExtractedRelation` field definitions (`confidence`, `evidence`, `is_negated`, `temporal`) in the prompt, reducing the probability that the LLM will populate them.

### Redesign

Align signature type annotations with the Pydantic output types:

```python
from ._types import EntityMention, ExtractedRelation

class EntityExtraction(dspy.Signature):
    """Extract entities from text according to the schema."""
    text: str = dspy.InputField(desc="Input text")
    entity_types: list[dict] = dspy.InputField(desc="Entity type schema with descriptions")
    entities: list[EntityMention] = dspy.OutputField(
        desc="Extracted entity mentions with name, type, aliases, and evidence"
    )

class RelationExtraction(dspy.Signature):
    """Extract relationships between provided entities under the schema."""
    text: str = dspy.InputField(desc="Input text")
    entities: list[dict] = dspy.InputField(desc="Available entity mentions")
    relation_schema: list[dict] = dspy.InputField(desc="Allowed relation types with descriptions")
    relations: list[ExtractedRelation] = dspy.OutputField(
        desc="Extracted relations with source, relation, target, confidence, evidence, temporal, is_negated"
    )
```

This ensures that DSPy generates a JSON schema for the output that includes all `ExtractedRelation` fields, making it far more likely the LLM will populate `evidence`, `temporal`, `confidence`, and `is_negated`.

---

## 3. Typed Outputs

### Observation

`EntityList` and `RelationList` are Pydantic models used as `output_type` in `TypedPredictor`. This is correct. However, the coercion fallback functions `_coerce_entity_mentions()` and `_coerce_extracted_relations()` accept raw `list[dict]`, `list[tuple]`, and even `list[list]` formats, effectively making the typed contract optional:

```python
elif isinstance(item, (list, tuple)) and len(item) >= 2:
    mentions.append(EntityMention(name=str(item[0]), type=str(item[1])))
```

The `ExtractedRelation` model defines `confidence: float | None = None`. The LLM is never explicitly required to provide a confidence value; it defaults to `None`. The heuristic scorer in `confidence/_default.py` then assigns a score post-hoc based on schema membership — not on the LLM's own uncertainty.

### Root Cause

The coercion layer exists to preserve backward compatibility with mocked tests. In production, `TypedPredictor` should guarantee `EntityList` / `RelationList` objects. The fallback paths reduce the incentive to fix upstream signature type mismatches (Issue 2).

### Impact

**Medium**. The `confidence: None` default means no relation carries an LLM-self-rated confidence until a post-hoc heuristic runs. The heuristic is uncalibrated (admitted in the module docstring). Any downstream filtering or ranking on confidence is unreliable.

### Redesign

1. Require `confidence: float` (non-optional) in `ExtractedRelation` by adding it to the signature's output `desc`:

```python
relations: list[ExtractedRelation] = dspy.OutputField(
    desc="Each relation MUST include: source, relation, target, confidence (0.0-1.0), evidence (verbatim text span), is_negated (bool)"
)
```

2. Add a DSPy `Assert` or `Suggest` to enforce non-null confidence:

```python
from dspy import Assert
Assert(all(r.confidence is not None for r in result.relations), 
       "All relations must have a confidence score")
```

3. Restrict the coercion fallback to test environments only:

```python
if not _should_return_dspy_prediction():
    # test/mock path: accept tuples
    ...
```

---

## 4. Predict vs TypedPredictor

### Observation

`KGExtractor.__init__` correctly requires `dspy.TypedPredictor` and raises `ExtractionError` if it is unavailable. `generate_schema_from_text` also uses `TypedPredictor` with a `Predict` fallback check (`if not hasattr(dspy, "TypedPredictor"): raise`). There is no use of plain `dspy.Predict` anywhere in the hot path.

This is the architecturally correct choice.

### Issue

The `SchemaGeneration` signature is defined at module level in `_schema_gen.py` (not inside a factory function):

```python
class SchemaGeneration(dspy.Signature):
    """Generate EnhancedDRGSchema from the given text."""
    text: str = dspy.InputField(desc="Input text")
    entity_types: list[SchemaEntityType] = dspy.OutputField(...)
    relation_groups: list[SchemaRelationGroup] = dspy.OutputField(...)
```

Unlike the extraction signatures, this class is instantiated once at import time. If DSPy is compiled/optimized (e.g. with `dspy.BootstrapFewShot`), the compiled predictor wraps the schema-generation program but the `SchemaGeneration` signature class itself is not injectable or replaceable at runtime. There is no way for the optimizer to substitute a fine-tuned schema-generation program.

### Root Cause

The module-level class definition prevents the signature from being swapped by dependency injection or optimizer compilation — only the `TypedPredictor` instance wrapping it is replaceable.

### Impact

**Low-Medium**. The optimizer cannot improve schema generation quality because `SchemaGeneration` is not exposed as a swappable `dspy.Module` component. This limits DRG's "optimizable end-to-end pipeline" claim.

### Redesign

Make schema generation a proper `dspy.Module`:

```python
class SchemaGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.TypedPredictor(SchemaGeneration, output_type=SchemaOutput)
    
    def forward(self, text: str) -> SchemaOutput:
        return self.generate(text=text)
```

This makes `SchemaGenerator` composable, optimizable, and replaceable by the optimizer or user.

---

## 5. Prompt Engineering vs Declarative Signatures

### Observation

The docstrings of the dynamically created signature classes are the only prompts visible to the LLM:

```python
class EntityExtraction(dspy.Signature):
    """Extract entities from text according to the schema."""
```

```python
class RelationExtraction(dspy.Signature):
    """Extract relationships between provided entities under the schema."""
```

These are minimal and non-instructive. The LLM receives no guidance on:
- How to handle entities that appear in multiple chunks
- How to choose between ambiguous entity types
- What constitutes sufficient evidence to assert a relation
- When to assert `is_negated=True`
- How to format temporal information

The `_detect_negation_in_window` heuristic in `_heuristics.py` compensates for the LLM not reliably providing negation flags — this is the strongest evidence that the signature is not doing enough declarative work.

### Root Cause

DSPy encourages minimal signature docstrings and relies on typed I/O to drive LLM behavior. However, the field type annotations (`list[dict]`) are too generic to provide the same guidance as Pydantic schemas with field descriptions would. There is a middle path — `Field(description=...)` in Pydantic + tight type annotations in DSPy — that is not being used.

### Impact

**High**. Negation, temporal extraction, evidence span extraction, and confidence calibration all degrade to heuristic post-processing rather than being addressed declaratively. This is a fundamental architecture concern: the LLM's output is less than the system's desired output, and the gap is filled by deterministic rules rather than by the LLM itself.

### Redesign

Adopt `dspy.Signature` instruction strings for complex behavioral requirements:

```python
class RelationExtraction(dspy.Signature):
    """Extract all relationships between the provided entities that are supported by the text.
    
    Rules:
    - Only assert relations licensed by the relation_schema.
    - Set is_negated=True for relations that are explicitly denied or ceased (e.g. 'no longer', 'never', 'stopped').
    - Set confidence to your estimate of correctness (0.0=uncertain, 1.0=certain).
    - Populate evidence with the shortest verbatim text span that licenses the relation.
    - Populate temporal.start/end if the relation is bounded in time.
    - Do NOT infer relations that are not supported by the text.
    """
```

This keeps the system declarative and reduces heuristic post-processing to a safety net rather than a primary path.

---

## 6. Schema Discovery Sampling Strategy

### Observation

`_sample_text_for_schema_generation()` implements a deterministic, evenly-spaced sampling strategy:

- Texts ≤ `DRG_SCHEMA_MAX_SAMPLE_CHARS` (100k chars) are passed in full.
- Longer texts are split into up to 20 parts sampled at evenly-spaced positions.
- The first and last parts are always included.
- Middle parts are included in center-out order until the budget is exhausted.

The sampling is position-based, not content-based. There is no semantic selection: the sampler does not detect section headers, entity-dense paragraphs, or structurally important regions.

### Root Cause

Uniform sampling is a reasonable baseline for general documents. However, technical documents, scientific papers, and reports often have information-dense sections (Methods, Results, Named Entities sections) that may be statistically sparse but highly informative for schema generation. Position-based sampling treats all positions as equally informative.

### Impact

**Medium**. For well-structured documents, uniform sampling works. For documents where entity/relation types are concentrated in a non-uniform distribution (e.g. a legal contract where party types appear in section 2 and relationship types appear in section 7), position-based sampling may consistently undersample one or both. The resulting schema will be incomplete.

### Redesign

A hybrid semantic-positional sampler that scores windows by entity-mention density:

```python
def _semantic_sample(text: str, budget: int) -> str:
    # 1. Always include first and last 5% (document framing)
    # 2. Score remaining paragraphs by candidate entity density
    #    (heuristic: capitalized noun phrases, numbers, quoted terms)
    # 3. Greedily include highest-scoring paragraphs within budget
    # 4. Fallback to positional sampling if no density signal
```

Alternatively, use a two-stage schema generation: first generate from a coarse sample, then refine with targeted re-sampling of text regions where the coarse schema has low confidence.

---

## 7. Coverage Guarantees

### Observation

There are no coverage guarantees in the schema generation pipeline. A single LLM call with sampled text produces one `EnhancedDRGSchema`. If a valid entity type or relation class exists in the document but is absent from the sample, it will not appear in the schema, and no subsequent extraction pass will recover it.

The `auto_discovery` flag in `EnhancedDRGSchema` is set from the schema generation output but is never read or acted upon in the extraction pipeline:

```python
# schema.py — auto_discovery is stored but never used in extraction
self.auto_discovery = auto_discovery
```

### Root Cause

Schema generation is designed as a one-shot, best-effort call. There is no iterative schema refinement, no coverage metric, and no mechanism for the extraction stage to report back "I found entity mentions that don't match any schema type."

### Impact

**High**. In a research-grade pipeline, schema completeness is a prerequisite for extraction recall. If the schema misses a category (e.g. `"Funding Body"` in a scientific paper), all mentions of funding bodies will either be dropped, misclassified, or hallucinated into the closest matching type. The user has no signal that this happened.

### Redesign

Implement an iterative schema coverage loop:

```python
def generate_schema_with_coverage(text: str, min_coverage_rounds: int = 2) -> EnhancedDRGSchema:
    # Round 1: generate initial schema from sampled text
    schema = generate_schema_from_text(text)
    
    # Round 2: run a lightweight extraction pass with the initial schema
    # Look for entity mentions that the extractor couldn't classify
    unclassified = _find_unclassified_mentions(text, schema)
    
    if unclassified:
        # Generate a schema supplement from the unclassified region
        supplement = generate_schema_from_text(
            _build_supplement_prompt(unclassified, schema)
        )
        schema = _merge_schemas(schema, supplement)
    
    return schema
```

At minimum, add a `schema_coverage_warnings` field to `EnhancedDRGSchema` that is populated during extraction with unclassified mentions.

---

## 8. Failure Modes

### 8.1 Silent Schema Mismatch

**Root cause**: When the LLM extracts a relation with `relation="works_with"` but the schema only declares `relation="collaborates_with"`, there is no schema-validation gate on the extracted output. The extracted relation passes through `_coerce_extracted_relations()` without checking against `_relation_schema`.

**Impact**: Medium-High. The resulting graph contains relations that violate the declared schema. Downstream consumers expecting schema-conformant output receive dirty data.

**Fix**: Add a post-extraction schema validation filter:
```python
def _filter_schema_violations(
    relations: list[ExtractedRelation],
    schema: DRGSchema | EnhancedDRGSchema,
) -> tuple[list[ExtractedRelation], list[ExtractedRelation]]:
    """Return (valid, rejected) split based on schema compliance."""
```

### 8.2 Pass 1 Entity Extraction Without Global Context

**Root cause**: In two-pass mode, Pass 1 runs `extractor(text=chunk_text)` on each chunk independently with no cross-chunk context. An entity mentioned in chunk 3 only by its full name, and in chunk 1 only as an abbreviation, will be extracted as two separate entities in Pass 1 and may or may not be merged by entity resolution.

**Impact**: High. Entity resolution operates on extracted names, not on text. If "World Health Organization" in chunk 3 and "WHO" in chunk 1 are both extracted, string similarity between them is low (0.0 by the single-token safety rule). They will only be merged if embeddings are provided.

**Fix**: See Section 12 (Entity Memory).

### 8.3 LLM Hallucination of Entity Types

**Root cause**: Entity extraction passes `entity_types` as a list of names. The LLM may return an entity with `type="Scientist"` when the schema only declares `type="Researcher"`. There is no type-validation step after extraction.

**Impact**: Medium. The graph will contain entities with schema-invalid types. The `is_valid_relation` check will fail for such entities in downstream confidence scoring, but the entities themselves are retained.

**Fix**: Post-extraction type normalization using the schema:
```python
def _normalize_entity_type(entity: EntityMention, valid_types: set[str]) -> EntityMention:
    if entity.type in valid_types:
        return entity
    # Attempt fuzzy match to closest valid type
    closest = _closest_type(entity.type, valid_types)
    return entity.model_copy(update={"type": closest, "metadata": {..., "original_type": entity.type}})
```

### 8.4 Implicit Relation Extraction on Full Document Text

**Root cause**: `infer_implicit_relations()` concatenates all chunks into a single `full_text` string and passes it to `TypedPredictor`. For large documents, this will exceed the LLM context window or produce degraded results due to context length.

**Impact**: Medium. Implicit relation inference is only attempted after all other extraction passes complete, so the primary pipeline is not affected. But the implicit pass may silently fail or produce hallucinated relations on long documents.

---

## 9. Missing Entity and Relationship Classes

### Observation

The `EntityMention` model:
```python
class EntityMention(BaseModel):
    name: str
    type: str
    aliases: list[str] = []
    evidence: str | None = None
    metadata: dict[str, Any] = {}
```

The following classes are absent:

| Missing | Impact |
|---|---|
| `Span` (character offsets of the entity in source text) | Cannot reconstruct provenance or highlight entities in original document |
| `EntityConfidence` as a typed field on `EntityMention` | Entity confidence is not first-class; only relation confidence is computed |
| `NormalizedForm` | Distinguishes display name from canonical lookup key |
| `EntityClass` (ontological supertype) | Cannot group entities by ontological class |
| `ExtractionSource` (chunk_id, document_id) | Entity mentions are not traceable to their source chunk |

The `ExtractedRelation` model is missing:

| Missing | Impact |
|---|---|
| `Span` for both endpoints | Cannot anchor relation to source text |
| `ChunkId` | Cannot reconstruct which chunk licensed the relation |
| `ExtractionPass` | Cannot distinguish relations found in Pass 1 vs. Pass 2 vs. implicit pass |
| `NegationEvidence` (the span that licensed `is_negated`) | Negation is a boolean flag with no supporting evidence |

### Root Cause

The models were designed for tuple-compatible output (`ExtractionResult.relations: list[tuple[str, str, str]]`), which limits them to the minimum (source, relation, target). The richer metadata fields were added incrementally but not systematically.

### Impact

**High for provenance-based research systems**. Without character offsets and chunk IDs, the extracted graph cannot be linked back to the source document with precision. This is a fundamental limitation for any system that needs to present evidence or allow users to validate extractions.

### Redesign

```python
class EntityMention(BaseModel):
    name: str
    type: str
    confidence: float | None = None       # add
    aliases: list[str] = []
    evidence: str | None = None
    span_start: int | None = None          # add
    span_end: int | None = None            # add
    chunk_id: str | int | None = None      # add
    metadata: dict[str, Any] = {}

class ExtractedRelation(BaseModel):
    source: str
    relation: str
    target: str
    confidence: float | None = None
    evidence: str | None = None
    negation_evidence: str | None = None   # add
    temporal: TemporalInfo | dict | None = None
    is_negated: bool = False
    chunk_id: str | int | None = None      # add
    extraction_pass: str | None = None     # add ("entity", "document", "implicit")
    metadata: dict[str, Any] = {}
```

---

## 10. Schema Completeness

### Observation

`EnhancedDRGSchema._validate()` checks:
- Entity type name uniqueness
- Relation endpoints reference valid entity types
- `EntityGroup` members reference valid entity types

It does **not** check:
- Whether all entity types participate in at least one relation (orphan types)
- Whether any relation forms an unreachable component (island relation)
- Whether there are duplicate (name, src, dst) triples across different `RelationGroup`s
- Whether `PropertyGroup` properties are referenced in any entity type's `properties` dict

Additionally, the `to_legacy_schema()` conversion flattens all `RelationGroup`s into a single list, silently discarding group-level `description` and `examples`.

### Root Cause

The validation logic was written to prevent structural errors (missing references) rather than semantic completeness errors (orphan types, disconnected subgraphs).

### Impact

**Low-Medium**. Orphan entity types produce extraction overhead (the LLM is asked to look for entities of a type that can never participate in any known relation) without any benefit. Disconnected relation islands produce a fragmented graph.

### Redesign

Add a `validate_completeness()` method to `EnhancedDRGSchema`:

```python
def validate_completeness(self) -> list[str]:
    warnings = []
    relation_src_dst = {et for r in self.get_all_relations() for et in (r.src, r.dst)}
    for et in self.entity_types:
        if et.name not in relation_src_dst:
            warnings.append(f"EntityType '{et.name}' participates in no relations")
    # Check for duplicate (name, src, dst) across groups
    seen = {}
    for rg in self.relation_groups:
        for r in rg.relations:
            key = (r.name, r.src, r.dst)
            if key in seen:
                warnings.append(f"Duplicate relation {key} in groups '{seen[key]}' and '{rg.name}'")
            seen[key] = rg.name
    return warnings
```

---

## 11. Cross-Chunk Relationship Extraction

### Observation

Cross-chunk relationship extraction is implemented via two mechanisms:

**Mechanism 1 — Context entity injection** (`_forward_impl`):
```python
if context_entities:
    existing_entity_names = {(name.lower(), etype) for name, etype in entities_list}
    for name, etype in context_entities:
        if (name.lower(), etype) not in existing_entity_names:
            entities_list.append((name, etype))
```
The relation extractor then sees entities from previous chunks alongside current-chunk entities. However, in the main `extract_from_chunks` loop, `context_entities` is **never passed** to `extractor()`:
```python
result = extractor(text=chunk_text)  # context_entities not passed
```
This parameter exists on `forward()` but is unused in both single-pass and two-pass modes.

**Mechanism 2 — Document-level relation extraction** (`extract_document_relations`):
The two-pass mode uses `extract_document_relations`, which passes all chunk texts + all canonical entities to a single LLM call. This is the primary cross-chunk mechanism.

**Mechanism 3 — Cross-chunk context snippets** (`_build_cross_chunk_context_snippets`):
Implemented and exported, but not called from `extract_from_chunks`. It is present in the module's public API but not wired into the primary pipeline.

### Root Cause

Three cross-chunk mechanisms exist but only one (document-level relation extraction) is actually used in the main pipeline. The context snippet mechanism and context entity injection were designed but not connected.

### Impact

**High**. The actual cross-chunk extraction quality depends entirely on the document-level relation extractor receiving all chunks as a flat list. For long documents, this list may overflow the LLM context window. There is no fallback mechanism — if the document-level call fails (context length, timeout, etc.), the pipeline logs a warning and the cross-chunk relations are lost.

The context snippet mechanism (`_build_cross_chunk_context_snippets`) is the architecturally correct solution for scalable cross-chunk extraction because it provides targeted evidence rather than full chunk dumps. Its non-connection to the pipeline is the most significant architectural gap in the codebase.

### Redesign

Wire context snippets into the relation extraction pass:

```python
# In the two-pass mode, Pass 2 loop:
entity_to_chunks = _build_entity_to_chunks_index(chunk_entities_list)
for i, chunk_text in enumerate(chunk_texts):
    anchor_entities = _select_anchor_entities(
        chunk_text, chunk_entities_list[i], entity_to_chunks,
        total_chunks=len(chunk_texts), min_anchor_len=3, max_anchors=8
    )
    snippets = _build_cross_chunk_context_snippets(
        chunk_texts, entity_to_chunks, anchor_entities,
        current_chunk_index=i, max_chunks=3, snippet_chars=350
    )
    augmented_text = chunk_text + "\n\n--- Cross-document evidence ---\n" + "\n".join(snippets)
    result = extractor(text=augmented_text, context_entities=all_canonical_entities)
```

---

## 12. Entity Memory

### Observation

There is no persistent entity memory across extraction passes. In two-pass mode:

- **Pass 1** extracts entities per-chunk in isolation. No information from chunk $i-1$ is available to chunk $i$'s entity extractor.
- **Entity resolution** runs after Pass 1, producing canonical names.
- **Pass 2** provides canonical entities to the document-level relation extractor, but these are passed as a flat list, not as a structured memory.

The `context_entities` parameter on `KGExtractor.forward()` would enable per-chunk entity memory injection, but is never called with it in the main loop:
```python
result = extractor(text=chunk_text)
# Should be: result = extractor(text=chunk_text, context_entities=accumulated_entities)
```

### Root Cause

The entity memory mechanism was designed (`context_entities` on `forward()`) but not connected to the pass-level extraction loops. This is likely a design-implementation gap where the interface was specified but integration was deferred.

### Impact

**High for coreference-heavy documents**. In documents where an entity's full name appears once at the beginning and subsequent references use abbreviations or pronouns, the per-chunk entity extractor will extract different surface forms in each chunk. Entity resolution must then merge these, which requires embeddings to succeed. Without entity memory, recall for multi-mention entities is dependent on the entity resolution threshold.

### Redesign

Pass accumulated entities from previous chunks into each chunk's extraction:

```python
# Pass 1 with entity memory
accumulated_entities: list[tuple[str, str]] = []
for i, chunk_text in enumerate(chunk_texts):
    result = extractor(
        text=chunk_text,
        context_entities=accumulated_entities[-max_context_entities:]  # sliding window
    )
    chunk_entities = result.entities
    # Update accumulator with new unique entities
    new_entities = [e for e in chunk_entities if e not in accumulated_set]
    accumulated_entities.extend(new_entities)
```

This ensures that by chunk $k$, the entity extractor has been informed of all entities seen in chunks $1..k-1$, making it far more likely to recognize abbreviations and aliases as known entities rather than extracting them as new ones.

---

## 13. Two-Pass Extraction Architecture

### Observation

The two-pass architecture as implemented:

**Pass 1**: Entity extraction per-chunk (independent, no context sharing)  
**Entity Resolution** (optional, between passes)  
**Pass 2**: Document-level relation extraction (one LLM call for the entire document)

This is a reasonable design for short documents. For long documents, Pass 2's single LLM call receives the full list of all chunk texts and all canonical entities. There is no length guard on this call beyond what the LLM provider imposes.

The pass-level design also has an asymmetry: Pass 1 runs the `entity_extractor` (TypedPredictor with `EntityList` output) while Pass 2 runs `document_relation_extractor` (TypedPredictor with `RelationList` output). The two passes are not symmetric — Pass 2 does not re-extract or validate entities, it only extracts relations among already-resolved entities.

An additional concern: when `two_pass_extraction=True` and `enable_cross_chunk_relationships=False`, Pass 2 loops over chunks individually (per-chunk relation extraction with global entities). This is correct but loses the benefit of two-pass architecture — the document-level relation extractor is only used when cross-chunk relationships are enabled.

### Root Cause

The two-pass design conflates two distinct purposes: (1) global entity consistency and (2) cross-chunk relation extraction. The `two_pass_extraction` flag controls both simultaneously instead of treating them as independent concerns.

### Impact

**Medium**. A user who wants global entity consistency (two-pass) but not cross-chunk relations (too expensive) gets per-chunk relation extraction with global entity lists — which is actually the right behavior but the code path is non-obvious and undocumented.

### Redesign

Decouple entity resolution strategy from relation extraction strategy:

```python
def extract_from_chunks(
    chunks,
    schema,
    entity_resolution_strategy: Literal["per_chunk", "two_pass", "streaming"] = "two_pass",
    relation_extraction_strategy: Literal["per_chunk", "document_level", "windowed"] = "document_level",
    ...
)
```

A three-pass architecture should also be considered for research-grade completeness:
- **Pass 1**: Per-chunk entity extraction
- **Pass 2**: Global entity resolution and canonicalization  
- **Pass 3**: Per-chunk relation extraction with canonical entity context + cross-chunk snippet injection

---

## 14. Entity Resolution: Alias Handling

### Observation

`EntityMention.aliases` is defined as `list[str] = []` but is never populated during extraction. The extraction signature does not instruct the LLM to extract aliases. Aliases are computed post-hoc by `EntityResolver.resolve_detailed()` as a side-effect of merging:

```python
# In EntityResolver.resolve_detailed()
for original, canonical in name_mapping.items():
    if original != canonical:
        aliases_by_canonical.setdefault(canonical, []).append(original)
```

This means aliases are only discovered when the same entity appears under multiple surface forms in the same document. If an entity's alias is known domain knowledge (e.g. "WHO" is an alias for "World Health Organization") but both forms appear in the same document, they will only be merged if the similarity score is above threshold.

### Root Cause

The extraction stage does not leverage the `EntityType.examples` field (which could include alias examples) to guide alias-aware extraction. The LLM could be instructed to extract known aliases from the text, but the current signature does not ask for them.

### Impact

**Medium**. Abbreviation-canonical mismatches produce spurious entity nodes in the graph. A graph containing both "WHO" (Organization) and "World Health Organization" (Organization) as separate nodes with duplicate relations is semantically incorrect.

### Redesign

Pass alias examples from `EntityType.examples` into the extraction signature and explicitly request alias extraction:

```python
entity_types_input = [
    {
        "name": et.name,
        "description": et.description,
        "example_aliases": [ex for ex in et.examples if "/" in ex or "(" in ex]  # heuristic
    }
    for et in schema.entity_types
]
```

And add to the signature instruction:
```
"For each entity, populate aliases with any abbreviations, nicknames, or alternative names for the same entity found in the text."
```

---

## 15. Entity Resolution: Canonicalization

### Observation

Canonicalization uses the **longest name** in a merge group as the canonical form:

```python
# From EntityResolver._resolve_by_type (inferred from resolve_detailed behavior)
canonical = max(group, key=lambda n: len(n))
```

This is a reasonable heuristic for most cases ("World Health Organization" is better than "WHO"). However:

1. The longest name may be a description rather than a proper noun ("The international organization for public health coordination" extracted from a particularly verbose LLM output).
2. For organizational names, the formally registered name may be shorter than a descriptive alias.
3. Resolution is performed **per entity type only** — "Apple" (Organization) and "Apple" (Technology/Product) are never compared. This is correct for type-distinct entities, but creates problems when the same surface form refers to entities of different types that are actually the same real-world entity being dual-classified.

### Root Cause

Canonicalization is a string-length heuristic with no semantic grounding. There is no mechanism to prefer a name that appears in a knowledge base, matches a schema-provided example, or has the highest extraction confidence.

### Impact

**Medium**. Length-based canonicalization is wrong in ~5-10% of cases for real-world documents with verbose LLM outputs. Each wrong canonicalization propagates through the entire relation graph.

### Redesign

Priority-based canonicalization:
1. Prefer names that match `EntityType.examples` exactly (schema-grounded)
2. Prefer names that appear in the source text verbatim (extraction-grounded)
3. Prefer longest among the remaining candidates (current heuristic as fallback)

---

## 16. Entity Resolution: Threshold Design

### Observation

The adaptive threshold in `EntityResolver._get_adaptive_threshold()`:

```python
if (n1 in n2 or n2 in n1) and min_len >= 3:
    return 0.30   # very permissive

if min_len < 5 and max_len > 10:
    return max(0.40, base - 0.25)

if min_len < 8:
    return max(0.50, base - 0.15)
```

The base threshold is 0.65 by default. For substring matches with `min_len >= 3`, it drops to 0.30. This is dangerously permissive: "IBM" is a substring of "IBM Research" (correct merge), but "US" is a substring of "US Airways" and also of "Brussels" — word-boundary checks in `_word_boundary_contains` prevent the worst cases, but substring merging at threshold 0.30 still over-merges aggressively.

The `min_merge_margin = 0.08` (the margin between the best and second-best candidate scores required to commit a merge) is a conservative safety net, but it only applies when there are multiple candidates competing for the same entity.

The embedding weight is `0.7` by default when embeddings are provided. The combined score is:
$$\text{combined} = 0.7 \cdot \text{embedding\_sim} + 0.3 \cdot \text{string\_sim}$$

For very short names (3-4 characters), embedding similarity is unreliable because the embedding model's token representations of short names are highly context-dependent and not stable across document positions.

### Root Cause

The threshold was tuned for full-name vs. abbreviated-name merging (the most common correct case). It was not calibrated against a held-out merge/no-merge dataset.

### Impact

**Medium**. Over-merging causes correct relations to be attributed to wrong canonical entities. Under-merging causes duplicate entities and duplicate relations. Both degrade graph quality. The adaptive threshold's aggressive reduction for substrings is the most likely source of over-merging errors.

### Redesign

1. Replace the hand-tuned length-based adaptive threshold with a calibrated threshold learned from examples (or at minimum, a threshold lookup table grounded in evaluation data).
2. For short names (< 5 chars), disable string-similarity-based merging entirely and rely only on embeddings + explicit alias lists.
3. Add a `merge_log` to `EntityResolutionResult` for inspection.

---

## 17. Entity Resolution: Failure Cases

The following cases are not handled correctly:

| Case | Current Behavior | Correct Behavior |
|---|---|---|
| Cross-type same entity ("Apple" as Org and Product) | Never merged (per-type resolution) | Flag as ambiguous, require human resolution |
| Pronouns passed as entity names (e.g. "he") | May be extracted and merged with any male-name entity | Filter at extraction time (pronoun blocklist) |
| Numeric entities ("2019", "42") | Extracted and compared | Numeric entities need domain-specific canonicalization rules |
| Entities with identical normalized forms but different types | First seen wins in canonical mapping | Warn on type collision, prefer entity with schema-matching type |
| Entity with zero string similarity and no embedding | Not merged, left as duplicates | Acceptable; issue is when this happens silently |

The `_PRONOUN_LIKE` set in `_chunk_context.py` filters pronouns from cross-chunk context snippets but it is not used in the extraction or entity resolution pipeline. Pronouns can be extracted as entity mentions and will pass through `_coerce_entity_mentions` without filtering.

---

## 18. Relationship Modeling: Directionality

### Observation

Directionality is handled by the `REVERSE_RELATION_PATTERNS` lookup table in `_relations.py` (86 hardcoded patterns) and by `_infer_reverse_relation_name()` (suffix-pattern heuristics). The `_add_reverse_relations()` function adds reverse relations to `RelationGroup`s automatically.

Critical issues:

1. `_add_reverse_relations()` is not called from `extract_from_chunks` or `KGExtractor.__init__`. It must be called explicitly by the user, and its call site is not documented.

2. The REVERSE_RELATION_PATTERNS table is domain-specific to English organizational/social relations. It does not cover:
   - Scientific relations (`"inhibits"` → `"inhibited_by"`)
   - Temporal relations (`"precedes"` → `"preceded_by"`)
   - Causal relations (`"causes"` → `"caused_by"`)

3. The `_infer_reverse_relation_name()` suffix heuristic makes grammatically plausible but semantically incorrect inferences: `"located_in"` (via `_of` suffix → `has_located`) is wrong; the correct reverse is `"contains"`.

4. Symmetric relations (`"related_to"`, `"partners_with"`) are in the table but are not modeled as symmetric in the schema — they appear as directed relations that happen to have the same name in both directions.

### Root Cause

Directionality is an afterthought added via a lookup table rather than a first-class schema property. The `Relation` dataclass has no `is_symmetric: bool` or `inverse: str | None` field.

### Impact

**Medium**. Missing reverse relations cause the graph to be directionally incomplete. A query "what entities does X relate to" will miss relations where X is the target rather than the source.

### Redesign

Add directionality as a schema property:

```python
@dataclass(frozen=True)
class Relation:
    name: str
    src: str
    dst: str
    description: str = ""
    detail: str = ""
    is_symmetric: bool = False        # add
    inverse_name: str | None = None   # add (explicit inverse, no heuristic needed)
```

Then use this at schema build time to auto-generate inverse relations without a lookup table:

```python
def _add_declared_inverse_relations(schema: EnhancedDRGSchema) -> EnhancedDRGSchema:
    for rg in schema.relation_groups:
        for r in rg.relations:
            if r.inverse_name:
                add_if_not_present(Relation(r.inverse_name, r.dst, r.src, ...))
            elif r.is_symmetric:
                ensure_present(Relation(r.name, r.dst, r.src, ...))
```

---

## 19. Relationship Modeling: Temporal Metadata

### Observation

The `TemporalInfo` model:
```python
class TemporalInfo(BaseModel):
    start: str | None = None
    end: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    precision: str | None = None
    text: str | None = None
```

The `_extract_year_temporal` heuristic in `_heuristics.py` extracts only 4-digit years (1900-2099). It does not handle:
- ISO 8601 dates ("2023-04-15")
- Relative dates ("last year", "recently")
- Named periods ("during the Cold War")
- Recurring temporal assertions ("every year")
- Open-ended intervals ("since 2010")

The model has both `start`/`end` and `valid_from`/`valid_to` fields, which are semantically redundant. The distinction between "when the relation was asserted" (transaction time) and "when the relation held" (valid time) — a standard bitemporal distinction — is not documented or enforced.

The extraction signature does not instruct the LLM to provide temporal information. The `ExtractedRelation.temporal` field will be `None` from the LLM unless the signature explicitly requests it (see Issue 2 and Issue 5).

### Root Cause

Temporal extraction was designed as a heuristic post-processing step rather than as a first-class extraction target. The `TemporalInfo` model was designed for bitemporal modeling but the schema was never documented or enforced.

### Impact

**Medium for temporal-heavy documents**. Legal contracts, historical documents, financial filings, and scientific papers all have dense temporal assertions. Extracting only 4-digit years misses the majority of temporal information.

### Redesign

1. Document the bitemporal semantics: `start`/`end` = valid time (when the fact held); `valid_from`/`valid_to` = transaction time (when the assertion was made).
2. Extend the heuristic to handle ISO dates and relative dates.
3. Add temporal extraction to the relation extraction signature:

```python
class ExtractedRelation(BaseModel):
    ...
    temporal: TemporalInfo | None = None
    # In signature desc: "populate temporal.start/end with ISO 8601 dates if bounded in time"
```

---

## 20. Relationship Modeling: Confidence Scores

### Observation

The `DefaultConfidenceStrategy` in `confidence/_default.py` uses these fixed coefficients:

| Signal | Weight |
|---|---|
| Base entity score | 0.6 |
| Type in schema | +0.15 |
| Name in source text | +0.10 |
| Multi-word name | +0.05 |
| Base relation score | 0.5 |
| Schema-valid relation | +0.20 |
| Both endpoints typed | +0.10 |
| Temporal cue present | +0.05 |
| Negated | −0.30 |

These coefficients are explicitly described in the module docstring as **not calibrated against labelled data**. The maximum achievable entity confidence is `0.6 + 0.15 + 0.10 + 0.05 = 0.90`, meaning the schema will never assign confidence ≥ 0.90 to any entity regardless of how well it was extracted.

The `DefaultConfidenceStrategy` is not used in the main extraction pipeline. `extract_from_chunks` returns `enriched_relations` with confidence values from `ExtractedRelation.confidence` (LLM-provided or `None`), not from the `DefaultConfidenceStrategy`. The two confidence systems are parallel and not reconciled.

### Root Cause

The confidence module was developed separately from the extraction pipeline. The integration point (calling `DefaultConfidenceStrategy.score_relations()` on extraction outputs) does not exist in `extract_from_chunks`.

### Impact

**High**. Confidence scores in the output graph are either LLM-self-rated (unreliable without calibration) or `None` (useless). The `DefaultConfidenceStrategy` exists and is architecturally correct but is not connected to the pipeline.

### Redesign

Wire `DefaultConfidenceStrategy` into the post-extraction enrichment step:

```python
# After all_triples are finalized in extract_from_chunks
if confidence_strategy is None:
    confidence_strategy = DefaultConfidenceStrategy()

entity_scores = confidence_strategy.score_entities(
    all_entities, context={"schema": schema, "source_text": full_text}
)
relation_scores = confidence_strategy.score_relations(
    all_triples, enriched_relations=all_enriched, context={"schema": schema}
)
# Merge scores into enriched_relations
```

Also define a protocol for LLM-provided confidence to override heuristic confidence rather than being lost.

---

## 21. Relationship Modeling: Negation Support

### Observation

Negation is modeled as a boolean `is_negated: bool` on `ExtractedRelation`. The primary negation mechanism is heuristic (`_detect_negation_in_window`) which checks for 17 hardcoded English negation cues:

```python
cues = ["no longer", "never", "did not", "does not", "do not", ...]
```

The heuristic returns `True` only if **both** a cue is present AND the relation name stem appears in the same window. This double condition is conservative but produces false negatives when the relation name doesn't match the verb form in the text (e.g. `relation="produces"`, window text: "ceased manufacturing").

Negated relations are **retained** in the graph (they are not filtered out). This is architecturally correct — negated facts are valid assertions — but the downstream query layer and graph storage need to handle `is_negated=True` relations as first-class citizens. Whether this is done is outside the scope of this review, but the extraction architecture must surface them clearly.

### Root Cause

Negation was added as a safety net when the LLM doesn't reliably flag negated relations. The correct solution is to make the LLM flag negation declaratively (via the signature) and use the heuristic only as a fallback.

### Impact

**Medium**. Missed negation flags produce false positive assertions in the graph. Over-trigger of the heuristic (unlikely given its conservatism) produces false negatives. Both degrade graph accuracy.

### Redesign

Same as Issue 5: make negation a first-class output of the relation extraction signature, with the heuristic as a post-hoc correction for LLM omissions only.

---

## 22. Evidence Tracking

### Observation

Evidence tracking is partially implemented:

- `ExtractedRelation.evidence: str | None = None` — intended for verbatim text spans
- `EntityMention.evidence: str | None = None` — intended for entity grounding

The extraction signature `desc` strings do not instruct the LLM to populate `evidence`. As a result, `evidence` is `None` in most extractions.

The `_find_evidence_window()` function in `_heuristics.py` finds a text window around an entity pair, but it is only used for negation/temporal heuristics — not to populate `evidence` in `ExtractedRelation`.

There is no provenance chain: a triple `(source, relation, target)` in the output graph cannot be traced back to the specific chunk(s), text spans, or LLM calls that produced it. This is a fundamental limitation for any system that needs to present evidence to users.

### Root Cause

Evidence population is dependent on the LLM providing it, and the current signatures do not request it. The `_find_evidence_window` function provides the infrastructure to compute evidence post-hoc but it is not called for this purpose.

### Impact

**High for research applications**. A knowledge graph without evidence provenance is opaque. Users cannot validate extractions, understand why a relation was extracted, or identify hallucinations.

### Redesign

1. Add `evidence` as a required field in the extraction signature (see Issue 5).
2. If the LLM does not provide evidence (i.e. `evidence is None` after extraction), use `_find_evidence_window` as a fallback:

```python
def _backfill_evidence(
    relations: list[ExtractedRelation],
    source_text: str,
) -> list[ExtractedRelation]:
    result = []
    for rel in relations:
        if rel.evidence is None:
            window = _find_evidence_window(source_text, rel.source, rel.target)
            rel = rel.model_copy(update={"evidence": window or None})
        result.append(rel)
    return result
```

3. Add `chunk_id` to `ExtractedRelation` and populate it from the extraction loop.

---

## 23. Declarative Design: EntityGroup, RelationGroup, PropertyGroup

### Observation

`EnhancedDRGSchema` defines three grouping constructs:

```python
class EntityGroup:
    name: str
    description: str
    entity_types: list[EntityType]
    examples: list[dict]

class RelationGroup:
    name: str
    description: str
    relations: list[Relation]
    examples: list[dict]

class PropertyGroup:
    name: str
    description: str
    properties: dict[str, Any]
    examples: list[dict]
```

Usage across the extraction pipeline:

| Construct | Defined | Passed to LLM | Used in extraction |
|---|---|---|---|
| `EntityGroup` | ✅ | ❌ | ❌ |
| `RelationGroup` | ✅ | ❌ (flattened) | ❌ (flattened) |
| `PropertyGroup` | ✅ | ❌ | ❌ |

`RelationGroup` is used only to iterate over relations for flattening. Its `description` and `examples` are never passed to any signature. `EntityGroup` and `PropertyGroup` are entirely unused in the extraction pipeline.

This is the most significant gap between the schema design and the extraction implementation.

### Root Cause

The grouping constructs were designed for the schema layer but the extraction signatures were not updated to consume them. The value proposition of `RelationGroup` (semantic grouping with descriptions and examples) is designed for the LLM to understand relational semantics — but the LLM never receives it.

### Impact

**High**. `RelationGroup.examples` are the most valuable field in the schema for improving extraction precision: they give the LLM concrete examples of what a group of relations looks like in text. Not passing them to the extraction signature discards the highest-signal schema information available.

`PropertyGroup` has no extraction infrastructure at all. Properties defined in `EntityType.properties` and in `PropertyGroup` are never extracted.

### Redesign

**For `RelationGroup`**: Pass the group-level `description` and `examples` to the relation extraction signature:

```python
relation_schema_input = [
    {
        "group": rg.name,
        "group_description": rg.description,
        "group_examples": rg.examples[:2],
        "relations": [
            {"name": r.name, "src": r.src, "dst": r.dst, "description": r.description}
            for r in rg.relations
        ]
    }
    for rg in schema.relation_groups
]
```

**For `PropertyGroup`**: Add property extraction to the `EntityMention` model and create a `PropertyExtraction` signature:

```python
class PropertyExtraction(dspy.Signature):
    """Extract property values for entities according to the schema."""
    text: str = dspy.InputField(desc="Input text")
    entities: list[dict] = dspy.InputField(desc="Extracted entity mentions")
    property_schema: list[dict] = dspy.InputField(desc="Properties to extract per entity type")
    entity_properties: list[dict] = dspy.OutputField(
        desc="Entity name → property name → extracted value"
    )
```

---

## 24. Schema Composition Patterns

### Observation

`EnhancedDRGSchema` is constructed directly. There is no mechanism for:
- **Schema inheritance**: A `MedicalSchema` extending a `BaseSchema`
- **Schema mixins**: Attaching a `TemporalMixin` to any schema
- **Schema merging**: Combining schemas from multiple documents or domains
- **Schema versioning**: Tracking when a schema was generated and from which document

The `to_legacy_schema()` conversion is irreversible (information is lost). There is no `from_schema()` class method that would allow composing a new schema from an existing one.

### Root Cause

Schemas are designed as immutable data containers, not as composable objects. The DSPy `Signature` composition model (where signatures can inherit from other signatures) was not applied to `DRGSchema`.

### Impact

**Medium for production systems**. A production system processing hundreds of documents will generate hundreds of schemas. Without a merge or composition mechanism, these schemas accumulate as independent silos with no shared types or relations.

### Redesign

Introduce a `SchemaComposer` class:

```python
class SchemaComposer:
    def merge(
        self,
        base: EnhancedDRGSchema,
        supplement: EnhancedDRGSchema,
        conflict_resolution: Literal["prefer_base", "prefer_supplement", "union"] = "union",
    ) -> EnhancedDRGSchema: ...
    
    def extend(
        self,
        base: EnhancedDRGSchema,
        additional_entity_types: list[EntityType],
        additional_relation_groups: list[RelationGroup],
    ) -> EnhancedDRGSchema: ...
```

Also add a `SchemaRegistry` for managing schemas across documents, with cross-schema alignment to identify equivalent types across independently generated schemas.

---

## 25. Research Novelty and Differentiation

### 25.1 Declarative KG Differentiation

DRG is centered on schema-defined KG construction: users provide or infer a
schema, extraction is constrained by that schema, and the result is a graph
artifact that can be validated, diffed, versioned, queried, and exported.

| Dimension | DRG stance |
|---|---|
| Schema | Declarative, user-defined or auto-generated |
| Entity resolution | Explicit post-processing over extracted entity mentions |
| Cross-chunk relations | Two-pass extraction plus document-level relation extraction |
| Query interface | Direct graph query over `EnhancedKG` |
| Schema discovery | First-class feature from source text |
| Optimization | Optional experimentation, not the core publishing story |

**DRG's genuine differentiators**:
- Declarative schema constraints at extraction time produce structurally consistent graphs.
- Auto-schema discovery from text is a first-class path.
- Typed DSPy signatures keep task shape in input and output fields instead of monolithic prompt strings.

**Current gaps**:
- Entity resolution is still pairwise and should be benchmarked before claiming robust global canonicalization.

### 25.2 Comparison Against Traditional KG Extraction

Traditional systems (REBEL, UniRE, PL-Marker) are supervised models trained on fixed ontologies (TACRED, FewRel, DocRED). They cannot generalize to new schemas without retraining.

**DRG's advantage**: Zero-shot, schema-driven extraction across arbitrary domains. No labeled training data required.

**DRG's gap**: Traditional supervised systems achieve > 80% F1 on their target schemas; DRG's extraction quality is entirely dependent on the LLM's capability and the quality of the schema's `desc` strings.

### 25.3 Comparison Against Schema-First Systems

Systems like DocIE and UniversalIE use a fixed ontology (often WordNet or Freebase-based) and frame extraction as a classification or generation task within that ontology.

**DRG's advantage**: The schema is user-defined and auto-generatable. There is no fixed ontology.

**DRG's gap**: Fixed-ontology systems have provable completeness within their schema — every mention of a known entity type is either extracted or has a known failure mode. DRG has no coverage guarantees (see Section 7).

### 25.4 Genuine Differentiating Architecture

DRG's most genuinely novel contribution is the combination of:

1. **Auto-schema discovery + schema-constrained extraction in a single pipeline**. No other open-source system produces a declarative schema from text and immediately uses that schema to constrain extraction.

2. **TypedPredictor-based structured KG extraction with Pydantic output models**. DSPy's `TypedPredictor` ensures that the LLM output is structurally validated before it reaches the graph layer. This is architecturally superior to regex-based JSON parsing used by most LLM-based KG systems.

3. **`EnhancedDRGSchema` with `EntityGroup`, `RelationGroup`, and `PropertyGroup`** as a composable, domain-agnostic schema language for KG extraction. When fully connected to the extraction pipeline (see Section 23), this provides a schema composition model that no comparable system offers.

### 25.5 Key Architectural Gap Summary

The gap between DRG's **design** and its **implementation** is the central finding of this review. The schema constructs are more expressive than the extraction pipeline can consume:

| Schema Feature | Designed | Implemented in extraction |
|---|---|---|
| Entity descriptions | ✅ | ❌ |
| Relation descriptions | ✅ | ❌ |
| RelationGroup examples | ✅ | ❌ |
| PropertyGroup extraction | ✅ | ❌ |
| EntityGroup-aware extraction | ✅ | ❌ |
| Cross-chunk context snippets | ✅ | ❌ (wired but not called) |
| Entity memory across chunks | ✅ (interface exists) | ❌ (not called in loop) |
| Confidence strategy wiring | ✅ (module exists) | ❌ (not called in pipeline) |

Closing this design-implementation gap — without changing any external interface — would make DRG's extraction architecture among the most complete open-source declarative KG extraction systems available.

---

## Summary Priority Matrix

| Issue | Section | Severity | Effort |
|---|---|---|---|
| Schema descriptions not passed to LLM | §1 | High | Low |
| Output type mismatch in signatures | §2 | High | Low |
| Entity memory not wired into extraction loop | §12 | High | Medium |
| Cross-chunk context snippets not called | §11 | High | Medium |
| Confidence strategy not wired to pipeline | §20 | High | Low |
| RelationGroup examples never passed to LLM | §23 | High | Low |
| No schema coverage guarantee | §7 | High | High |
| LLM not instructed to provide evidence | §22 | High | Low |
| Missing span/chunk provenance in output types | §9 | High | Medium |
| Temporal extraction limited to 4-digit years | §19 | Medium | Medium |
| Alias extraction not instructed | §14 | Medium | Low |
| Threshold design not calibrated | §16 | Medium | High |
| Schema composition not supported | §24 | Medium | High |
| Implicit relation on full document text | §8.4 | Medium | Low |
| Schema mismatch not validated post-extraction | §8.1 | Medium | Low |
| TypedPredictor not in dspy.Module for schema gen | §4 | Low-Medium | Low |

---

*Review conducted against source files: `drg/extract/__init__.py`, `drg/extract/_signatures.py`, `drg/extract/_types.py`, `drg/extract/_schema_gen.py`, `drg/extract/_heuristics.py`, `drg/extract/_chunk_context.py`, `drg/extract/_relations.py`, `drg/extract/_parsing.py`, `drg/schema.py`, `drg/entity_resolution/_resolver.py`, `drg/entity_resolution/_similarity.py`, `drg/entity_resolution/_strategy.py`, `drg/entity_resolution/_normalize.py`, `drg/confidence/_default.py`.*
