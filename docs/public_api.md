# Public API Boundaries

DRG is still alpha, but the core extraction surface is now treated as frozen for
the alpha series. Treat the API in three tiers.

## Stable For Alpha

These are the preferred imports for downstream users:

```python
from drg import (
    DRGSchema,
    EnhancedDRGSchema,
    Entity,
    EntityType,
    Relation,
    extract_from_chunks,
    extract_from_chunks_async,
    extract_triples,
    extract_typed,
    extract_typed_async,
)
from drg.graph.builders import build_enhanced_kg
from drg.evaluation import BenchmarkRunner, PipelinePrediction, load_benchmark_datasets
```

Stable-alpha extraction contracts:

- `extract_typed(text, schema, ...)` extracts typed entities and relation triples
  from a single text input.
- `extract_from_chunks(chunks, schema, ...)` extracts across chunk dictionaries
  or chunk-like text inputs and is the preferred long-document entry point.
- `extract_typed_async(...)` and `extract_from_chunks_async(...)` preserve the
  synchronous API shape while running work in a thread.
- `extract_triples(text, schema)` remains a backward-compatible convenience
  wrapper that returns only triples.
- By default, extraction returns `(entities, triples)`. With
  `return_enriched=True`, it returns `(entities, triples, enriched_relations)`.
- Missing LM configuration continues to return empty extraction in non-strict
  test/offline mode, and raises `LLMConfigError` when `DRG_REQUIRE_LM=1`,
  `DRG_STRICT=1`, or production strict mode is enabled.

Do not rename these functions or change their core return behavior without a
changelog entry, migration note, and deprecation path.

CLI commands intended to remain stable through the alpha series:

- `drg extract`
- `drg validate`
- `drg diff`
- `drg versions`
- `drg eval run`
- `drg eval compare`
- `drg eval list`

## Optional Extraction Surface

DSPy-backed extraction is optional at install time, even though the public API
names are stable. Use:

```bash
pip install "drg-kg[dspy]"
```

or:

```bash
pip install "drg-kg[extract]"
```

Then import extraction entry points lazily:

```python
from drg import extract_typed
```

Graph-only, validation, and evaluation workflows do not require DSPy.

## Experimental

The following surfaces may change before a stable release:

- optimizer internals
- confidence calibration heuristics and calibration data formats
- long-document optimization knobs and windowing heuristics
- MCP server internals
- clustering strategy classes
- event extraction prompt/signature internals
- API server response details outside documented endpoints
- UI template implementation details

Prefer documented constructors, CLI commands, and report JSON artifacts over
deep imports from private modules.

## Deprecation Rule

Before the first stable release, DRG may replace experimental APIs directly. For
stable-alpha APIs, changes should preserve behavior whenever possible. If a
breaking change is unavoidable, it must include:

- a changelog entry,
- a migration note in this document or the relevant README section,
- and, where practical, a compatibility wrapper for at least one alpha minor
  release.
