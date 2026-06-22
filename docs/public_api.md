# Public API Boundaries

DRG is still alpha. Treat the API in three tiers.

## Stable For Alpha

These are the preferred imports for downstream users:

```python
from drg import DRGSchema, EnhancedDRGSchema, Entity, EntityType, Relation
from drg.graph.builders import build_enhanced_kg
from drg.evaluation import BenchmarkRunner, PipelinePrediction, load_benchmark_dataset
```

CLI commands intended to remain stable through the alpha series:

- `drg extract`
- `drg validate`
- `drg diff`
- `drg versions`
- `drg eval run`
- `drg eval compare`
- `drg eval list`

## Optional Extraction Surface

DSPy-backed extraction is optional at install time. Use:

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
- MCP server internals
- clustering strategy classes
- event extraction prompt/signature internals
- API server response details outside documented endpoints
- UI template implementation details

Prefer documented constructors, CLI commands, and report JSON artifacts over
deep imports from private modules.

## Deprecation Rule

Before the first stable release, DRG may replace experimental APIs directly.
For stable-alpha APIs, changes should either preserve behavior or include a
clear migration note in docs and release notes.
