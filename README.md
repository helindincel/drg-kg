# DRG - Declarative Relationship Generation

[![CI](https://github.com/helindincel/drg-kg/actions/workflows/ci.yml/badge.svg)](https://github.com/helindincel/drg-kg/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

DRG is a **schema-driven Knowledge Graph extraction framework**. It turns
unstructured text into explainable Knowledge Graph artifacts by combining
declarative schemas with DSPy-backed extraction.

The core promise is extraction: define or infer a domain schema, extract typed
entities and relations, and build an `EnhancedKG` that can be inspected,
validated, queried deterministically, versioned, evaluated, or exported. DRG is
not a GraphRAG framework, not a RAG framework, and not a retrieval or chat
application stack. Query, API, MCP, Neo4j, optimizer, and evaluation modules are
supporting layers around extracted graphs.

Turkish documentation is available in [`README.tr.md`](README.tr.md).

## Alpha Status

DRG is currently an alpha-stage project. The primary extraction APIs are now
treated as stable for the alpha series, while JSON shapes, CLI flags, and
optional integration surfaces may still change before `v1.0`. Pin versions for
serious experiments and review [`CHANGELOG.md`](CHANGELOG.md) before upgrading.

## Why DRG?

Most tools around text, LLMs, and graphs solve one layer of the problem. DRG is
the lifecycle layer for teams that need the whole path: create the KG, enrich
it, update it, version it, evaluate it, query it, and expose it to downstream
systems.

Many projects stop at "extract triples from text." DRG treats extraction as the
first step in a longer engineering workflow around graph evolution, quality,
traceability, and integration.

| Tool | Best at | Where DRG is different |
|:---|:---|:---|
| LangChain | LLM application orchestration and chains | DRG focuses on the KG lifecycle: schema-driven extraction, provenance, versioning, evaluation, and deterministic graph querying. |
| LlamaIndex | Document indexing and LLM workflow helpers | DRG builds graph-native structures; chat and generation layers are outside this repo's product scope. |
| Neo4j | Persistent graph database and Cypher querying | DRG creates, validates, enriches, versions, evaluates, and exports KGs; Neo4j can be a downstream storage target. |
| NetworkX | In-memory graph algorithms | DRG adds extraction, schema validation, provenance, temporal metadata, entity resolution, lifecycle operations, CLI/API surfaces, and export workflows. |
| Raw DSPy programs | Typed LLM programs | DRG wraps DSPy extraction in a full KG lifecycle: chunking, schema generation, graph building, updates, reasoning, evaluation, and serving. |

## Lifecycle Scope

DRG's core abstraction is not just "KG output." It is the lifecycle around a KG:

| Lifecycle stage | DRG responsibility |
|:---|:---|
| Design | Define or infer an extraction schema for the domain. |
| Build | Extract entities/relations and construct an `EnhancedKG`. |
| Trust | Attach provenance, evidence, confidence, and validation results. |
| Evolve | Merge new documents, resolve entities, diff graph snapshots, and keep versions. |
| Reason | Run deterministic graph queries, temporal lookups, multi-hop traversal, and rule-based inference. |
| Evaluate | Measure extraction, graph query behavior, graph structure, and performance regressions. |
| Integrate | Serve through CLI/API/MCP, export JSON, and sync to Neo4j. |

## What DRG Is / Is Not

DRG is:

- A Knowledge Graph extraction framework.
- A schema-first library for extracting typed entities and relations from text.
- A graph construction and enrichment toolkit for text-derived entities and
  relations.
- A deterministic query, evaluation, versioning, provenance, and export layer
  around `EnhancedKG`.
- A practical CLI/API/MCP package for local experiments and integration
  prototypes.

DRG is not:

- A general LLM application framework.
- A GraphRAG framework.
- A RAG framework.
- A chatbot framework.
- A database or search backend.
- A retrieval framework.
- A vector or hybrid search layer; embeddings are optional helper signals for
  entity resolution/export, not a serving index.
- A replacement for Neo4j, NetworkX, LangChain, LlamaIndex, or DSPy.
- A hosted product or fully stable production platform yet.

## Architecture

```text
Unstructured Text
      |
      v
Schema / Auto-Schema
      |
      v
Chunking + DSPy Extraction
      |
      v
EnhancedKG
      |
      +--> Provenance / Confidence / Validation
      |
      +--> Entity Resolution / Incremental Updates / Versioning
      |
      +--> Temporal Metadata / Multi-Document Reasoning
      |
      v
Query + Reasoning + Evaluation
      |
      v
CLI / FastAPI UI / MCP / Neo4j Export / JSON
```

## Feature Matrix

| Area | Feature | Status | Notes |
|:---|:---|:---|:---|
| Extraction | Declarative schemas | Available | Entity types, relation groups, examples, metadata. |
| Extraction | Auto-schema generation | Available | Bootstraps a schema from raw text. |
| Graph core | `EnhancedKG` | Available | Typed nodes, edges, clusters, JSON export. |
| Trust | Provenance | Available | Evidence/source metadata can travel with nodes and edges. |
| Trust | Confidence scoring | Available | Confidence metadata and filtering strategies. |
| Lifecycle | Versioning | Available | Snapshot, diff, and rollback helpers for graph evolution. |
| Lifecycle | Incremental updates | Available | Merge new documents into an existing KG. |
| Intelligence | Entity resolution | Available | Canonical entity merge and alias handling. |
| Intelligence | Temporal query | Available | Timeline helpers and compact temporal lookups. |
| Intelligence | Multi-document reasoning | Available | Rule-based inference over graph paths and bridges. |
| Integration | FastAPI + Cytoscape UI | Available | Local graph exploration and API endpoints. |
| Integration | Neo4j export | Available | Sync/export graph data to Neo4j. |
| Integration | MCP server | Available | Exposes KG operations to MCP-compatible clients. |
| Quality | Evaluation framework | Available | Extraction, graph-query, structural, and performance metrics. |

## Experimental Features

The following modules are useful for research and integration experiments, but
are not part of the frozen core extraction API yet:

- Optimizer integration: DSPy optimizer workflows are experimental and may
  change as training-data and metric conventions settle.
- Confidence calibration: confidence strategies and calibrated scores are
  heuristic/experimental unless you fit them on your own labelled data.
- Long-document optimization: chunking, cross-chunk relation recovery, and
  windowed relation extraction are evolving.
- API/UI/MCP integration details: documented commands and endpoints are intended
  to work, but internal response shapes and UI implementation details may change.

## Use Cases

- News analysis: extract people, companies, events, acquisitions, conflicts,
  and timelines from reporting.
- Enterprise documents: turn policies, reports, contracts, and internal notes
  into explainable graph structures.
- Research reports: connect findings, methods, entities, datasets, and citations
  across papers or technical documents.
- Multi-document knowledge fusion: merge partial facts from many sources into a
  single graph with provenance.
- Knowledge graph operations: keep extracted facts queryable, versioned,
  explainable, and ready for downstream graph databases or analytics.

## Roadmap

`v0.2` targets:

- Keep the top-level Python extraction API stable across the alpha series.
- Expand no-key and mocked demos so new users can evaluate DRG quickly.
- Improve evaluation coverage for extraction, temporal metadata, and graph queries.
- Tighten optional integration tests for API, MCP, Neo4j, and benchmark flows.

`v1.0` targets:

- Commit to stable public API boundaries and migration policy.
- Publish production-ready package metadata and release workflow.
- Provide generated API reference docs and clearer architecture decision records.
- Raise confidence in graph correctness with broader regression and benchmark
  coverage.

## Related Work

DRG builds around ideas from several ecosystems:

- [DSPy](https://github.com/stanfordnlp/dspy): typed LLM programs and
  optimization.
- [LangChain](https://github.com/langchain-ai/langchain): LLM application
  orchestration.
- [LlamaIndex](https://github.com/run-llama/llama_index): indexing over
  data sources.
- [Neo4j](https://neo4j.com/): graph persistence and Cypher querying.
- [NetworkX](https://networkx.org/): in-memory graph algorithms.

## Install

DRG supports Python 3.10, 3.11, 3.12, and 3.13.

```bash
# Source checkout, minimal graph/query usage
pip install -e .

# Development tooling
pip install -e ".[dev]"

# Optional focused installs
pip install -e ".[extract]"  # DSPy extraction
pip install -e ".[api]"      # FastAPI UI
pip install -e ".[mcp]"      # MCP server
pip install -e ".[neo4j]"    # Neo4j export
```

After the public PyPI release, install the package with the extras you need:

```bash
pip install "drg-kg[all]"
pip install "drg-kg[extract]"  # DSPy extraction only
```

## Quickstart

For the complete first-run guide, see
[`docs/getting_started.md`](docs/getting_started.md).

The first runnable example needs no API key and no LLM:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
drg --help
python examples/query_layer_example.py
```

Live extraction needs a model provider:

```bash
pip install -e ".[extract]"
cp .env.example .env

export DRG_MODEL=openai/gpt-4o-mini
export OPENAI_API_KEY=sk-...

# Or Gemini
export DRG_MODEL=gemini/gemini-2.0-flash-exp
export GEMINI_API_KEY=...

# Or local Ollama
export DRG_MODEL=ollama_chat/llama3
export DRG_BASE_URL=http://localhost:11434
```

Production extraction should fail fast when no LM is configured. Set
`DRG_REQUIRE_LM=1` or `DRG_STRICT=1` to turn missing-LM mock mode into an
`LLMConfigError`; without those flags the Python API keeps returning empty
extractions for tests and offline examples.

Run a story-oriented CLI extraction:

```bash
drg extract story.txt --auto-schema --output-format enhancedkg -o story_kg.json
drg validate story_kg.json
```

For the first publishing scenario, start with a roughly 20-page story document.
DRG samples the document, creates a detailed `EnhancedDRGSchema` with entity
types, examples, properties, relation groups, and relation examples, then uses
that schema as structured DSPy input for KG extraction. The output is a graph
artifact you can validate, diff, version, query deterministically, or export.

Or use the Python API:

```python
from drg import EnhancedDRGSchema, EntityType, Relation, RelationGroup, extract_typed
from drg.graph.builders import build_enhanced_kg

schema = EnhancedDRGSchema(
    entity_types=[
        EntityType(
            name="Place",
            description="Story locations and meaningful settings",
            examples=["the old harbor", "the archive tower"],
            properties={"atmosphere": "mood or sensory quality", "role": "story function"},
        ),
        EntityType(
            name="Artifact",
            description="Objects that carry plot, memory, or symbolic meaning",
            examples=["silver compass", "sealed letter"],
            properties={"condition": "observed state", "significance": "why it matters"},
        ),
        EntityType(
            name="Conflict",
            description="Tensions, obstacles, or unresolved story problems",
            examples=["blocked passage", "broken alliance"],
            properties={"stakes": "what may be lost", "status": "open/resolved/etc."},
        ),
    ],
    relation_groups=[
        RelationGroup(
            name="narrative_structure",
            description="How story elements shape one another",
            relations=[
                Relation(
                    "reveals",
                    "Artifact",
                    "Conflict",
                    description="An artifact exposes or clarifies a conflict",
                ),
                Relation(
                    "located_in",
                    "Artifact",
                    "Place",
                    description="An artifact is found or kept in a place",
                ),
            ],
        )
    ],
)

text = "The sealed letter in the archive tower revealed why the harbor gate was locked."
entities, triples = extract_typed(text, schema)
kg = build_enhanced_kg(entities_typed=entities, triples=triples, schema=schema, source_text=text)

print(kg.to_json())
```

To try a deterministic repository demo without setting an API key:

```bash
python examples/query_layer_example.py
```

## Public API Stability

The preferred Python API names are now treated as frozen for the alpha series:
`extract_typed()`, `extract_from_chunks()`, `extract_typed_async()`,
`extract_from_chunks_async()`, and the backward-compatible `extract_triples()`
wrapper. Their documented names and core return behavior should not change
without a changelog entry and migration note.

Schema types such as `DRGSchema`, `EnhancedDRGSchema`, `EntityType`, `Relation`,
and the `build_enhanced_kg()` builder are also part of the documented user
surface. Deep imports from optimizer, API, MCP, UI, and prompt internals remain
experimental. See [`docs/public_api.md`](docs/public_api.md).

## Example Gallery

| Example | What it demonstrates |
|:---|:---|
| [`examples/quickstarts/01_wikipedia_article.py`](examples/quickstarts/01_wikipedia_article.py) | Small biographical/encyclopedic extraction with an inline schema. |
| [`examples/quickstarts/02_financial_news.py`](examples/quickstarts/02_financial_news.py) | Corporate and financial-news entity/relation extraction. |
| [`examples/quickstarts/03_biomedical.py`](examples/quickstarts/03_biomedical.py) | Biomedical drug, disease, and gene graph extraction. |
| [`examples/full_pipeline_example.py`](examples/full_pipeline_example.py) | End-to-end pipeline with chunking, extraction, graph build, clustering, and reports. |
| [`examples/api_server_example.py`](examples/api_server_example.py) | Local FastAPI server and Cytoscape graph UI. |
| [`examples/incremental_update_example.py`](examples/incremental_update_example.py) | Merging new documents into an existing graph. |
| [`examples/temporal_query_example.py`](examples/temporal_query_example.py) | Temporal metadata and timeline-style queries. |
| [`examples/query_layer_example.py`](examples/query_layer_example.py) | Deterministic graph query layer usage. |
| [`examples/multi_document_reasoning_example.py`](examples/multi_document_reasoning_example.py) | Cross-document reasoning and inferred graph bridges. |
| [`examples/event_extraction_example.py`](examples/event_extraction_example.py) | Event-oriented extraction pipeline. |
| [`examples/evaluation_framework_example.py`](examples/evaluation_framework_example.py) | Evaluation metrics and report generation. |
| [`examples/mcp_demo.py`](examples/mcp_demo.py) | MCP integration flow. |
| [`examples/optimizer_demo.py`](examples/optimizer_demo.py) | DSPy optimizer experiments around extraction. |

Input -> output artifact walkthrough (Turkish):
[`docs/input_output_examples.tr.md`](docs/input_output_examples.tr.md).

## CLI

| Command | Purpose |
|:---|:---|
| `drg extract` | Extract a KG from a file or stdin. |
| `drg validate` | Validate a persisted KG JSON file. |
| `drg diff` | Compare two KG snapshots. |
| `drg versions list` | List graph version snapshots. |
| `drg versions diff` | Compare graph versions. |
| `drg versions rollback` | Restore a previous graph version. |
| `drg eval run` | Run a benchmark dataset. |
| `drg eval list` | List bundled benchmark datasets and adapters. |
| `drg eval compare` | Compare evaluation reports. |

Incremental update example:

```bash
drg extract new_article.txt --update global_kg.json --infer
drg validate global_kg.json
drg diff previous_kg.json global_kg.json --json
```

## API, UI, MCP, and Evaluation

```bash
# Interactive Cytoscape UI
python examples/api_server_example.py

# Neo4j sync preview
curl -X POST "http://localhost:8000/api/neo4j/sync?dry_run=true"

# MCP server for Cursor / Claude Desktop
python -m drg.mcp_server

# Benchmark run
drg eval run examples/benchmarks/corporate_acquisition_benchmark.json \
  --measure-performance \
  -o reports/current.json \
  --markdown-output reports/current.md
```

See [`docs/api_server.md`](docs/api_server.md),
[`docs/mcp_integration.md`](docs/mcp_integration.md), and
[`docs/evaluation_framework.md`](docs/evaluation_framework.md) for details.

## Project Map

```text
drg/
├── schema.py              # Enhanced schema definitions
├── extract/               # DSPy-backed extraction
├── chunking/              # Token and sentence chunkers
├── graph/                 # EnhancedKG, provenance, diffing, versioning
├── query/                 # Deterministic query and analytics layer
├── temporal/              # Temporal reasoning and timeline helpers
├── reasoning/             # Multi-document inference
├── evaluation/            # Metrics, reports, benchmark adapters
├── api/                   # FastAPI server and Cytoscape UI
├── events/                # Event extraction pipeline
└── cli.py                 # CLI entry point
```

## Documentation

- First run: [`docs/getting_started.md`](docs/getting_started.md)
- Installation and configuration: [`docs/setup.md`](docs/setup.md)
- Architecture: [`docs/project_overview.md`](docs/project_overview.md)
- Pipeline: [`docs/pipeline_overview.md`](docs/pipeline_overview.md)
- Input/output walkthrough (TR): [`docs/input_output_examples.tr.md`](docs/input_output_examples.tr.md)
- Schema design: [`docs/schema_design.md`](docs/schema_design.md)
- Public API: [`docs/public_api.md`](docs/public_api.md)
- Benchmarks: [`docs/benchmarking.md`](docs/benchmarking.md)
- Quickstart scripts: [`examples/quickstarts/README.md`](examples/quickstarts/README.md)

## Known Limitations

- DRG extracts and operates on Knowledge Graph artifacts; it does not provide a
  GraphRAG or RAG serving stack.
- Optimizer integration is experimental and should be treated as a research
  workflow, not a stable production tuning layer.
- Long-document optimization is evolving; chunking and cross-chunk relation
  recovery can require domain-specific tuning.
- Confidence calibration is heuristic/experimental unless calibrated against
  labelled data for your domain.
- Live extraction quality depends on the configured LLM, schema quality, and
  provider behavior.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check .
mypy drg
```

Contribution guidelines are in [`CONTRIBUTING.md`](CONTRIBUTING.md). Security
reporting is covered in [`SECURITY.md`](SECURITY.md).

## License

MIT © [Helin Dinçel](https://github.com/helindincel)
