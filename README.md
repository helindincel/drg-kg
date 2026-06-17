# DRG — Declarative Relationship Generation

[![CI](https://github.com/helindincel/drg-kg/actions/workflows/ci.yml/badge.svg)](https://github.com/helindincel/drg-kg/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/drg-kg.svg)](https://pypi.org/project/drg-kg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

DRG is a **high-performance, DSPy-based framework** for building deep, searchable, and explainable **Knowledge Graphs (KG)** from unstructured text. Unlike traditional RAG systems that rely on fuzzy vector lookups, DRG uses **declarative schema extraction** to build deterministic graph structures that support complex reasoning, temporal analysis, and multi-document synthesis.

> 🇹🇷 **Türkçe okuyucular için:** [`README.tr.md`](README.tr.md) | 🗺 **Roadmap:** [`STATUS.md`](STATUS.md)

---

## 🚀 Key Features

### 🧠 Intelligent Extraction
- **Declarative Schema**: Define your domain model; DRG handles the extraction logic via **DSPy TypedPredictors**.
- **Enhanced Schema**: Rich type definitions with descriptions, examples, and property groups.
- **Auto-Schema Generation**: Bootstraps an initial schema from raw text—no manual modeling required.
- **Chunk-Aware Processing**: Handles long documents with context-aware chunking and cross-chunk relation merging.

<<<<<<< Updated upstream
- [`docs/project_overview.md`](https://github.com/helindincel/drg-kg/blob/main/docs/project_overview.md)
=======
### 🕸 Graph Intelligence
- **Incremental Ingestion**: Add new documents to an existing graph with automated entity resolution and relationship deduplication.
- **Temporal KG**: Native support for `valid_from`/`valid_to` metadata, partial dates, and timeline building.
- **Multi-Document Reasoning**: Rule-based inference engine that discovers cross-document bridges (e.g., A knows B, B knows C → A connected to C).
- **Clustering & Communities**: Automated community detection (Louvain, Leiden) with LLM-powered group summarization.
>>>>>>> Stashed changes

### 🛠 Production Ready
- **Query & Reasoning Layer**: Deterministic graph traversal with multi-hop path finding and provenance-backed answers.
- **Evaluation Framework**: Integrated metrics (P/R/F1, NDCG, Hits@K) and regression testing for extraction quality.
- **API & UI**: Built-in FastAPI server and interactive Cytoscape.js web interface.
- **Multi-LLM**: Works with OpenAI, Gemini, Anthropic, Ollama, and more.
- **MCP Integration**: Exposes KG capabilities via Model Context Protocol.

---

## 📦 Installation

```bash
# Core package
pip install drg-kg

# Development tools & all optional features
pip install "drg-kg[all]"
```

<<<<<<< Updated upstream
For detailed installation and troubleshooting, see
[`docs/setup.md`](https://github.com/helindincel/drg-kg/blob/main/docs/setup.md).

### Requirements

- Python `>= 3.10`
- `dspy >= 2.5.0, < 3.0.0`
- `pydantic >= 2.0.0`

---

## Configuration

DRG is configured via environment variables:

```bash
cp .env.example .env
# Edit .env and fill in the relevant API key.
```

Common variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DRG_MODEL` | `openai/gpt-4o-mini` | DSPy / LiteLLM model id |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / ... | — | Provider-specific API key |
| `DRG_TEMPERATURE` | `0.0` | LLM temperature |
| `DRG_MAX_TOKENS` | `1500` | LLM output budget |
| `DRG_BASE_URL` | — | Ollama / self-hosted gateway |

Full variable list: [`docs/setup.md`](https://github.com/helindincel/drg-kg/blob/main/docs/setup.md).

---

## Quickstart

### Basic usage (legacy schema)
=======
---

## ⚡ Quickstart
>>>>>>> Stashed changes

### 1. Define and Extract (Enhanced Schema)
```python
from drg import EnhancedDRGSchema, EntityType, Relation, extract_typed, EnhancedKG

schema = EnhancedDRGSchema(
    entity_types=[
        EntityType(name="Company", description="Firms and organizations"),
        EntityType(name="Person", description="Individuals")
    ],
    relations=[Relation("founded_by", "Company", "Person")]
)

text = "TechCorp was founded by Jane Doe in 2015."
entities, triples = extract_typed(text, schema)

kg = EnhancedKG.from_typed(entities, triples, schema=schema)
print(kg.to_json())
```

<<<<<<< Updated upstream
### Runnable showcase examples

For end-to-end, copy-pasteable scripts on three different domains, see
[`examples/quickstarts/`](https://github.com/helindincel/drg-kg/tree/main/examples/quickstarts/):

| Script | Domain |
|--------|--------|
| `01_wikipedia_article.py` | Biographical / encyclopedic text |
| `02_financial_news.py` | Corporate news (M&A, funding rounds) |
| `03_biomedical.py` | Drug / disease / gene relationships |

Each script is self-contained: define schema, run extraction, dump JSON KG.
=======
### 2. Auto-Schema Ingestion
```bash
# Automatically infer schema and extract KG
drg extract sample.txt --auto-schema -o output_kg.json
```
>>>>>>> Stashed changes

---

## 🛠 Modules & CLI

### CLI Subcommands
| Command | Description |
|:---|:---|
| `drg extract` | Extract entities/relations from a file or stdin. |
| `drg eval run` | Execute a benchmark against a gold-standard dataset. |
| `drg eval compare` | Detect quality regressions between two runs. |

### Incremental Updates & Reasoning
```bash
# Add a new doc to existing graph
drg extract new_article.txt --update global_kg.json --infer
```
*The `--update` flag merges data into the existing graph. The `--infer` flag runs the reasoning layer to discover new connections.*

---

## 📊 Evaluation Framework

Measure the quality of your KG pipeline with granular metrics:
- **Extraction**: Entity and Relation P/R/F1.
- **Retrieval**: NDCG, Recall@K, and Hits@K for RAG evaluation.
- **Structural**: Graph density, coverage, and orphan node rates.

```bash
drg eval run benchmarks/corporate.json -o reports/current.json
```

---

## ⏳ Temporal Support

DRG treats time as a first-class citizen inside the graph.
- **State Transition**: Track how an entity's properties change over time.
- **Timeline Building**: Generate a chronological history for any node.
- **Conflict Detection**: Identify temporal contradictions (e.g., a person being CEO of two competing firms simultaneously).

---

## 🏗 Project Structure

```text
drg/
├── schema.py           # Enhanced Schema definitions
├── extract/            # DSPy-based extraction logic
├── graph/              # EnhancedKG and graph manipulation
├── temporal/           # Temporal reasoning & timelines
├── reasoning/          # Multi-document inference
├── evaluation/         # Metrics and benchmarking
├── query/              # Deterministic query engine
├── api/                # FastAPI & Cytoscape UI
└── cli.py              # Unified CLI entry point
```

---

## 🤝 Contributing

We use **uv** for dependency management and **pytest** for testing.
```bash
<<<<<<< Updated upstream
pip install -e ".[api]"
python examples/api_server_example.py
# UI:    http://localhost:8000
# Docs:  http://localhost:8000/docs
```

Details and endpoint list: [`docs/api_server.md`](https://github.com/helindincel/drg-kg/blob/main/docs/api_server.md).

---

## API Reference (summary)

### Schema classes

| Class | Use case |
|-------|----------|
| `DRGSchema` | Legacy: simple `Entity` + `Relation` |
| `EnhancedDRGSchema` | Recommended: rich `EntityType` + `RelationGroup` |

### Extraction

```python
extract_typed(text, schema)
# -> (entities, triples)
#    entities: List[Tuple[str, str]]            # (entity_name, entity_type)
#    triples:  List[Tuple[str, str, str]]       # (source, relation, target)

extract_triples(text, schema)
# Triples only (kept for backward compatibility).
```

### Knowledge Graph

```python
kg = KG.from_typed(entities, triples)
kg = KG.from_triples(triples)
print(kg.to_json(indent=2))
```

For the rich KG (`EnhancedKG`, `KGNode`, `KGEdge`, `Cluster`), see the
`drg.graph` module.

---

## Project Structure

```
DRG/
├── drg/                       # Main package (monolithic codebase)
│   ├── __init__.py            # Public API + lazy loading
│   ├── schema.py              # EnhancedDRGSchema, EntityType, RelationGroup, ...
│   ├── protocols.py           # Structural interfaces (KGExtractor / Embedding / Clustering / LLM)
│   ├── errors.py              # Typed exception hierarchy (DRGError + 11 subclasses)
│   ├── config.py              # LMConfig — DSPy LM setup (env-driven)
│   ├── extract/               # DSPy extraction package (KGExtractor + cross-chunk + heuristics)
│   ├── coreference_resolution/# Pronoun resolution (strategy pattern: NLP + heuristic)
│   ├── entity_resolution/     # Entity merging (strategy pattern: String + Hybrid)
│   ├── chunking/              # Token / sentence / semantic chunkers
│   ├── embedding/             # Provider abstraction (OpenAI / Gemini / Local / OpenRouter)
│   ├── graph/                 # EnhancedKG, schema_generator, community_report,
│   │                          # relationship_model/ (package), visualization_adapter/ (package),
│   │                          # hub_mitigation, query_engine, auto_clusters, neo4j_exporter
│   ├── clustering/            # Louvain / Leiden / Spectral + summarization
│   ├── optimizer/             # DSPy optimizer + metrics
│   ├── api/                   # FastAPI server + Cytoscape UI
│   ├── mcp_api.py             # MCP integration
│   ├── cli.py                 # `drg` CLI entry point
│   └── utils/                 # env_loader, llm_throttle, strict, logging, cache (shared LRU)
├── docs/                      # Documentation (NO CODE — currently in Turkish)
├── examples/                  # full_pipeline_example, api_server_example, ...
├── tests/                     # Unit + integration + multi_dataset evaluation
├── outputs/                   # Generated artifacts (gitignored)
├── inputs/                    # Sample text files
├── pyproject.toml             # Single source of truth (deps + tooling)
└── README.md
```

---

## Testing and Development

```bash
# Developer install
=======
>>>>>>> Stashed changes
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

<<<<<<< Updated upstream
DRG supports the following providers via DSPy/LiteLLM. Model IDs use the
`provider/model` format:

- **OpenAI** — `openai/gpt-4o-mini`, `openai/gpt-4`, ...
- **Google Gemini** — `gemini/gemini-2.0-flash-exp`, ...
- **Anthropic** — `anthropic/claude-3-5-sonnet`, ...
- **Perplexity** — `perplexity/llama-3.1-sonar-large-128k-online`, ...
- **OpenRouter** — `openrouter/<model>`
- **Ollama (local)** — `ollama_chat/llama3`, `ollama_chat/mistral`, ...

The model is selected via the `DRG_MODEL` environment variable.

---

## Optional Dependencies

Modular dependency layout:

| Extra | Contents |
|-------|----------|
| `api` | FastAPI, uvicorn |
| `neo4j` | Neo4j driver |
| `openai` / `gemini` / `openrouter` | LLM/embedding clients |
| `local` | sentence-transformers (local embedding) |
| `louvain` / `leiden` / `spectral` | Clustering backends |
| `networkx` | Graph processing |
| `coreference` | spaCy + coreferee |
| `dev` | pytest, ruff, mypy, pre-commit, pytest-cov |
| `all` | All of the above |

---

## Documentation

The two highest-value docs are now available in English. The rest are
still Turkish; English translations are tracked in `STATUS.md`. The
Python API, code comments, and error messages are English.

| Doc | EN | TR |
|---|:-:|:-:|
| Architecture & philosophy | [`project_overview.md`](https://github.com/helindincel/drg-kg/blob/main/docs/project_overview.md) | [`project_overview.tr.md`](https://github.com/helindincel/drg-kg/blob/main/docs/project_overview.tr.md) |
| Pipeline flow | [`pipeline_overview.md`](https://github.com/helindincel/drg-kg/blob/main/docs/pipeline_overview.md) | [`pipeline_overview.tr.md`](https://github.com/helindincel/drg-kg/blob/main/docs/pipeline_overview.tr.md) |
| Setup | [`setup.md`](https://github.com/helindincel/drg-kg/blob/main/docs/setup.md) | — |
| API + UI | [`api_server.md`](https://github.com/helindincel/drg-kg/blob/main/docs/api_server.md) | — |
| Schema design | — | [`schema_design.md`](https://github.com/helindincel/drg-kg/blob/main/docs/schema_design.md) |
| Chunking strategy | — | [`chunking_strategy.md`](https://github.com/helindincel/drg-kg/blob/main/docs/chunking_strategy.md) |
| Relationship model | — | [`relationship_model.md`](https://github.com/helindincel/drg-kg/blob/main/docs/relationship_model.md) |
| Clustering & summarisation | — | [`clustering_summarization.md`](https://github.com/helindincel/drg-kg/blob/main/docs/clustering_summarization.md) |
| DSPy optimizer | — | [`optimizer_design.md`](https://github.com/helindincel/drg-kg/blob/main/docs/optimizer_design.md) |
| MCP integration | — | [`mcp_integration.md`](https://github.com/helindincel/drg-kg/blob/main/docs/mcp_integration.md) |

---

## Citation

If you use DRG in academic or research work, please cite the repository:

```bibtex
@software{drg_kg_2026,
  author  = {Dinçel, Helin},
  title   = {DRG: Declarative Relationship Generation for Knowledge Graphs},
  year    = {2026},
  url     = {https://github.com/helindincel/drg-kg}
}
```

---

## License

MIT — see the `LICENSE` file for details.
=======
MIT © [Helin Dinçel](https://github.com/helindincel)
>>>>>>> Stashed changes
