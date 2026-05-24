# DRG — Declarative Relationship Generation

DRG is a **DSPy-based, declarative** Python library for extracting **Knowledge
Graphs (KG)** from text. You define the schema; DRG handles entity and relation
extraction, and layers clustering, community reports, and visualization on top.

> **⚠️ Alpha:** This repo is at version `0.1.0a0`. APIs may change.

> 🇹🇷 **Türkçe okuyucular için:** [`README.tr.md`](README.tr.md)

---

## Detailed Project Overview

For a deeper dive into **architecture and philosophy** (recommended for first-time
readers), see:

- [`docs/project_overview.md`](docs/project_overview.md)

That document clarifies in particular:

- DRG's **DSPy-based, declarative** extraction approach
- Why DRG is **not a RAG/serving framework** (the UI query path is a
  deterministic KG lookup, not LLM generation)
- The dataset-agnostic design and Enhanced schema approach
- Pipeline flow, UI, and repo layout

> 📌 **Note on documentation language:** Most files under `docs/` are currently
> in Turkish. English translations are part of the roadmap. The Python API,
> code comments, and error messages are in English.

---

## Features

- **Declarative Schema** — Define entity types and relations; DRG handles the rest
- **DSPy Integration** — Structured extraction via `TypedPredictor`
- **Enhanced Schema** — Rich definitions with `EntityType`, `RelationGroup`,
  `EntityGroup`, `PropertyGroup`
- **Automatic Schema Generation** — Derive a schema from raw text via
  `generate_schema_from_text()`
- **Chunk-Based Extraction** — Context-aware chunking for long documents
- **Knowledge Graph Layer** — `EnhancedKG` (`KGNode`, `KGEdge`, `Cluster`)
- **Clustering & Community Reports** — Louvain / Leiden / Spectral +
  summarization
- **API Server + UI** — FastAPI + Cytoscape.js interactive visualization
- **Multi-LLM Support** — OpenAI, Gemini, Anthropic, Perplexity, OpenRouter,
  Ollama
- **Optional Neo4j Export** — Graph persistence

---

## Installation

```bash
git clone https://github.com/helindincel/drg-kg.git
cd drg-kg

# Core only
pip install -e .

# With all optional features (api, embeddings, clustering, ...)
pip install -e ".[all]"

# Developer mode (test + lint + type-check tooling)
pip install -e ".[dev]"
```

For detailed installation and troubleshooting, see
[`docs/setup.md`](docs/setup.md).

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

Full variable list: [`docs/setup.md`](docs/setup.md).

---

## Quickstart

### Basic usage (legacy schema)

```python
from drg import Entity, Relation, DRGSchema, extract_typed, KG

schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")],
)

text = "Apple released the iPhone 16 in September 2025."
entities, triples = extract_typed(text, schema)

kg = KG.from_typed(entities, triples)
print(kg.to_json())
```

### Enhanced schema (recommended)

```python
from drg import (
    EntityType,
    RelationGroup,
    Relation,
    EnhancedDRGSchema,
    extract_typed,
    KG,
)

schema = EnhancedDRGSchema(
    entity_types=[
        EntityType(
            name="Company",
            description="Business organizations that produce products",
            examples=["Apple", "Google", "Microsoft"],
            properties={"industry": "tech"},
        ),
        EntityType(
            name="Product",
            description="Goods produced by companies",
            examples=["iPhone", "Android", "Windows"],
        ),
    ],
    relation_groups=[
        RelationGroup(
            name="production",
            description="How companies create products",
            relations=[
                Relation("produces", "Company", "Product"),
                Relation("manufactures", "Company", "Product"),
            ],
        )
    ],
    auto_discovery=True,
)

text = "Apple produces iPhones. Google develops Android."
entities, triples = extract_typed(text, schema)
kg = KG.from_typed(entities, triples)
print(kg.to_json())
```

### Runnable showcase examples

For end-to-end, copy-pasteable scripts on three different domains, see
[`examples/quickstarts/`](examples/quickstarts/):

| Script | Domain |
|--------|--------|
| `01_wikipedia_article.py` | Biographical / encyclopedic text |
| `02_financial_news.py` | Corporate news (M&A, funding rounds) |
| `03_biomedical.py` | Drug / disease / gene relationships |

Each script is self-contained: define schema, run extraction, dump JSON KG.

---

## CLI

```bash
# Extract from file
drg extract input.txt -o output.json

# From standard input
echo "Apple released iPhone 16" | drg extract - -o output.json

# With a custom model
drg extract input.txt -o output.json --model "gemini/gemini-2.0-flash-exp"

# Ollama (local)
drg extract input.txt -o output.json \
  --model "ollama_chat/llama3" \
  --base-url "http://localhost:11434"
```

---

## API Server & UI

```bash
pip install -e ".[api]"
python examples/api_server_example.py
# UI:    http://localhost:8000
# Docs:  http://localhost:8000/docs
```

Details and endpoint list: [`docs/api_server.md`](docs/api_server.md).

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
pip install -e ".[dev]"

# Non-integration tests (no API key required)
pytest -m "not integration"

# With coverage
pytest -m "not integration" --cov=drg --cov-report=term-missing

# Lint + format
ruff check drg tests examples
ruff format drg tests examples

# Type check
mypy drg

# Install pre-commit hooks
pre-commit install
```

---

## Supported Models

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

> Currently Turkish; English translations are tracked in the roadmap.

- [`docs/project_overview.md`](docs/project_overview.md) — Architecture &
  philosophy
- [`docs/setup.md`](docs/setup.md) — Detailed setup
- [`docs/api_server.md`](docs/api_server.md) — API + UI usage
- [`docs/pipeline_overview.md`](docs/pipeline_overview.md) — Pipeline flow
- [`docs/schema_design.md`](docs/schema_design.md) — Schema design principles
- [`docs/chunking_strategy.md`](docs/chunking_strategy.md) — Chunking strategies
- [`docs/relationship_model.md`](docs/relationship_model.md) — Relationship model
- [`docs/clustering_summarization.md`](docs/clustering_summarization.md) —
  Clustering
- [`docs/optimizer_design.md`](docs/optimizer_design.md) — DSPy optimizer
- [`docs/mcp_integration.md`](docs/mcp_integration.md) — MCP integration

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
