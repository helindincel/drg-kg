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

### 🕸 Graph Intelligence
- **Incremental Ingestion**: Add new documents to an existing graph with automated entity resolution and relationship deduplication.
- **Temporal KG**: Native support for `valid_from`/`valid_to` metadata, partial dates, and timeline building.
- **Multi-Document Reasoning**: Rule-based inference engine that discovers cross-document bridges (e.g., A knows B, B knows C → A connected to C).
- **Clustering & Communities**: Automated community detection (Louvain, Leiden) with LLM-powered group summarization.

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

---

## ⚡ Quickstart

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

### 2. Auto-Schema Ingestion
```bash
# Automatically infer schema and extract KG
drg extract sample.txt --auto-schema -o output_kg.json
```

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
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

MIT © [Helin Dinçel](https://github.com/helindincel)
