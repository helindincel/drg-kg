# Dataset-Agnostic Semantic Pipeline: Overview

> 🇹🇷 **Türkçe sürüm:** [`pipeline_overview.tr.md`](pipeline_overview.tr.md)

## 1. Architectural Principles

### 1.1 Dataset-Agnostic Design

The pipeline is designed to operate independently of any specific data
source. This agnosticism rests on the following principles:

- **Abstraction layers**: Clean interfaces between data sources, chunking
  strategies, and embedding models.
- **Pluggable components**: Each component can be swapped and tested
  independently.
- **Metadata preservation**: Each chunk carries rich metadata about its
  origin data source and processing history.
- **Domain adaptation**: Domain-specific optimisations can be added
  without modifying the core pipeline.

### 1.2 Monolithic-Modular Architecture

The system is composed of modular components inside a monolithic
structure:

- **Monolithic**: All components live in the same codebase and ship as a
  single deployment unit.
- **Modular**: Each component talks to others through independent
  interfaces.
- **Loose coupling**: Dependencies between components are minimal and
  explicit.
- **High cohesion**: Related functionality is grouped into a single
  module.

## 2. Pipeline Flow Diagram (Conceptual)

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA INGESTION LAYER                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │  Text    │  │  PDF     │  │  Markdown│  │  JSON    │         │
│  │  Files   │  │  Docs    │  │  Files   │  │  Streams │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │             │              │             │              │
│       └─────────────┴──────────────┴─────────────┘              │
│                          │                                      │
│                    [Normalizer]                                 │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CHUNKING & SEGMENTATION LAYER                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Token-Based Chunker (512–1024 token windows)        │       │
│  │  - Overlap strategy: 10–20 % sliding window          │       │
│  │  - Boundary detection: Sentence/paragraph aware      │       │
│  │  - Metadata injection: chunk_id, sequence_idx, origin│       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                      │
│                    [Chunk Validator]                            │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC ENRICHMENT LAYER                    │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Semantic Tagging                                    │       │
│  │  - Topic Classification                              │       │
│  │  - Entity Recognition (NER)                          │       │
│  │  - Intent Detection                                  │       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                      │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Embedding Abstraction Layer                         │       │
│  │  - OpenAI Embeddings (text-embedding-3-small/large)  │       │
│  │  - Gemini Embeddings (embedding-001)                 │       │
│  │  - OpenRouter (unified API)                          │       │
│  │  - Local Models (optional)                           │       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH LAYER                        │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Entity & Relation Extraction (DRG)                  │       │
│  │  - Schema-based extraction                           │       │
│  │  - Graph construction                                │       │
│  │  - Node/Edge metadata                                │       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                      │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Graph Database (e.g. Neo4j, NetworkX in-memory)     │       │
│  │  - Node embeddings (optional)                        │       │
│  │  - Edge weights                                      │       │
│  │  - Graph algorithms                                  │       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                CLUSTERING & SUMMARISATION LAYER                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Graph Clustering                                    │       │
│  │  - Louvain algorithm                                 │       │
│  │  - Leiden algorithm                                  │       │
│  │  - Spectral clustering                               │       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                      │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Cluster Summarisation                               │       │
│  │  - Per-cluster summary generation                    │       │
│  │  - Community report generation                       │       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                      │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Component Responsibilities

### 3.1 Ingestion Layer

**Responsibilities:**
- Multi-format support (text, PDF, Markdown, JSON, …).
- Format normalisation (all formats → unified text representation).
- Encoding handling (UTF-8, Latin-1, …).
- Metadata extraction (filename, date, source info).

**Design decisions:**
- Format-specific parsers must be pluggable.
- The normalisation pipeline should be reversible (for debugging).
- Metadata schema must be extensible.

### 3.2 Chunking Layer

**Responsibilities:**
- Token-based segmentation (512–1024 token windows).
- Overlap strategy application.
- Boundary detection (sentence/paragraph-aware).
- Chunk metadata injection.

**Design decisions:**
- Tokenizer abstraction (different tokenizers must be supported).
- Overlap strategy must be configurable.
- Chunk-ID generation: deterministic and unique.
- Sequence index: ordering inside the original document.

### 3.3 Semantic Enrichment Layer

**Responsibilities:**
- Semantic tagging (topic, entity, intent).
- Embedding generation (via the abstraction layer).
- Metadata enrichment.

**Design decisions:**
- Embedding provider abstraction: OpenAI, Gemini, OpenRouter, Local.
- Semantic tagging should be optional (cost / performance trade-off).
- The tagging model should be independent of chunking.

### 3.4 (Out of Scope) Vector Store Layer

Because this repo does not target a "serving / search" layer, the
vector-based index / similarity component is **out of scope** and has
been removed from the code.

If similarity-based helper signals are needed (e.g. as an entity-merge
support signal), they can be added optionally on top of the
**embedding provider** layer.

### 3.5 Knowledge Graph Layer

**Responsibilities:**
- Entity extraction (DRG schema-based).
- Relation extraction.
- Graph construction.
- Graph storage.

**Design decisions:**
- DRG schema must be declarative.
- Graph database abstraction (Neo4j, NetworkX, …).
- Node/edge metadata preservation.

### 3.6 Query & Analytics Helpers

**Responsibilities:**
- Query and analysis helpers on top of the knowledge graph
  (graph traversal, neighbourhood expansion).
- Community report generation and summarisation.
- Data preparation for export / visualisation.

**Design decisions:**
- This project is **not** a "serving / search framework"; the focus is
  **KG extraction + graph analysis / output**.
- KG-centric analysis / query helpers should be modular.

### 3.7 Clustering & Summarisation Layer

**Responsibilities:**
- Graph clustering (Louvain, Leiden, Spectral).
- Cluster summarisation.
- Community report generation.

**Design decisions:**
- Clustering algorithm is pluggable.
- Summarisation strategy is per-cluster.
- Report format is extensible.

## 4. Data Flow and Metadata Preservation

### 4.1 Chunk Metadata Schema

Each chunk carries the following metadata:

```
{
  "chunk_id": "unique_identifier",
  "sequence_index": 0,
  "origin_dataset": "dataset_name",
  "origin_file": "source_file_path",
  "token_count": 512,
  "char_count": 2048,
  "semantic_tags": {
    "topic": ["technology", "AI"],
    "entities": ["Apple", "iPhone"],
    "intent": "informational"
  },
  "embedding_model": "openai/text-embedding-3-small",
  "extraction_timestamp": "2025-01-XX",
  "chunk_text": "..."
}
```

### 4.2 Graph Node Metadata

Each graph node carries the following metadata:

```
{
  "node_id": "entity_name",
  "entity_type": "Company",
  "source_chunks": ["chunk_id_1", "chunk_id_2"],
  "embedding": [0.1, 0.2, ...],
  "extraction_confidence": 0.95,
  "first_seen": "chunk_id_1",
  "frequency": 5
}
```

### 4.3 Graph Edge Metadata

Each graph edge carries the following metadata:

```
{
  "source": "entity_1",
  "target": "entity_2",
  "relation": "produces",
  "source_chunks": ["chunk_id_1"],
  "weight": 1.0,
  "extraction_confidence": 0.92
}
```

## 5. Design Trade-offs

### 5.1 Chunking Trade-offs

**Token window size:**
- **512 tokens**: more chunks, finer-grained context, higher cost.
- **1024 tokens**: fewer chunks, broader context, lower storage cost.
- **Decision**: configurable between 512 and 1024, default 768.

**Overlap strategy:**
- **10 % overlap**: less redundancy, lower cost, higher risk of losing
  context at entity boundaries.
- **20 % overlap**: more redundancy, higher cost, better entity
  preservation.
- **Decision**: 15 % default, configurable for domain-specific tuning.

### 5.2 Embedding Trade-offs

**Model selection:**
- **OpenAI `text-embedding-3-small`**: fast, cheap, 1536 dimensions.
- **OpenAI `text-embedding-3-large`**: slower, more expensive, 3072
  dimensions, higher quality.
- **Gemini `embedding-001`**: alternative provider, different semantic
  space.
- **OpenRouter**: unified API, multi-model support.

**Decision criteria:**
- **Cost**: token-based pricing, batch-processing optimisations.
- **Latency**: real-time vs. batch use cases.
- **Portability**: model lock-in risk.
- **Semantic consistency**: cross-domain performance.

### 5.3 Query / Analytics Trade-offs

**KG query & analysis:**
- **Graph traversal**: strong for relational questions; depends on graph
  quality.
- **Community reports**: aid interpretability on large graphs; subject
  to generation cost.

**Decision criteria:**
- **Graph quality**: if extraction quality is poor, traversal results
  also degrade.
- **Latency requirements**: online vs. batch analysis.

## 6. Extensibility and Extension Points

### 6.1 Pluggable Components

- **Chunking strategy**: token-based, sentence-based, paragraph-based,
  semantic-based.
- **Embedding provider**: OpenAI, Gemini, OpenRouter, local models.
- **Graph database**: Neo4j, NetworkX, ArangoDB.
- **Clustering algorithm**: Louvain, Leiden, Spectral, custom.

### 6.2 Domain Adaptation

Domain-specific optimisations can be added without modifying the core
pipeline:

- **Domain-specific chunking**: code-aware chunking for technical docs.
- **Domain-specific tagging**: ICD-10 tagging for the medical domain.
- **Domain-specific schemas**: DRG schemas tailored to a domain.

## 7. Evaluation Methodology

### 7.1 Pipeline Metrics

- **Chunking quality**: entity-boundary preservation, semantic coherence.
- **Embedding quality**: semantic-similarity accuracy, cross-domain
  consistency.
- **Graph quality**: entity-extraction F1, relation-extraction F1, graph
  completeness, duplicate-entity rate.

### 7.2 Multi-Dataset Evaluation

Evaluation against 3–4 heterogeneous datasets:

- Long narrative text (a 20-page story).
- Factual text (Wikipedia biography).
- Technical / structured document.
- Informal dialogue (chat / forum).

For each dataset:

- Chunking-quality analysis.
- KG-extraction quality metrics (entity/relation F1, duplicate rate,
  cross-chunk edge retention).
- Entity-extraction effectiveness.
- Failure cases and edge behaviours.

### 7.3 Comparison Framework

For this repo, the axis of comparison is **not** "serving / search
frameworks"; it is the quality / robustness impact of the pipeline
components themselves:

- Effect of chunking strategies on extraction quality.
- Effect of schema-sampling strategies on coverage.
- Effect of coreference / entity-resolution post-processing.
- Effect of cross-chunk context injection (deterministic snippets).
