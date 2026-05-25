# DRG (Declarative Relationship Generation) — Project Overview

> 🇹🇷 **Türkçe sürüm:** [`project_overview.tr.md`](project_overview.tr.md)

## 1) What is DRG?

**DRG (Declarative Relationship Generation)** is a **dataset-agnostic**,
**DSPy-based**, **declarative** semantic pipeline designed to produce
**knowledge graphs (KG)** from text. Its core goals are to:

- Take raw text input,
- (Optionally) auto-generate an **EnhancedDRGSchema** from the text,
- Run **chunk-based extraction** to derive **entity + relation** triples,
- Build the result as an **EnhancedKG**,
- Run **clustering** and produce **community reports** over the KG,
- Export results as JSON and visualise them through the UI.

As a research-grade codebase, DRG generates the "raw material" for
**GraphRAG / RAG experimentation**: the primary focus is **KG extraction
plus graph analysis / output**, not retrieval-augmented generation itself.

---

## 2) What DRG Is *Not* (Explicit Non-Goals)

This section matters: DRG is easy to mistake for a RAG project. It isn't.

- **Not a RAG framework.**
  - DRG's UI query endpoint performs **deterministic KG lookup**.
  - There is no goal of "answer questions with an LLM" /
    "retrieval-augmented generation serving".
- **Not a serving / search platform.**
  - A vector-store layer (Chroma / Qdrant / Faiss / …) is **not** a core
    product component of this repo.
  - It can be plugged in as an experimental / optional element if you
    actually need it.
- **Not provider-locked.**
  - LLM and embedding providers are abstracted behind interfaces;
    selection happens via environment variables.

In short: DRG is a **"graph-producing and graph-analysing pipeline"**, not
a **"query-answering application"**.

---

## 3) Why DSPy? What Does "Declarative Extraction" Mean?

DRG's main distinguishing trait is that extraction is expressed in a
**declarative** form rather than as ad-hoc, prompt-heavy code.

- **Declarative approach**: "What should I extract?" instead of "How
  should I extract it?".
  - The user / researcher defines the schema (entity types, relation
    groups, relations, descriptions, examples).
  - DRG then generates **dynamic signatures** from this schema through
    DSPy and runs the extraction program.
- **DSPy**: Treats LLM calls as a *program*, making structured outputs
  and systematic flow easier to express.
  - DRG modularises entity / relation extraction through DSPy (e.g. use
    `TypedPredictor` if available, fall back gracefully otherwise).

As a result, DRG:
- Can run the same pipeline against different datasets,
- Changes extraction behaviour predictably when the schema changes,
- Makes research / experiment design reproducible.

---

## 4) Dataset-Agnostic Design and the "Enhanced Schema"

### 4.1 Dataset-Agnostic

DRG is not hard-coded to any specific domain or dataset. To make this
possible:
- **Abstraction layers** separate ingestion, chunking, embedding, graph
  construction, and clustering.
- Chunks and graph elements carry **rich metadata** (origin, chunk_id,
  processing history).

### 4.2 EnhancedDRGSchema

DRG's preferred schema format is **`EnhancedDRGSchema`**:
- **`EntityType`**: `name`, `description`, `examples`, `properties`.
- **`RelationGroup`**: organises relations semantically into groups.
- **`Relation`**: `name`, `src`, `dst` plus descriptive fields.
  - Carrying `description` / `detail` fields on relations is important
    here — they capture *why* something was linked and *in what
    context*.

Schemas can be used in two ways:
- **Manual schema**: You define everything directly with your domain
  knowledge.
- **Auto schema generation**: You derive a schema from the text via
  `generate_schema_from_text()`.

---

## 5) Pipeline (High-Level Flow)

DRG's "default" conceptual flow is:

1. **Text Input**
2. **Schema Generation / Load**
3. **Chunking**
4. **KG Extraction (chunk-based)**
5. **(Optional) Embeddings**
6. **Clustering**
7. **Community Reports**
8. **Export (JSON) + UI Visualisation**

### 5.1 Why Chunk-Based?

In long documents:
- Entities can appear in one chunk while their relations sit in another.
- DRG therefore uses chunk-aware extraction and optional **cross-chunk
  context injection** techniques.

### 5.2 KG Construction Logic

Extraction outputs are:
- `entities`: list of `(entity_name, entity_type)`.
- `triples`: list of `(source, relation, target)`.

From these:
- An `EnhancedKG` is built (`KGNode`, `KGEdge`, `Cluster`).
- Node / edge metadata is preserved, enabling downstream analytic steps
  over the graph.

---

## 6) Monolithic-Modular Architecture

DRG is monolithic at the codebase / deployment-unit level, but **modular
inside**:
- Each layer owns its concern.
- Components decouple through interfaces.
- The goal is **loose coupling, high cohesion**.

For a research-grade code base this is pragmatic:
- Adding / removing experimental components is easy,
- A change in one component (e.g. chunking) has minimal impact on others.

---

## 7) UI and Query Behaviour (Important: No LLM)

The DRG UI:
- Visualises the KG (Cytoscape-based).
- "Load Full Graph" draws the entire graph.
- "Load Communities" shows the cluster-coloured view.

### 7.1 UI Query = Deterministic KG Lookup

The query box in the UI performs:
- Entity string matching,
- Optional relation filtering,
- Neighbourhood expansion around a seed entity.

This endpoint **does not produce answers via RAG / LLM**. That keeps the
UI fast and deterministic.

### 7.2 Note on "Hub" Visualisation

Some texts naturally produce a "star-shaped" graph (many relations
clustered around a single central company / character). DRG ships a
**UI-only anti-hub** option:
- Adds proxy nodes to improve the readability of the layout,
- Does **not** mutate the KG data (UI-only).

---

## 8) Configuration (Environment-Driven)

DRG's behaviour is controlled through environment variables (this is what
research + reproducibility demand).

Examples:

- `DRG_MODEL`: LLM model id (with provider prefix).
- `DRG_TEMPERATURE`: LLM temperature.
- `DRG_MAX_TOKENS`: LLM output budget.
- Provider keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`,
  `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, …
- Chunking: `DRG_CHUNK_SIZE`, `DRG_OVERLAP_RATIO`,
  `DRG_CHUNKING_STRATEGY`.
- UI: `DRG_API_PORT`, plus optional hub / visualisation parameters.

The benefits:
- Secrets stay out of the code,
- Experimental conditions are easy to reproduce.

---

## 9) Repository Structure

Summary tree:

```
DRG/
├── drg/                       # Core library (monolithic codebase)
│   ├── extract/               # DSPy extraction (KGExtractor + cross-chunk + heuristics)
│   ├── chunking/              # Chunking strategies + validators
│   ├── embedding/             # Embedding provider abstraction
│   ├── graph/                 # EnhancedKG, schema generation, community reports, vis adapters
│   ├── clustering/            # Louvain / Leiden / Spectral + summarisation
│   ├── coreference_resolution/# Pronoun resolution (NLP + heuristic strategies)
│   ├── entity_resolution/     # Entity merging (string + hybrid)
│   ├── optimizer/             # DSPy optimiser experiments
│   ├── api/                   # FastAPI server + UI templates
│   ├── schema.py              # EnhancedDRGSchema, EntityType, RelationGroup, …
│   ├── extract.py             # DSPy extraction logic
│   └── cli.py                 # CLI
├── docs/                       # Documentation (NO CODE)
├── examples/                   # Example scripts (full pipeline, API server)
├── tests/                      # Unit / integration tests
├── outputs/                    # Generated outputs (KG, schema, reports)
├── pyproject.toml              # Project configuration
└── README.md                   # User entry point
```

Note: `docs/` is documentation-only; it must not contain code.

---

## 10) Typical Usage Scenarios

### 10.1 "Produce a KG from text"
- Pass in the text,
- Generate or load a schema,
- Run chunking + extraction to produce the KG,
- Receive the outputs as JSON.

### 10.2 "Graph analysis"
- Run clustering,
- Produce cluster summarisation + community reports,
- Evaluate using graph quality metrics / heuristics.

### 10.3 "Explore via the UI"
- Launch the server with `examples/api_server_example.py`,
- Inspect the KG visually,
- Verify relations quickly through deterministic queries.

---

## 11) DRG vs. Other Projects (Short Summary)

- **Not RAG / serving**: A "graph-producing pipeline", not an
  "answer-producing system".
- **DSPy + declarative**: Extraction behaviour is defined by the schema;
  prompt churn is minimised.
- **Dataset-agnostic**: The same system can be reused across domains.
- **Enhanced schema + metadata**: Rich representation via
  `EntityType` / `RelationGroup` together with `description` / `detail`
  fields.
- **Graph-first analytics**: Graph-native outputs such as clustering and
  community reports.
