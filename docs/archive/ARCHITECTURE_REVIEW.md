# DRG Repository: Principal Architect Review

**Objective under review:** `Document → Declarative Schema Discovery → Knowledge Graph Extraction`

---

## 1. Repository Architecture Map

```
drg/
├── schema.py                    ← CORE: DRGSchema, EnhancedDRGSchema types
├── config.py                    ← CORE: DSPy LM configuration
├── errors.py                    ← CORE: typed error hierarchy
├── protocols.py                 ← CORE: ABC/Protocol contracts
│
├── extract/                     ← CORE: The primary extraction engine
│   ├── _signatures.py           ← CORE: DSPy Signature factories (schema-driven)
│   ├── _schema_gen.py           ← CORE: Auto-schema from text (generate_schema_from_text)
│   ├── _chunk_context.py        ← CORE: Cross-chunk anchor selection
│   ├── _heuristics.py           ← CONDITIONAL: English-only negation/temporal rules
│   ├── _parsing.py              ← CORE: JSON output parsing
│   ├── _relations.py            ← CORE: Relation normalization
│   ├── _types.py                ← CORE: Extraction result types
│   └── __init__.py              ← CORE: KGExtractor, extract_typed, extract_from_chunks
│
├── chunking/                    ← CORE: Text chunking (sentence, token, overlap)
│   ├── strategies.py            ← CORE: Chunking strategy implementations
│   └── validators.py            ← CORE: Chunk validation
│
├── graph/                       ← CORE (mostly): KG persistence layer
│   ├── kg_core.py               ← CORE: KGNode, KGEdge, EnhancedKG
│   ├── builders.py              ← CORE: build_enhanced_kg
│   ├── provenance.py            ← CORE: ProvenanceRecord, attach/find provenance
│   ├── diff.py                  ← CORE: SnapshotDiff (needed for versioning)
│   ├── incremental.py           ← CORE: GraphMerger (incremental updates)
│   ├── versioning.py            ← CORE: create_snapshot, rollback
│   ├── validation.py            ← CORE: schema validation of KG
│   ├── neo4j_exporter.py        ← OPTIONAL: Neo4j export (external dependency)
│   ├── visualization.py         ← OPTIONAL: Mermaid / PyVis export
│   ├── visualization_adapter/   ← OPTIONAL: Cytoscape / D3 / vis-network JSON
│   ├── community_report.py      ← OPTIONAL: Cluster community reports
│   ├── auto_clusters.py         ← OPTIONAL: Connected-components fallback clusters
│   ├── hub_mitigation.py        ← REMOVE: Visualization optimization only
│   ├── schema_generator.py      ← REMOVE: Duplicate schema machinery (not wired to extract/)
│   ├── relationship_model/      ← SCOPE CREEP: Rule+LLM relation classifier
│   │   ├── _types.py            ← SCOPE CREEP: Hard-coded RelationshipType enum
│   │   ├── _rule_based.py       ← SCOPE CREEP: Keyword-matching classifier
│   │   ├── _classifier.py       ← SCOPE CREEP: Dispatcher (rules + LLM classify)
│   │   └── _llm_based.py        ← SCOPE CREEP: DSPy TypedPredictor for relation typing
│   ├── query_engine.py          ← OPTIONAL: Legacy UI query engine
│   └── _legacy.py               ← REMOVE: Legacy KG shim
│
├── query/                       ← OPTIONAL (but coherent)
│   ├── _backend.py              ← OPTIONAL: QueryBackend Protocol
│   ├── _engine.py               ← OPTIONAL: GraphQuery facade
│   ├── _traversal.py            ← OPTIONAL: BFS/DFS multi-hop paths
│   ├── _search.py               ← OPTIONAL: Token-based entity search (no vectors)
│   ├── _communities.py          ← OPTIONAL: Community-aware queries
│   ├── _analytics.py            ← OPTIONAL: PageRank, degree centrality
│   ├── _temporal.py             ← OPTIONAL: Temporal graph queries
│   ├── _explain.py              ← OPTIONAL: Path explanation builder
│   ├── _evidence.py             ← OPTIONAL: Edge/node to view converters
│   ├── _memory.py               ← OPTIONAL: InMemoryBackend
│   └── _types.py                ← OPTIONAL: Query result types
│
├── reasoning/                   ← OPTIONAL: Rule-based graph inference
│   ├── _engine.py               ← OPTIONAL: MultiDocumentReasoner
│   ├── _rules.py                ← OPTIONAL: Default inference rules
│   ├── _explain.py              ← OPTIONAL: Inference explanation
│   └── _types.py                ← OPTIONAL: ReasoningConfig, InferenceRule
│
├── temporal/                    ← OPTIONAL: Temporal metadata on edges
│   ├── _types.py                ← OPTIONAL: TemporalScope
│   ├── _compare.py              ← OPTIONAL: Interval overlap/active-at checks
│   ├── _migrate.py              ← OPTIONAL: Legacy temporal data migration
│   └── _reasoning.py            ← OPTIONAL: Timeline, conflict detection
│
├── confidence/                  ← OPTIONAL: Confidence scoring framework
│   ├── _strategy.py             ← OPTIONAL: ConfidenceStrategy Protocol
│   ├── _default.py              ← OPTIONAL: DefaultConfidenceStrategy
│   └── _types.py                ← OPTIONAL: ConfidenceScore
│
├── entity_resolution/           ← CORE: Entity deduplication within a doc
│   ├── _normalize.py            ← CORE: Name normalization
│   ├── _resolver.py             ← CORE: EntityResolver
│   ├── _similarity.py           ← CORE: String + cosine similarity (no vector store)
│   └── _strategy.py             ← CORE: Resolution strategy
│
├── coreference_resolution/      ← OPTIONAL (heavy NLP dependency)
│   ├── _strategy.py             ← OPTIONAL: Strategy Protocol
│   ├── _heuristic_strategy.py   ← OPTIONAL: Pronoun-based heuristics
│   ├── _nlp_strategy.py         ← OPTIONAL: spaCy + coreferee/neuralcoref
│   ├── _pronouns.py             ← OPTIONAL: Pronoun maps
│   └── _scoring.py              ← OPTIONAL: Action/SVO scoring
│
├── clustering/                  ← OPTIONAL: Graph community detection
│   ├── algorithms.py            ← OPTIONAL: Louvain, Leiden, Spectral
│   └── summarization.py         ← OPTIONAL: LLM or template cluster summaries
│
├── events/                      ← SCOPE CREEP: Event extraction sub-pipeline
│   ├── _extraction.py           ← SCOPE CREEP: Full separate DSPy pipeline
│   ├── _graph_mapping.py        ← SCOPE CREEP: Events → KG nodes
│   ├── _postprocess.py          ← SCOPE CREEP: Event normalization
│   ├── _registry.py             ← SCOPE CREEP: EventType registry
│   └── _types.py                ← SCOPE CREEP: Event, EventRole, EventProvenance
│
├── embedding/                   ← SCOPE CREEP: Embedding providers (not used in KG core)
│   └── providers.py             ← SCOPE CREEP: OpenAI, Gemini, SentenceTransformers
│
├── optimizer/                   ← SCOPE CREEP: DSPy optimizer pipeline
│   ├── optimizer.py             ← SCOPE CREEP: DRGOptimizer (BootstrapFewShot, MIPRO, etc.)
│   └── metrics.py               ← SCOPE CREEP: ExtractionMetrics for optimizer
│
├── evaluation/                  ← OPTIONAL: Quality metrics
│   ├── _metrics.py              ← OPTIONAL: P/R/F1 for entities/relations
│   ├── _adapters.py             ← OPTIONAL: Bridge to graph output
│   ├── _performance.py          ← OPTIONAL: Latency/cost benchmarking
│   ├── _reporting.py            ← OPTIONAL: Report generation
│   ├── _runner.py               ← OPTIONAL: Evaluation runner
│   └── _suite.py                ← OPTIONAL: EvaluationSuite
│
├── api/                         ← OPTIONAL: FastAPI REST server
│   └── server.py                ← OPTIONAL: Visualization + query endpoints
│
├── mcp_server.py                ← OPTIONAL: Official MCP server (FastMCP)
├── mcp_api.py                   ← REMOVE: Deprecated manual JSON-RPC shim
│
├── cli.py                       ← OPTIONAL: CLI entrypoint
└── utils/                       ← CORE: Shared utilities
    ├── env_loader.py            ← CORE: .env loading
    ├── llm_throttle.py          ← CORE: LLM rate limiting
    ├── logging.py               ← CORE: Structured logging
    ├── cache.py                 ← OPTIONAL: Embedding cache
    └── strict.py                ← CORE: Strict-mode flag
```

---

## 2. Component Classification Table

| Component | Classification | Why It Exists | Aligns with DRG Vision? | Maintenance Cost | Architectural Complexity |
|---|---|---|---|---|---|
| `schema.py` | **Core** | Declarative schema types (the "D" in DRG) | ✅ Yes — central abstraction | Low | Low |
| `config.py` | **Core** | DSPy LM wiring from env vars | ✅ Yes | Low | Low |
| `errors.py` | **Core** | Typed exception hierarchy | ✅ Yes | Negligible | Low |
| `protocols.py` | **Core** | ABC contracts for DI | ✅ Yes | Negligible | Low |
| `extract/` | **Core** | Schema-driven DSPy extraction pipeline | ✅ Yes — the primary value | Medium | Medium |
| `extract/_heuristics.py` | **Optional** | English-only negation/temporal post-proc | ⚠️ Marginal — hardcoded English rules | Medium | Medium |
| `chunking/` | **Core** | Text → chunks for the extraction pipeline | ✅ Yes | Low | Low |
| `graph/kg_core.py` | **Core** | KGNode, KGEdge, EnhancedKG | ✅ Yes — the output type | Low | Low |
| `graph/builders.py` | **Core** | Triples + entities → EnhancedKG | ✅ Yes | Low | Low |
| `graph/provenance.py` | **Core** | Source traceability on nodes/edges | ✅ Yes | Low | Low |
| `graph/validation.py` | **Core** | Schema compliance checks on extracted KG | ✅ Yes | Low | Low |
| `graph/incremental.py` | **Core** | Merge new doc into existing KG | ✅ Yes — lifecycle | Medium | Medium |
| `graph/diff.py` | **Core** | Structural change tracking | ✅ Yes | Low | Low |
| `graph/versioning.py` | **Core** | Snapshot-based KG version history | ✅ Yes — lifecycle | Low | Medium |
| `entity_resolution/` | **Core** | Entity dedup within document | ✅ Yes | Low | Medium |
| `confidence/` | **Optional** | Score confidence of extracted facts | ✅ Useful | Low | Low |
| `temporal/` | **Optional** | Temporal metadata on KG edges | ✅ Useful for temporal KGs | Low | Medium |
| `reasoning/` | **Optional** | Rule-based graph inference (no LLM) | ✅ Useful, deterministic | Low | Medium |
| `query/` | **Optional** | Deterministic graph query layer | ✅ Yes — part of lifecycle | Medium | Medium |
| `evaluation/` | **Optional** | P/R/F1 and graph quality metrics | ✅ Useful for iterating on schemas | Medium | Medium |
| `coreference_resolution/` | **Optional** | Pronoun/alias resolution in extraction | ✅ Marginally useful | High (spaCy dep) | High |
| `clustering/` | **Optional** | Louvain/Leiden/Spectral graph communities | ⚠️ Useful for analysis, not extraction | Medium | Medium |
| `graph/community_report.py` | **Optional** | Human-readable cluster summaries | ⚠️ Useful for analysis | Low | Low |
| `graph/auto_clusters.py` | **Optional** | Connected-components cluster fallback | ⚠️ UI convenience only | Low | Low |
| `api/` | **Optional** | FastAPI REST + visualization UI | ⚠️ Integration surface; not core | High | High |
| `mcp_server.py` | **Optional** | MCP tool server (FastMCP) | ⚠️ Integration surface only | High | High |
| `graph/neo4j_exporter.py` | **Optional** | Export to external graph DB | ⚠️ Downstream concern | Medium | Medium |
| `graph/visualization.py` | **Optional** | Mermaid / PyVis rendering | ⚠️ Developer convenience | Low | Low |
| `graph/visualization_adapter/` | **Optional** | JS library JSON formats | ⚠️ UI-serving concern | Medium | Medium |
| `cli.py` | **Optional** | CLI entrypoint | ⚠️ Integration surface | Low | Low |
| **`events/`** | **Scope Creep** | Separate DSPy sub-pipeline for events | ❌ Parallel to core extraction, not part of it | High | High |
| **`embedding/providers.py`** | **Scope Creep** | Vector embedding providers | ❌ No vector search in DRG; embeddings not used in KG core path | High | Medium |
| **`optimizer/`** | **Scope Creep** | DSPy BootstrapFewShot/MIPRO/COPRO tuning | ❌ Meta-pipeline on top of extraction; not schema discovery | High | High |
| **`graph/relationship_model/`** | **Scope Creep** | Rule + LLM relation type classifier | ❌ Duplicates extraction logic; hardcoded RelationshipType enum | High | High |
| **`graph/schema_generator.py`** | **Remove** | Duplicate schema generation (disconnected from `extract/_schema_gen.py`) | ❌ Dead code; `generate_schema_from_text` already in `extract/` | Low | Low |
| **`graph/hub_mitigation.py`** | **Remove** | Visualization layout hack (proxy nodes) | ❌ Corrupts KG semantics for UI aesthetics | Low | Medium |
| **`graph/_legacy.py`** | **Remove** | Old KG shim | ❌ Backward compat only | Low | Low |
| **`mcp_api.py`** | **Remove** | Manually coded JSON-RPC (deprecated) | ❌ Already self-deprecated; misleads users | Low | Low |
| `utils/` | **Core** | Shared infra (logging, env, throttle) | ✅ Yes | Low | Low |
| `utils/cache.py` | **Optional** | Embedding cache | ⚠️ Only needed if embeddings used | Low | Low |

---

## 3. Dead Code Inventory

These files exist and are importable but serve no purpose in the core pipeline:

| File | Why Dead |
|---|---|
| `graph/schema_generator.py` | `PropertyDefinition`, `EntityClassDefinition`, `SchemaFileGenerator` are never called by any other module. Actual schema generation lives in `extract/_schema_gen.py`. This file exists in parallel but is not wired. |
| `graph/_legacy.py` | Legacy `KG` class retained for backward compat. Re-exported from `graph/__init__.py` but the actual type used everywhere is `EnhancedKG`. `KG` has no active callers in the live pipeline. |
| `mcp_api.py` | Self-marked as deprecated at import time with a `DeprecationWarning`. Has no transport. Cannot work as a real MCP server. The surface it exposed (`DRGMCPAPI`) is not part of any documented or tested integration. |
| `utils/cache.py` | Embedding cache exists but `embedding/providers.py` has no call site in the core extraction path (`KGExtractor` does not call embeddings). Cache is only useful if embeddings are wired — which they are not in the default pipeline. |
| `graph/hub_mitigation.py` | Called only from the visualization adapter path to split hub nodes into proxy nodes before rendering. Introduces `HubProxy` nodes into the graph in-place — a structural mutation done purely for rendering aesthetics. No test coverage validates that downstream queries still work correctly after proxy injection. |
| `optimizer/metrics.py` | `ExtractionMetrics.calculate_metrics` reimplements the same precision/recall/F1 already in `evaluation/_metrics.py`. Never imported by `optimizer.py`. |

---

## 4. Scope Creep Inventory

### 4.1 `drg/embedding/providers.py` — Vector Embedding Infrastructure

**What it is:** Full embedding provider framework with OpenAI, Gemini, SentenceTransformer backends, batching, dimension introspection, and cost tracking.

**Why it's scope creep:** The DRG project statement explicitly says *"not a vector database or search layer."* Embedding vectors appear as an optional field on `KGNode.embedding` and `KGEdge.embedding`, but nothing in the core extraction pipeline (`KGExtractor`, `build_enhanced_kg`, `GraphMerger`) ever calls an embedding provider. The `entity_resolution/_similarity.py` module uses `cosine_similarity` but only when a caller pre-populates embedding vectors manually. The infrastructure for generating those vectors is fully built but lives outside the pipeline.

**Risk:** Encourages users to think DRG is a vector/semantic-search layer. Adds three optional dependencies (`openai`, `google-generativeai`, `sentence-transformers`) to the install matrix.

---

### 4.2 `drg/optimizer/` — DSPy Prompt Optimization Pipeline

**What it is:** A complete DSPy few-shot learning and prompt optimization framework (BootstrapFewShot, MIPRO, COPRO, LabeledFewShot). It wraps `KGExtractor`, maintains training examples, evaluates extractions, and iteratively improves LLM prompts.

**Why it's scope creep:** The DRG vision is *declarative schema-driven extraction*, not iterative prompt tuning. Once you have a schema, the extraction should run deterministically against that schema. The optimizer introduces a meta-layer that: (a) requires labeled ground-truth triples, (b) makes repeated LLM calls for optimization, (c) defeats the "declarative" principle by moving intelligence into learned few-shot prompts rather than the schema itself. This is closer to prompt-engineered retrieval tooling than schema-driven extraction.

**Risk:** Conflates "improve the schema" (the right DRG lever) with "improve the prompt" (a pipeline-tuning approach).

---

### 4.3 `drg/events/` — Separate Event Extraction Sub-Pipeline

**What it is:** A complete parallel extraction pipeline for structured events (`Event`, `EventRole`, `EventTypeRegistry`). Has its own DSPy Signature factory, post-processor, graph mapper, and registry.

**Why it's scope creep:** Events can be naturally expressed as entities and relations within the schema. The DRG schema system already supports rich entity types, properties, and relation groups — an `Acquisition` event with `acquirer`, `target`, `date`, and `price` is just an entity type with relations. Building a full parallel sub-pipeline for events creates two divergent extraction paths, doubles maintenance overhead, and introduces a bespoke registry system that the schema layer already covers.

**Risk:** Any schema change now needs to be reflected in two places. The event sub-pipeline duplicates DSPy wiring, error handling, and post-processing already established in `extract/`.

---

### 4.4 `drg/graph/relationship_model/` — Relationship Type Classifier

**What it is:** A rule-based + LLM-backed relation type classifier with a hard-coded `RelationshipType` enum (`CAUSES`, `LOCATED_AT`, `COLLABORATES_WITH`, etc.), keyword patterns, and a DSPy TypedPredictor path.

**Why it's scope creep:** The entire purpose of the DRG schema is to declare what relation types are valid. Introducing a classifier that re-derives relation types from keywords and LLM calls is a contradiction of the declarative principle: the schema *is* the classifier. The hard-coded `RelationshipType` enum forces all schemas into a common universal ontology, which conflicts with DRG's domain-agnostic positioning. This is a pattern seen in retrieval-oriented and traditional NLP relation extraction tools.

**Risk:** Adds a second, implicit "schema" (the `RelationshipType` enum) that overrides the user's explicit schema. Contradicts the core value proposition.

---

### 4.5 `drg/clustering/` (Louvain/Leiden/Spectral) + `clustering/summarization.py`

**What it is:** Three community-detection algorithms (Louvain, Leiden, Spectral) plus an LLM-based cluster summarizer.

**Why it's borderline:** Clustering over an extracted KG is analytically useful, but it is a post-extraction analysis step, not extraction itself. The `ClusterSummarizer` with `use_llm=True` re-introduces an LLM call after extraction is complete, moving toward community-report generation instead of declarative KG construction.

**Risk:** The LLM-backed summarizer path in `ClusterSummarizer` and the `CommunityReportGenerator` are the clearest signal of retrieval-oriented scope creep in the codebase.

---

### 4.6 `drg/api/` + `drg/mcp_server.py` — Integration Surfaces as First-Class Citizens

**What they are:** A full FastAPI REST server with CORS, API key auth, static files, visualization endpoints, query endpoints, and a Neo4j sync endpoint; plus a FastMCP server exposing DRG tools to LLM agents.

**Why it's borderline:** These are valid downstream integration points, but they carry substantial complexity. The API server has become a second maintenance surface with its own error handling, input validation, and security surface area. The MCP server's in-memory state management (`_schemas`, `_knowledge_graphs` dicts) means KGs are lost on restart.

**Risk:** The API surface growing ahead of the library surface means bugs can appear in integration layers before the core is stable (Alpha status).

---

## 5. Specific Anti-Patterns Flagged

### 5.1 Retrieval-Oriented Community Logic

| Location | Pattern |
|---|---|
| `graph/community_report.py` | Generates `CommunityReport` with `summary`, `top_actors`, `top_relationships`, `themes` |
| `clustering/summarization.py` | `ClusterSummarizer(use_llm=True)` calls DSPy to produce narrative cluster descriptions post-extraction |
| `query/_communities.py` | Community-based query routing moves beyond direct graph operations |

### 5.2 Search-Oriented Fallbacks

| Location | Pattern |
|---|---|
| `query/_search.py` | `_score_entity()` is token-based (good), but the design anticipates a vector similarity fallback via the `embedding` field on nodes |
| `entity_resolution/_similarity.py` | `cosine_similarity()` ready for embedding-backed matching even though no embedding provider is called in the core path |
| `graph/relationship_model/_enriched.py` | `EnrichedRelationship` carries similarity scores alongside extracted relation types |

### 5.3 Vector Retrieval Infrastructure (No Active Use)

`embedding/providers.py` provides a complete production-grade embedding client (batching, retry-friendly, dimension validation), but there are **zero call sites** in the core `Document → Schema → KG` pipeline. The `KGNode.embedding` field is populated only if callers explicitly attach embeddings, which no pipeline step does.

### 5.4 Prompt-Heavy DSPy

| Location | Degree |
|---|---|
| `extract/_signatures.py` | ✅ Lean — minimal InputField/OutputField; schema speaks for itself |
| `events/_extraction.py` | ⚠️ Medium — builds role descriptions and property lists as prompt content |
| `optimizer/optimizer.py` | ❌ Heavy — full MIPRO/COPRO pipelines effectively function as automated prompt engineers |
| `clustering/summarization.py` | ❌ Heavy — open-ended LLM call for narrative summaries with no schema constraint |
| `graph/relationship_model/_llm_based.py` | ❌ Heavy — DSPy TypedPredictor for relation classification outside the main schema |

### 5.5 Hardcoded Extraction Rules

| Location | Rule Type |
|---|---|
| `graph/relationship_model/_rule_based.py` | 13 hard-coded `(keyword_tuple, RelationshipType, confidence)` rules with English keywords like `"causes"`, `"located"`, `"influences"` |
| `graph/relationship_model/_types.py` | Hard-coded `RelationshipType` enum with 20+ fixed relation types |
| `extract/_heuristics.py` | English-only negation/temporal detection via regex; gated behind `DRG_LANGUAGE` env var but defaults to English-only |
| `extract/_relations.py` | Hard-coded `REVERSE_RELATION_PATTERNS` dictionary for bidirectional relation inference |

---

## 6. Refactoring Plan

### Phase 1 — Remove Dead Code (no behavior change)

| Action | Target |
|---|---|
| Delete | `graph/schema_generator.py` |
| Delete | `graph/_legacy.py` (retain `KG` re-export in `graph/__init__.py` for one version, then drop) |
| Delete | `mcp_api.py` |
| Delete | `optimizer/metrics.py` |
| Delete | `graph/hub_mitigation.py` |

### Phase 2 — Quarantine Scope Creep (move to `drg/extras/` or separate package)

| Action | Target | Reason |
|---|---|---|
| Move to `drg/extras/optimizer/` | `optimizer/optimizer.py` | DSPy prompt tuning is a meta-layer, not the core pipeline |
| Move to `drg/extras/events/` | `events/` | Parallel pipeline; can be expressed via schema |
| Move to `drg/extras/embedding/` | `embedding/providers.py`, `utils/cache.py` | No core pipeline usage |
| Move to `drg/extras/relationship_classifier/` | `graph/relationship_model/` | Contradicts declarative principle |
| Move to `drg/extras/community/` | `clustering/summarization.py` LLM path | Community-summary pattern; keep template path in `clustering/` |

### Phase 3 — Harden the Core Pipeline

| Action | Detail |
|---|---|
| Make `extract/_heuristics.py` purely opt-in | Remove default activation; require caller to enable heuristics explicitly |
| Decouple `REVERSE_RELATION_PATTERNS` | Move to user-configurable schema property instead of hard-coded dict |
| Remove `KGNode.embedding` from core dataclass | Move to `extras/embedding/` enriched node type |
| Remove `cluster_summarizer` LLM path | Keep template-based summary, remove DSPy call from `ClusterSummarizer` |
| Define `QueryBackend` protocol in `protocols.py` | Currently defined only in `query/_backend.py` |

### Phase 4 — Simplify Integration Surfaces

| Action | Detail |
|---|---|
| Slim the API server | Remove visualization adapter, Neo4j sync, and provenance graph endpoints from the default API — expose only `POST /extract`, `GET /graph`, `GET /query` |
| Make MCP server stateless | Replace in-memory dict store with file-based KG reference; accept a `kg_path` parameter |
| Mark clustering as post-extraction analysis | Move `clustering/` import to opt-in, same as `neo4j` |

---

## 7. Proposed Target Architecture

```
 ┌─────────────────────────────────────────┐
 │             User Input                  │
 │  (raw text OR pre-defined schema)       │
 └────────────────────┬────────────────────┘
                      │
              ┌───────▼────────┐
              │  drg.schema    │  ← DRGSchema / EnhancedDRGSchema
              │  (declarative) │    (the "D" in DRG)
              └───────┬────────┘
                      │
         ┌────────────▼────────────┐
         │  drg.extract            │
         │  ┌─────────────────┐    │
         │  │ _schema_gen     │    │  ← Auto-discover schema from text
         │  │ (optional path) │    │    (generate_schema_from_text)
         │  └────────┬────────┘    │
         │           │             │
         │  ┌────────▼────────┐    │
         │  │ chunking        │    │  ← Text → chunks
         │  └────────┬────────┘    │
         │           │             │
         │  ┌────────▼────────┐    │
         │  │ _signatures     │    │  ← Schema → DSPy Signatures
         │  │ KGExtractor     │    │    (entity + relation extraction)
         │  └────────┬────────┘    │
         └───────────┼─────────────┘
                     │
         ┌───────────▼─────────────┐
         │  drg.graph              │
         │  builders → EnhancedKG  │  ← Core output type
         │  provenance, validation │
         └───────────┬─────────────┘
                     │
        ┌────────────┼─────────────┐
        │            │             │
        ▼            ▼             ▼
  entity_     incremental/    evaluation/
  resolution  versioning/     (P/R/F1)
  (dedup)     diff
        │            │
        └────────────▼
             [Optional lifecycle]
                   │
        ┌──────────┼───────────┐
        ▼          ▼           ▼
     query/    reasoning/  temporal/
  (traversal)  (rules)    (metadata)
        │          │
        └──────────▼
         [Integration surfaces]
               │
     ┌─────────┼──────────┐
     ▼         ▼          ▼
   cli.py   api/       mcp_server
             (slim)     (stateless)
```

**What is NOT in the target architecture:**
- No embedding providers in the core package
- No optimizer / prompt tuning
- No hard-coded relation type ontology (`RelationshipType` enum)
- No LLM-backed cluster summarization
- No hub-proxy node mutations
- No legacy MCP JSON-RPC shim
- No parallel event extraction pipeline (events are schema entity types)

---

## 8. Summary Scores

| Dimension | Current State | Target State |
|---|---|---|
| Mission alignment | ~55% of code directly serves schema → KG extraction | >85% after Phase 1+2 |
| Mandatory deps | 1 (pydantic) — good | 1 — maintain |
| Optional extras | 15 packages across 10 extras | 6 focused extras |
| Hard-coded rules | 3 locations with English-specific/ontology-specific rules | 0 in core |
| LLM call sites outside extraction | 4 (optimizer, cluster summary, relationship classifier, event extraction) | 0 in core |
| Dead files | 6 | 0 |
| Retrieval-oriented scope-creep files | 3 (`community_report`, `clustering/summarization` LLM, `relationship_model/`) | 0 in core |
| API surface stability | Low (Alpha; two parallel MCP modules) | Stable (one MCP module) |

---

> **Bottom line:** The core extraction path (`schema → chunking → DSPy signatures → EnhancedKG → provenance/versioning`) is well-designed and aligned with the DRG mission. The scope creep is localized and clearly bounded: `optimizer/`, `events/`, `embedding/`, `graph/relationship_model/`, and the LLM paths in `clustering/` represent the primary divergence from the stated objective. Removing or quarantining these five areas would reduce total module count by ~30%, eliminate all hardcoded ontology assumptions, and leave a cleaner declarative-extraction story.
