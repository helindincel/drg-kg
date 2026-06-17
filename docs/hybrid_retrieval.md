# Hybrid Retrieval

Hybrid retrieval combines DRG's graph-first query layer with optional semantic
document-chunk retrieval.

Status: opt-in, pure stdlib by default. No vector database is required.

## Current Retrieval Baseline

Verified source components:

- `drg.query.GraphQuery` exposes graph retrieval over `EnhancedKG`.
- `InMemoryBackend` indexes nodes, edges, relation types, and clusters.
- Graph retrieval already supports entity lookup, relationship filtering,
  evidence bundles, neighborhoods, paths, explanations, events, temporal
  queries, communities, and deterministic free-text search.
- `drg.chunking.strategies` creates chunks with `chunk_id`, `chunk_text`,
  origin metadata, token counts, and boundary metadata.
- `drg.graph.builders.build_enhanced_kg` stores edge evidence snippets and
  `source_ref` metadata.
- `drg.embedding.providers` defines embedding providers, and
  `EnhancedKG.add_entity_embeddings()` can attach embeddings to graph nodes.

Missing before hybrid retrieval:

- No vector-store abstraction.
- No document chunk index attached to query-time retrieval.
- No merged graph/vector ranking.
- No result type that explains both graph structures and chunks.

## Architecture

```
EnhancedKG ──► InMemoryBackend ──► graph seeds, edges, events, paths
                                  │
                                  ▼
                           hybrid_search()
                                  ▲
                                  │
Document chunks ──► VectorStore protocol ──► vector chunk hits
```

`VectorStore` is a small protocol:

```python
class VectorStore:
    def search(self, query_embedding: list[float], *, limit: int = 10):
        ...
```

The built-in `InMemoryVectorStore` is intended for tests, demos, and small
local projects. Chroma, Qdrant, Weaviate, and Neo4j Vector can implement the
same protocol without changing `GraphQuery`.

## Usage

```python
from drg.query import GraphQuery, InMemoryVectorStore, VectorDocumentChunk

chunks = [
    VectorDocumentChunk(
        chunk_id="doc1_chunk_000",
        document_id="doc1",
        text="Microsoft acquired GitHub in a major developer-platform deal.",
        embedding=[1.0, 0.0, 0.0],
    )
]

store = InMemoryVectorStore.from_chunks(chunks)
gq = GraphQuery(kg, vector_store=store, embedding_provider=embedding_provider)

results = gq.hybrid_search("What acquisitions involve Microsoft?")
for result in results:
    print(result.entity.id, result.score)
    print(result.explanation.to_dict())
```

If no vector store or embedding provider is supplied, `hybrid_search()` falls
back to graph-only retrieval while returning the same result schema.

## Ranking Methodology

Each result has visible `ranking_components`:

| Component | Meaning |
|---|---|
| `graph_relevance` | Entity-name match, relation/detail token overlap, and graph expansion relevance. |
| `vector_similarity` | Best semantic chunk score for chunks mentioning the entity. |
| `confidence` | Average/max available node and edge confidence scores. |
| `evidence_quality` | Evidence item count, source-document count, and supporting chunk count. |
| `hop_closeness` | Multi-hop penalty: direct matches score highest; farther entities decay. |

Default weights:

```python
HybridRankingWeights(
    graph_relevance=0.35,
    vector_similarity=0.25,
    confidence=0.15,
    evidence_quality=0.15,
    hop_closeness=0.10,
)
```

The final score is a weighted sum clamped to `[0, 1]`. Override weights per
query when a workflow should emphasize graph structure or semantic similarity.

## Explainability

Every `HybridSearchResult` includes:

- `entity`: the retrieved graph entity.
- `graph_edges`: relationships that contributed.
- `events`: incident event nodes.
- `document_chunks`: semantic chunk hits.
- `explanation.why`: retrieval reasons such as `entity_name_match`,
  `direct_graph_neighbor`, `multi_hop_graph_expansion`, and
  `semantic_chunk_match`.
- `explanation.graph_structures`: exact graph structures and chunk-entity
  links that contributed.
- `explanation.evidence`: edge-level evidence with snippets, confidence, and
  source references.
- `explanation.ranking_components`: transparent score breakdown.

## Storage Choice

The simplest compatible architecture is an in-memory vector-store protocol:

- It adds no required dependency.
- It does not couple DRG to one vector database.
- It preserves all existing graph workflows.
- It can be replaced by Chroma, Qdrant, Weaviate, or Neo4j Vector adapters
  later by implementing `VectorStore.search()`.
