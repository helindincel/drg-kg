# Graf Kümeleme ve Özetleme: Topluluk Raporu Altyapısı

## 1. Genel Bakış

Graph clustering ve summarization, knowledge graph'ı anlamlı community'lere böler ve her community için özet üretir. Bu, büyük graph'ları interpretable ve scalable hale getirir.

## 2. Clustering Pipeline

### 2.1 Pipeline Flow

```
Chunks → KG Nodes → Graph Construction → Clustering → Cluster Summarization → Community Report
```

**Adımlar:**

1. **Chunks → KG Nodes**: Entity extraction, graph node'ları oluştur
2. **Graph Construction**: Node'lar arası edge'leri oluştur
3. **Clustering**: Graph'ı community'lere böl
4. **Cluster Summarization**: Her cluster için özet üret
5. **Community Report**: Human-readable report oluştur

### 2.2 Graph Construction for Clustering

**Node Representation:**

- **Entity Nodes**: Extracted entity'ler
- **Chunk Nodes** (optional): Chunk'ları da node olarak ekle
- **Not**: Chunk node yaklaşımı opsiyoneldir; “serving/arama katmanı” anlamına gelmez.

**Edge Representation:**

- **Direct Relations**: Extracted relations (produces, owns, etc.)
- **Co-occurrence**: Aynı chunk'ta geçen entity'ler
- **Semantic Similarity**: Entity embedding similarity
- **Weighted**: Edge weight = relation confidence + co-occurrence frequency

**Graph Properties:**

- **Undirected**: Clustering için undirected graph
- **Weighted**: Edge weights clustering'de kullanılır
- **Attributed**: Node attributes (embedding, metadata)

## 3. Clustering Algorithms

### 3.1 Louvain Algorithm

**Algoritma:**

1. Her node kendi community'sinde başlar
2. Her node için, komşu community'lere taşıma yap
3. Modularity artışı varsa, node'u taşı
4. Community'leri node olarak collapse et
5. Adım 2-4'ü tekrarla (hierarchical)

**Modularity:**

```
Q = (1/2m) * Σ[Aij - (ki*kj/2m)] * δ(ci, cj)

Aij: Edge weight between i and j
ki: Degree of node i
m: Total edge weight
ci: Community of node i
δ: Kronecker delta (1 if same community, 0 otherwise)
```

**Avantajlar:**
- Hızlı (O(n log n))
- Hierarchical structure
- Scalable (large graphs)
- Deterministic (with fixed random seed)

**Dezavantajlar:**
- Local optima riski
- Resolution limit (small communities merge)
- Parameter tuning gerekli (resolution parameter)

**Kullanım Senaryoları:**
- Large-scale graphs (> 10K nodes)
- Fast clustering requirement
- Hierarchical structure desired

### 3.2 Leiden Algorithm

**Algoritma:**

Louvain'in geliştirilmiş versiyonu:

1. Louvain gibi başla
2. Her iteration'da, community'lerin connected olup olmadığını kontrol et
3. Disconnected community'leri ayır
4. Refinement phase: Node'ları local move ile optimize et

**Avantajlar:**
- Louvain'den daha iyi quality
- Connected communities garantisi
- Hala hızlı (O(n log n))
- Better modularity scores

**Dezavantajlar:**
- Louvain'den biraz daha yavaş
- Hala resolution limit var
- Parameter tuning gerekli

**Kullanım Senaryoları:**
- Quality-critical applications
- Connected communities required
- Medium to large graphs (5K-50K nodes)

### 3.3 Spectral Clustering

**Algoritma:**

1. Graph Laplacian matrix'ini hesapla
2. Eigenvalue decomposition yap
3. Top-K eigenvectors'ı al
4. K-means clustering uygula

**Graph Laplacian:**

```
L = D - A

D: Degree matrix (diagonal)
A: Adjacency matrix
```

**Normalized Laplacian:**

```
L_norm = D^(-1/2) * L * D^(-1/2)
```

**Avantajlar:**
- Global optimum'a yakın
- Theoretical guarantees
- Good for well-separated communities

**Dezavantajlar:**
- Yavaş (O(n³) eigenvalue decomposition)
- K (number of clusters) önceden belirlenmeli
- Memory intensive (large matrices)

**Kullanım Senaryoları:**
- Small to medium graphs (< 5K nodes)
- Well-separated communities
- Quality-critical applications

### 3.4 Algorithm Comparison

| Algorithm | Speed | Quality | Scalability | Parameters |
|-----------|-------|---------|-------------|------------|
| Louvain | Very Fast | Good | Excellent | Resolution |
| Leiden | Fast | Very Good | Excellent | Resolution |
| Spectral | Slow | Excellent | Limited | K (clusters) |

**Öneri:**
- **Large graphs** (> 10K nodes): Louvain veya Leiden
- **Medium graphs** (1K-10K nodes): Leiden
- **Small graphs** (< 1K nodes): Spectral
- **Quality-critical**: Leiden veya Spectral
- **Speed-critical**: Louvain

## 4. Cluster Summarization

### 4.1 Summarization Strategy

**Per-Cluster Summary:**

Her cluster için:
1. Cluster node'larını topla
2. Cluster edge'lerini topla
3. Cluster metadata'sını topla
4. Summary generation (LLM-based veya template-based)

**Summary Components:**

- **Cluster Name**: Cluster'ı özetleyen isim
- **Key Entities**: Cluster'daki önemli entity'ler
- **Key Relations**: Cluster'daki önemli relationship'ler
- **Cluster Description**: Cluster'ın semantic açıklaması
- **Cluster Statistics**: Node count, edge count, density

### 4.2 LLM-Based Summarization

**Prompt Template:**

```
Given the following knowledge graph cluster:

Nodes: [entity_1, entity_2, ...]
Relations: [entity_1 → relation → entity_2, ...]
Source Chunks: [chunk_id_1, chunk_id_2, ...]

Generate a summary:
1. Cluster name (1-3 words)
2. Key themes (3-5 themes)
3. Main relationships (3-5 relationships)
4. Cluster description (2-3 sentences)
```

**Avantajlar:**
- Natural language summaries
- Context-aware
- Flexible

**Dezavantajlar:**
- LLM cost
- Latency
- Non-deterministic

### 4.3 Template-Based Summarization

**Template:**

```
Cluster: {cluster_name}
Size: {node_count} nodes, {edge_count} edges
Density: {density}

Key Entities:
- {top_entities}

Key Relations:
- {top_relations}

Description:
{statistical_description}
```

**Avantajlar:**
- Deterministic
- Fast
- Cost-free

**Dezavantajlar:**
- Less natural
- Limited flexibility

### 4.4 Çok Aşamalı Özetleme (Multi-Stage)

**Strategy:**

1. Template-based summary oluştur (fast, deterministic)
2. LLM-based refinement (optional, quality-critical clusters)
3. Human review (optional, critical clusters)

## 5. Community Report Generation

### 5.1 Report Structure

**Executive Summary:**
- Total clusters
- Graph statistics (nodes, edges, density)
- Key insights

**Cluster Details:**
- Her cluster için:
  - Cluster name
  - Cluster summary
  - Key entities
  - Key relations
  - Cluster statistics
  - Source chunks (optional)

**Visualizations:**
- Cluster network graph
- Cluster size distribution
- Cluster density heatmap

### 5.2 Report Formats

**Markdown:**
- Human-readable
- Version control friendly
- Easy to share

**JSON:**
- Machine-readable
- API-friendly
- Structured data

**HTML:**
- Interactive visualizations
- Web-friendly
- Rich formatting

**PDF:**
- Professional reports
- Print-friendly
- Static format

### 5.3 Interpretability Features

**Cluster Hierarchy:**
- Hierarchical clustering results (Louvain/Leiden)
- Parent-child relationships
- Multi-level summaries

**Cluster Overlap:**
- Overlapping clusters (soft clustering)
- Shared entities
- Boundary analysis

**Cluster Evolution:**
- Temporal clustering (if time-series data)
- Cluster merge/split events
- Trend analysis

## 6. Scalability Considerations

### 6.1 Large-Scale Clustering

**Challenges:**
- Memory limits (graph size)
- Computation time (clustering algorithm)
- Storage (cluster assignments)

**Solutions:**
- **Sampling**: Large graph'ı sample et, cluster et
- **Distributed Clustering**: Parallel clustering (multiple machines)
- **Incremental Clustering**: New nodes'ları mevcut cluster'lara assign et

### 6.2 Incremental Updates

**Strategy:**

1. Initial clustering yap
2. New chunks geldiğinde:
   - New entities extract et
   - Graph'a ekle
   - New entities'leri mevcut cluster'lara assign et (similarity-based)
   - Eğer similarity düşükse, yeni cluster oluştur
3. Periodic re-clustering (e.g., weekly)

**Assignment Algorithm:**

```
For new entity:
  1. Embed entity
  2. Calculate similarity to cluster centroids
  3. If max_similarity > threshold:
       Assign to cluster
  4. Else:
       Create new cluster
```

### 6.3 Storage Optimization

**Cluster Storage:**

- **Cluster Assignments**: node_id → cluster_id mapping
- **Cluster Metadata**: cluster_id → summary, statistics
- **Compression**: Cluster assignments'ı compress et

**Query Optimization:**

- **Cluster Index**: cluster_id → node_ids mapping
- **Entity Index**: entity_id → cluster_id mapping
- **Fast Lookup**: Hash indexes

## 7. Evaluation Methodology

### 7.1 Clustering Quality Metrics

**Modularity:**
- Higher is better
- Target: > 0.3 (good), > 0.5 (excellent)

**Silhouette Score:**
- Measures cluster cohesion and separation
- Range: -1 to 1
- Target: > 0.5

**Conductance:**
- Measures cluster boundary quality
- Lower is better (tighter clusters)
- Target: < 0.5

### 7.2 Summarization Quality Metrics

**Coverage:**
- Summary'deki entity'lerin cluster'daki entity'lere oranı
- Target: > 0.8

**Relevance:**
- Summary'nin cluster'a semantic relevance'i
- Human evaluation (0-5 scale)
- Target: > 4.0

**Completeness:**
- Summary'nin cluster'ı ne kadar kapsadığı
- Human evaluation (0-5 scale)
- Target: > 3.5

### 7.3 Report Quality Metrics

**Readability:**
- Report'u okuma kolaylığı
- Human evaluation (0-5 scale)
- Target: > 4.0

**Usefulness:**
- Report'un kullanışlılığı
- Human evaluation (0-5 scale)
- Target: > 4.0

## 8. Implementation Considerations

### 8.1 Library Selection

**Clustering:**
- **python-louvain**: Louvain algorithm
- **leidenalg**: Leiden algorithm
- **scikit-learn**: Spectral clustering
- **igraph**: Graph algorithms

**Graph Processing:**
- **NetworkX**: Python graph library
- **igraph**: Fast graph library
- **Neo4j**: Graph database (for large graphs)

### 8.2 Performance Optimization

**Parallel Processing:**
- Multiple clusters'ı parallel summarize et
- Batch LLM calls
- Parallel graph operations

**Caching:**
- Cluster assignments cache
- Summary cache (same cluster → same summary)
- Graph structure cache

**Early Stopping:**
- Convergence criteria (Louvain/Leiden)
- Maximum iterations
- Quality threshold

## 9. Trade-off Özeti

| Aspect | Louvain | Leiden | Spectral |
|--------|---------|--------|----------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Quality | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Scalability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Interpretability | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**Öneri:** Production için Leiden, research için Spectral, large-scale için Louvain.

## 10. Future Directions

### 10.1 Learned Clustering

- Graph neural networks ile clustering
- Supervised clustering (labeled data)
- Adaptive clustering (query-driven)

### 10.2 Multi-Resolution Clustering

- Hierarchical clustering at multiple resolutions
- User-selectable granularity
- Dynamic resolution adjustment

### 10.3 Interactive Clustering

- User feedback ile cluster refinement
- Interactive cluster merging/splitting
- Real-time cluster visualization

