# Dataset-Agnostic Semantic Pipeline: Genel Bakış

## 1. Mimari Prensipler

### 1.1 Dataset-Agnostic Tasarım

Pipeline, herhangi bir veri kaynağından bağımsız olarak çalışacak şekilde tasarlanmıştır. Bu agnostik yaklaşım şu prensiplere dayanır:

- **Abstraction Layers**: Veri kaynağı, chunking stratejisi ve embedding modeli arasında net arayüzler
- **Pluggable Components**: Her bileşen bağımsız olarak değiştirilebilir ve test edilebilir
- **Metadata Preservation**: Her chunk, orijin veri kaynağı ve işlem geçmişi hakkında zengin metadata taşır
- **Domain Adaptation**: Domain-specific optimizasyonlar, core pipeline'ı değiştirmeden eklenebilir

### 1.2 Monolithic-Modular Mimarisi

Sistem, monolitik bir yapı içinde modüler bileşenlerden oluşur:

- **Monolithic**: Tüm bileşenler aynı codebase içinde, tek bir deployment unit
- **Modular**: Her bileşen bağımsız interface'ler üzerinden iletişim kurar
- **Loose Coupling**: Bileşenler arası bağımlılıklar minimal ve açıkça tanımlıdır
- **High Cohesion**: İlgili fonksiyonellik aynı modülde gruplanır

## 2. Pipeline Akış Diyagramı (Kavramsal)

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA INGESTION LAYER                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Text    │  │  PDF     │  │  Markdown│  │  JSON    │      │
│  │  Files   │  │  Docs    │  │  Files   │  │  Streams │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │             │              │             │             │
│       └─────────────┴──────────────┴─────────────┘             │
│                          │                                      │
│                    [Normalizer]                                 │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CHUNKING & SEGMENTATION LAYER                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Token-Based Chunker (512-1024 token windows)        │    │
│  │  - Overlap Strategy: 10-20% sliding window           │    │
│  │  - Boundary Detection: Sentence/paragraph aware      │    │
│  │  - Metadata Injection: chunk_id, sequence_idx, origin │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                      │
│                    [Chunk Validator]                           │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC ENRICHMENT LAYER                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Semantic Tagging                                    │    │
│  │  - Topic Classification                              │    │
│  │  - Entity Recognition (NER)                          │
│  │  - Intent Detection                                  │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Embedding Abstraction Layer                         │    │
│  │  - OpenAI Embeddings (text-embedding-3-small/large)  │    │
│  │  - Gemini Embeddings (embedding-001)                 │    │
│  │  - OpenRouter (unified API)                          │    │
│  │  - Local Models (optional)                           │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH LAYER                        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Entity & Relation Extraction (DRG)                  │    │
│  │  - Schema-based extraction                           │    │
│  │  - Graph construction                                │    │
│  │  - Node/Edge metadata                               │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Graph Database (e.g., Neo4j, NetworkX in-memory)   │    │
│  │  - Node embeddings (optional)                        │    │
│  │  - Edge weights                                      │    │
│  │  - Graph algorithms                                  │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLUSTERING & SUMMARIZATION LAYER             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Graph Clustering                                    │    │
│  │  - Louvain Algorithm                                 │    │
│  │  - Leiden Algorithm                                  │    │
│  │  - Spectral Clustering                               │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Cluster Summarization                               │    │
│  │  - Per-cluster summary generation                    │    │
│  │  - Community report generation                       │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                      │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Bileşen Sorumlulukları

### 3.1 Ingestion Layer

**Sorumluluklar:**
- Çoklu format desteği (text, PDF, Markdown, JSON, etc.)
- Format normalization (tüm formatlar → unified text representation)
- Encoding handling (UTF-8, Latin-1, etc.)
- Metadata extraction (dosya adı, tarih, kaynak bilgisi)

**Tasarım Kararları:**
- Format-specific parser'lar pluggable olmalı
- Normalization pipeline'ı reversible olmalı (debugging için)
- Metadata schema'sı genişletilebilir olmalı

### 3.2 Chunking Layer

**Sorumluluklar:**
- Token-based segmentation (512-1024 token windows)
- Overlap stratejisi uygulama
- Boundary detection (sentence/paragraph aware)
- Chunk metadata injection

**Tasarım Kararları:**
- Tokenizer abstraction (farklı tokenizer'lar desteklenmeli)
- Overlap stratejisi konfigüre edilebilir olmalı
- Chunk ID generation: deterministic ve unique
- Sequence index: orijinal doküman içindeki sıralama

### 3.3 Semantic Enrichment Layer

**Sorumluluklar:**
- Semantic tagging (topic, entity, intent)
- Embedding generation (abstraction layer üzerinden)
- Metadata enrichment

**Tasarım Kararları:**
- Embedding provider abstraction: OpenAI, Gemini, OpenRouter, Local
- Semantic tagging opsiyonel olmalı (cost/performance trade-off)
- Tagging model'i chunking'den bağımsız olmalı

### 3.4 (Kapsam Dışı) Vector Store Katmanı

Bu repo, “serving/arama” katmanı hedeflemediği için vektör tabanlı indeks/benzerlik bileşeni **kapsam dışına alınmıştır** ve koddan çıkarılmıştır.
Benzerlik temelli yardımcı sinyaller (ör. entity merge destek sinyali) gerekiyorsa, bu **embedding provider** üzerinden opsiyonel olarak eklenebilir.

### 3.5 Knowledge Graph Layer

**Sorumluluklar:**
- Entity extraction (DRG schema-based)
- Relation extraction
- Graph construction
- Graph storage

**Tasarım Kararları:**
- DRG schema declarative olmalı
- Graph database abstraction (Neo4j, NetworkX, etc.)
- Node/edge metadata preservation

### 3.6 Sorgu & Analiz Yardımcıları

**Sorumluluklar:**
- Knowledge graph üzerinde sorgulama ve analiz yardımcıları (graph traversal, komşuluk genişletme)
- Community report üretimi ve özetleme
- Export/visualization için veri hazırlama

**Tasarım Kararları:**
- Bu proje bir “serving/arama framework” değildir; odak **KG extraction + graph analiz/çıktı**dır.
- KG odaklı analiz/sorgu yardımcıları modüler olmalı.

### 3.7 Clustering & Summarization Layer

**Sorumluluklar:**
- Graph clustering (Louvain, Leiden, Spectral)
- Cluster summarization
- Community report generation

**Tasarım Kararları:**
- Clustering algorithm pluggable
- Summarization strategy per-cluster
- Report format extensible

## 4. Veri Akışı ve Metadata Preservation

### 4.1 Metadata Schema

Her chunk aşağıdaki metadata'yı taşır:

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

Her graph node aşağıdaki metadata'yı taşır:

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

Her graph edge aşağıdaki metadata'yı taşır:

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

## 5. Tasarım Trade-off'ları

### 5.1 Chunking Trade-offs

**Token Window Size:**
- **512 tokens**: Daha fazla chunk, daha ince-granular bağlam, daha yüksek maliyet
- **1024 tokens**: Daha az chunk, daha geniş context, daha düşük storage cost
- **Karar**: 512-1024 arası konfigüre edilebilir, varsayılan 768

**Overlap Strategy:**
- **10% overlap**: Daha az redundancy, daha düşük cost, entity boundary'lerde kayıp riski
- **20% overlap**: Daha fazla redundancy, daha yüksek cost, daha iyi entity preservation
- **Karar**: 15% varsayılan, domain-specific tuning için konfigüre edilebilir

### 5.2 Embedding Trade-offs

**Model Selection:**
- **OpenAI text-embedding-3-small**: Hızlı, ucuz, 1536 dimensions
- **OpenAI text-embedding-3-large**: Daha yavaş, daha pahalı, 3072 dimensions, daha iyi quality
- **Gemini embedding-001**: Alternatif provider, farklı semantic space
- **OpenRouter**: Unified API, multiple model support

**Karar Kriterleri:**
- **Cost**: Token-based pricing, batch processing optimizasyonu
- **Latency**: Real-time vs batch use case'ler
- **Portability**: Model lock-in riski
- **Semantic Consistency**: Cross-domain performance

### 5.3 Sorgulama/Analiz Trade-off'ları

**KG Query & Analysis:**
- **Graph traversal**: İlişkisel sorular için güçlü; graph kalitesine bağlı
- **Community reports**: Büyük graph'larda yorumlanabilirlik sağlar; üretim maliyetine bağlı

**Karar Kriterleri:**
- **Graph Quality**: Extraction kalitesi düşükse traversal sonuçları da zayıflar
- **Latency Requirements**: Online vs batch analiz

## 6. Genişletilebilirlik ve Extension Points

### 6.1 Pluggable Components

- **Chunking Strategy**: Token-based, sentence-based, paragraph-based, semantic-based
- **Embedding Provider**: OpenAI, Gemini, OpenRouter, Local models
- **Graph Database**: Neo4j, NetworkX, ArangoDB
- **Clustering Algorithm**: Louvain, Leiden, Spectral, Custom

### 6.2 Domain Adaptation

Domain-specific optimizasyonlar core pipeline'ı değiştirmeden eklenebilir:

- **Domain-specific chunking**: Technical docs için code-aware chunking
- **Domain-specific tagging**: Medical domain için ICD-10 tagging
- **Domain-specific schemas**: DRG schema'ları domain'e özel

## 7. Değerlendirme Metodolojisi

### 7.1 Pipeline Metrics

- **Chunking Quality**: Entity boundary preservation, semantic coherence
- **Embedding Quality**: Semantic similarity accuracy, cross-domain consistency
- **Graph Quality**: Entity extraction F1, relation extraction F1, graph completeness, duplicate entity oranı

### 7.2 Multi-Dataset Evaluation

3-4 heterojen dataset üzerinde değerlendirme:
- Long narrative text (20-page story)
- Factual text (Wikipedia biography)
- Technical/structured document
- Informal dialogue (chat/forum)

Her dataset için:
- Chunking quality analysis
- KG extraction kalite metrikleri (entity/relation F1, duplicate oranı, cross-chunk edge retention)
- Entity extraction effectiveness
- Failure cases ve edge behaviors

### 7.3 Comparison Framework

Bu repo için karşılaştırma ekseni “serving/arama framework’leri” değil, pipeline bileşenlerinin kalite/sağlamlık etkisidir:

- Chunking stratejilerinin extraction kalitesine etkisi
- Schema sampling stratejilerinin kapsama etkisi
- Coreference/entity resolution post-process etkisi
- Cross-chunk context injection (deterministik snippet) etkisi

