# Chunking Stratejisi: Dataset-Agnostic Metin Segmentasyonu

## 1. Genel Bakış

Chunking stratejisi, dataset-agnostic semantic pipeline'ın temel bileşenidir. Amacı, herhangi bir metin kaynağını **knowledge graph extraction** için güvenli ve tekrarlanabilir şekilde parçalara bölmektir.

## 2. Token Window Stratejisi

### 2.1 Token Window Aralığı: 512-1024

**Gerekçe:**

- **512 tokens**: Daha küçük bağlam penceresi, ince-granular ilişki/varlık çıkarımı için uygun
- **1024 tokens**: Maksimum context window, daha geniş semantic context sağlar
- **768 tokens**: Varsayılan değer, 512 ve 1024 arasında dengeli bir seçim

**Tokenizer Seçimi:**

- **Tiktoken (OpenAI)**: GPT modelleri için optimize, hızlı
- **SentencePiece**: Çoklu dil desteği, BPE tabanlı
- **HuggingFace Tokenizers**: Model-specific tokenization

**Abstraction Layer:**

Tokenizer seçimi, embedding model'ine bağlı olmalıdır. Örneğin:
- OpenAI embeddings → Tiktoken (cl100k_base)
- Gemini embeddings → SentencePiece
- Local models → Model-specific tokenizer

### 2.2 Dynamic Window Sizing

**Adaptive Chunking:**

- **Short documents** (< 512 tokens): Single chunk, overlap yok
- **Medium documents** (512-2048 tokens): Fixed window (768 tokens), 15% overlap
- **Long documents** (> 2048 tokens): Fixed window (768 tokens), 15% overlap, sequence indexing

**Domain-Specific Adjustments:**

- **Technical docs**: Code blocks'ları korumak için daha büyük window (1024)
- **Narrative text**: Paragraph boundary'lerini korumak için daha küçük window (512)
- **Dialogue**: Conversation turn'leri korumak için özel strateji

## 3. Overlap Stratejisi

### 3.1 Overlap Oranı: 10-20%

**Gerekçe:**

- **10% overlap**: Minimal redundancy, entity boundary'lerde kayıp riski
- **20% overlap**: Maksimum redundancy, entity preservation garantisi
- **15% overlap**: Varsayılan, dengeli yaklaşım

**Overlap Hesaplama:**

```
overlap_tokens = chunk_size * overlap_ratio
next_chunk_start = current_chunk_end - overlap_tokens
```

**Örnek (768 token window, 15% overlap):**

```
Chunk 1: tokens[0:768]
Chunk 2: tokens[654:1422]  (768 - 115 = 653, 768 + 115 = 883, ama boundary-aware)
Chunk 3: tokens[1308:2076]
```

### 3.2 Boundary-Aware Overlap

**Sentence Boundary Preservation:**

Overlap, sentence boundary'lerini korumalıdır:

```
Chunk 1: "... sentence A. Sentence B. Sentence C ..."
Chunk 2: "Sentence C ..." (overlap sentence C'den başlar, önceki chunk'ın son cümlesi)
```

**Paragraph Boundary Preservation:**

Paragraph boundary'leri korunmalıdır:

```
Chunk 1: "... paragraph 1 ... paragraph 2 ..."
Chunk 2: "paragraph 2 ..." (overlap paragraph 2'den başlar)
```

**Implementation Strategy:**

1. Token-based window hesapla
2. Sentence/paragraph boundary'leri tespit et
3. Overlap'i en yakın boundary'ye hizala
4. Chunk'ı boundary'den başlat/bitir

### 3.3 Overlap Trade-offs

**Avantajlar:**
- Entity boundary preservation
- Context continuity
- Multi-chunk entity extraction

**Dezavantajlar:**
- Storage cost artışı (15% overlap = ~15% daha fazla chunk)
- Embedding cost artışı
- Retrieval redundancy (aynı bilgi birden fazla chunk'ta)

**Optimizasyon:**

- **Deduplication**: Retrieval sırasında aynı chunk'ları filtrele
- **Smart Overlap**: Sadece entity-rich bölgelerde overlap kullan
- **Compressed Storage**: Overlap bölgelerini ayrı store et

## 4. Chunk Metadata Schema

### 4.1 Zorunlu Metadata

```json
{
  "chunk_id": "dataset_name_doc_001_chunk_000",
  "sequence_index": 0,
  "origin_dataset": "dataset_name",
  "origin_file": "path/to/source/file.txt",
  "token_count": 768,
  "char_count": 3072,
  "chunk_text": "..."
}
```

**chunk_id Format:**

`{dataset}_{doc_id}_{chunk_index}`

- Deterministic: Aynı input → aynı chunk_id
- Unique: Global uniqueness garantisi
- Human-readable: Debugging için parse edilebilir

**sequence_index:**

- 0-based index
- Orijinal doküman içindeki sıralama
- Retrieval sırasında context ordering için kullanılır

### 4.2 Opsiyonel Metadata

```json
{
  "semantic_tags": {
    "topic": ["technology", "AI"],
    "entities": ["Apple", "iPhone"],
    "intent": "informational"
  },
  "embedding_model": "openai/text-embedding-3-small",
  "embedding_dimension": 1536,
  "extraction_timestamp": "2025-01-XXT00:00:00Z",
  "chunking_strategy": "token_based_768_15pct",
  "boundary_info": {
    "starts_at_sentence": true,
    "ends_at_sentence": true,
    "starts_at_paragraph": false,
    "ends_at_paragraph": false
  }
}
```

**semantic_tags:**

- Topic classification: Chunk'ın ana konusu
- Entity list: Chunk içindeki önemli entity'ler
- Intent: Informational, instructional, narrative, etc.

**boundary_info:**

- Sentence/paragraph boundary bilgisi
- Chunking quality assessment için kullanılır

## 5. Chunking Algoritmaları

### 5.1 Token-Based Chunking (Varsayılan)

**Algoritma:**

1. Text'i tokenize et
2. Token window size'ı belirle (512-1024)
3. Overlap ratio'yu uygula (10-20%)
4. Sentence/paragraph boundary'leri tespit et
5. Chunk'ları boundary-aware oluştur
6. Metadata inject et

**Avantajlar:**
- Model token limit'lerine uyumlu
- Deterministic
- Hızlı

**Dezavantajlar:**
- Semantic coherence garantisi yok
- Entity boundary'lerde kesilme riski

### 5.2 Sentence-Based Chunking

**Algoritma:**

1. Text'i sentence'lara böl
2. Sentence'ları token count'a göre grupla
3. Her grup bir chunk
4. Overlap: son N sentence'ı bir sonraki chunk'a ekle

**Avantajlar:**
- Semantic coherence (sentence boundary'leri korunur)
- Entity boundary preservation

**Dezavantajlar:**
- Token count kontrolü zor
- Variable chunk size

### 5.3 Semantic Chunking (Gelişmiş)

**Algoritma:**

1. Text'i embedding'lerle segment et
2. Semantic similarity'ye göre chunk boundary'leri belirle
3. Token window constraint'lerini uygula
4. Overlap stratejisini uygula

**Avantajlar:**
- En yüksek semantic coherence
- Topic-aware chunking

**Dezavantajlar:**
- Yavaş (embedding gerektirir)
- Cost yüksek
- Deterministic değil (embedding model değişirse chunk'lar değişir)

## 6. Domain-Specific Adaptations

### 6.1 Long Narrative Text (20-page story)

**Challenges:**
- Temporal continuity
- Character consistency
- Plot progression

**Strategy:**
- 1024 token window (daha geniş context)
- 20% overlap (character/plot continuity)
- Chapter/section boundary awareness
- Character entity tracking across chunks

### 6.2 Factual Text (Wikipedia biography)

**Challenges:**
- Factual accuracy
- Entity consistency
- Chronological ordering

**Strategy:**
- 768 token window (balanced)
- 15% overlap
- Section boundary awareness
- Entity-centric chunking (entity'ler chunk boundary'lerinde kesilmemeli)

### 6.3 Technical/Structured Document

**Challenges:**
- Code block preservation
- Table structure
- Cross-references

**Strategy:**
- 1024 token window (code blocks için)
- Code block'ları atomic chunk olarak koru
- Table'ları atomic chunk olarak koru
- 10% overlap (structured content'te daha az gerekli)

### 6.4 Informal Dialogue (Chat/Forum)

**Challenges:**
- Conversation turn preservation
- Context switching
- Multi-party dialogue

**Strategy:**
- 512 token window (daha granular)
- Conversation turn boundary awareness
- Speaker attribution preservation
- 15% overlap (context continuity için)

## 7. Chunking Quality Metrics

### 7.1 Entity Boundary Preservation

**Metric:**
- Entity'lerin chunk boundary'lerinde kesilme oranı
- Target: < 5% entity boundary violation

**Measurement:**
1. Ground truth entity'leri extract et
2. Chunk boundary'lerini kontrol et
3. Entity kesilme oranını hesapla

### 7.2 Semantic Coherence

**Metric:**
- Chunk içi semantic similarity (intra-chunk)
- Chunk'lar arası semantic similarity (inter-chunk)
- Target: Intra-chunk > inter-chunk

**Measurement:**
1. Chunk'ları embed et
2. Intra-chunk similarity hesapla (chunk içi sentence'lar)
3. Inter-chunk similarity hesapla (komşu chunk'lar)
4. Ratio hesapla

### 7.3 Token Distribution

**Metric:**
- Chunk size distribution
- Target: Mean ≈ target window size, Std < 20% of mean

**Measurement:**
- Token count histogram
- Statistical analysis (mean, std, min, max)

## 8. Implementation Considerations

### 8.1 Performance

- **Batch Processing**: Büyük dataset'ler için batch chunking
- **Parallelization**: Multi-threading/multi-processing
- **Caching**: Tokenization sonuçlarını cache'le

### 8.2 Storage

- **Compression**: Chunk text'lerini compress et (gzip, etc.)
- **Deduplication**: Overlap bölgelerini ayrı store et
- **Indexing**: Chunk ID → metadata mapping için index

### 8.3 Error Handling

- **Encoding Errors**: UTF-8 fallback, error logging
- **Tokenization Errors**: Fallback tokenizer, error recovery
- **Boundary Detection Errors**: Default to token-based, warning log

## 9. Trade-off Özeti

| Strateji | Token Window | Overlap | Avantajlar | Dezavantajlar |
|----------|--------------|---------|------------|---------------|
| Conservative | 512 | 10% | Fine-grained, low cost | Entity boundary risk |
| Balanced | 768 | 15% | Optimal trade-off | - |
| Aggressive | 1024 | 20% | Wide context, entity preservation | High cost, redundancy |

**Öneri:** Balanced strateji (768 tokens, 15% overlap) varsayılan olarak kullanılmalı, domain-specific tuning için konfigüre edilebilir olmalıdır.

