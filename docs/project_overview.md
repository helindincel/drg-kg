# DRG (Dynamic Retrieval Graph) — Project Overview

## 1) DRG Nedir?

**DRG (Dynamic Retrieval Graph)**, metinlerden **bilgi grafiği (Knowledge Graph / KG)** üretmek için tasarlanmış, **dataset‑agnostic**, **DSPy tabanlı** ve **declarative** bir semantic pipeline’dır. Projenin ana hedefi:

- Ham metin girdisini alıp
- (Gerekirse) metinden otomatik **EnhancedDRGSchema** üretip
- Chunk‑based extraction ile **entity + relation** çıkarıp
- Bunları **EnhancedKG** olarak inşa edip
- KG üzerinde **clustering** ve **community report** üretmek
- Sonuçları JSON çıktıları olarak export etmek ve UI’da görselleştirmektir.

DRG research‑grade bir codebase olarak, **GraphRAG / RAG experimentation** için “ham malzeme” üretir: yani asıl odak **KG extraction + graph analiz/çıktı**dır.

---

## 2) DRG Ne Değildir? (Net Non‑Goals)

Bu bölüm özellikle önemli: DRG’yi “diğer RAG projeleri” ile karıştırmamak için.

- **Bir RAG framework değildir.**
  - DRG’nin UI query endpoint’i **deterministic KG lookup** yapar.
  - “LLM ile cevap üretme / retrieval‑augmented generation serving” gibi bir hedefi yoktur.
- **Bir serving/search platformu değildir.**
  - Vektör store katmanı (Chroma/Qdrant/Faiss vb.) “serving” amaçlı bir ürün bileşeni olarak bu repo’nun odağında değildir.
  - İhtiyaç olursa sadece deneysel/opsiyonel bileşen olarak entegre edilebilir.
- **Model‑özel (provider‑locked) değildir.**
  - LLM ve embedding provider’ları abstract edilmiştir; environment üzerinden seçilir.

Özet: DRG, **“query’yi cevaplayan uygulama”** olmaktan çok, **“graph çıkaran ve graph’i analiz eden pipeline”**dır.

---

## 3) Neden DSPy? “Declarative Extraction” Ne Demek?

DRG’nin temel farkı, extraction mantığının “prompt‑heavy, ad‑hoc” yerine **declarative** bir formda kurgulanmasıdır.

- **Declarative yaklaşım**: “Nasıl çıkarayım?” yerine “Neyi çıkarayım?” tanımlanır.
  - Kullanıcı/araştırmacı, şemayı (entity types, relation groups, relation’lar, açıklamalar, örnekler) tanımlar.
  - DRG, DSPy üzerinden bu şemadan **dinamik signature’lar** üretir ve extraction programını çalıştırır.
- **DSPy**: LLM çağrılarını “program” gibi ele alıp, structured outputs ve sistematik akış kurmayı kolaylaştırır.
  - DRG, DSPy ile entity/relation extraction akışını modülerleştirir (ör. typed predictor varsa kullan, yoksa degrade et).

Bu sayede DRG:
- Farklı dataset’lerde aynı pipeline’ı çalıştırabilir,
- Şema değiştiğinde extraction davranışı da kontrollü şekilde değişir,
- Araştırma/deney tasarımını daha tekrarlanabilir hale getirir.

---

## 4) Dataset‑Agnostic Tasarım ve “Enhanced Schema”

### 4.1 Dataset‑Agnostic

DRG, belirli bir domain/dataset’e hard‑code edilmez. Bunun için:
- **Abstraction layers** ile ingestion, chunking, embedding, graph inşa, clustering gibi katmanlar ayrıştırılmıştır.
- Chunk’lar ve graph elemanları **zengin metadata** taşır (origin, chunk_id, işlem geçmişi).

### 4.2 EnhancedDRGSchema

DRG’nin tercih ettiği şema formatı **EnhancedDRGSchema**’dır:
- **EntityType**: `name`, `description`, `examples`, `properties`
- **RelationGroup**: ilişkileri semantic olarak gruplayarak organizasyon sağlar
- **Relation**: `name`, `src`, `dst` + açıklayıcı alanlar
  - Bu repo’da relation’lar için **description / detail** gibi alanların taşınması önemsenir (neden bağlandı, bağlam ne?).

Şema iki şekilde kullanılabilir:
- **Manual schema**: Domain bilginizle direkt tanımlarsınız.
- **Auto schema generation**: Metinden otomatik schema üretirsiniz (`generate_schema_from_text()`).

---

## 5) Pipeline (Yüksek Seviye Akış)

DRG’nin “default” kavramsal akışı:

1. **Text Input**
2. **Schema Generation / Load**
3. **Chunking**
4. **KG Extraction (chunk‑based)**
5. **(Opsiyonel) Embeddings**
6. **Clustering**
7. **Community Reports**
8. **Export (JSON) + UI Visualization**

### 5.1 Chunk‑Based Neden?

Uzun metinlerde:
- Entity’ler bir chunk’ta, relation’lar başka chunk’ta geçebilir.
- DRG bu yüzden “chunk‑aware” extraction ve opsiyonel cross‑chunk bağlam enjeksiyonu gibi teknikler kullanır.

### 5.2 KG Inşa Mantığı

Extraction çıktıları:
- `entities`: `(entity_name, entity_type)` listesi
- `triples`: `(source, relation, target)` listesi

Bu çıktılardan:
- `EnhancedKG` inşa edilir (`KGNode`, `KGEdge`, `Cluster`).
- Node/edge metadata korunur; graph üstünde analitik adımlar yapılabilir.

---

## 6) Monolithic‑Modular Mimari

DRG tek bir codebase/deployment unit içinde monolitiktir; ama içeride modülerdir:
- Her katman “kendi sorumluluğu”nu taşır.
- Bileşenler interface’ler üzerinden ayrışır.
- “Loose coupling / high cohesion” hedeflenir.

Bu yaklaşım research‑grade bir kod tabanı için pratik:
- Deneysel bileşenleri eklemek/çıkarmak kolaylaşır,
- Bir bileşendeki değişiklik (ör. chunking) diğer katmanları minimal etkiler.

---

## 7) UI ve Query Davranışı (Önemli: LLM Yok)

DRG UI:
- KG’yi görselleştirir (Cytoscape tabanlı).
- “Load Full Graph” ile tüm grafı çizer.
- “Load Communities” ile cluster renklendirmeli görünümü verir.

### 7.1 UI Query = Deterministic KG Lookup

UI’daki query kutusu:
- entity string matching
- opsiyonel relation filter
- seed entity etrafında neighborhood expansion

yapar. Bu endpoint **RAG/LLM ile cevap üretmez**. Bu, UI’nın hızlı ve deterministik olmasını sağlar.

### 7.2 “Hub” Görselleştirme Notu

Bazı metinler doğal olarak “star‑shape” graph üretir (tek merkezli şirket/karakter etrafında çok ilişki). DRG UI’da bunun için **UI‑only anti‑hub** seçeneği vardır:
- Proxy node’lar ekleyerek layout’un okunabilirliğini artırır
- KG datasını değiştirmez (UI‑only)

---

## 8) Konfigürasyon (Environment‑Driven)

DRG davranışını environment variable’larla yönetir (research + reproducibility için).

Örnek:

- `DRG_MODEL`: LLM modeli (provider prefix ile)
- `DRG_TEMPERATURE`: LLM temperature
- `DRG_MAX_TOKENS`: LLM output budget
- Provider key’leri: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, vb.
- Chunking: `DRG_CHUNK_SIZE`, `DRG_OVERLAP_RATIO`, `DRG_CHUNKING_STRATEGY`
- UI: `DRG_API_PORT`, opsiyonel hub/visualization parametreleri

Bu yaklaşım:
- Kodun içine secret koymayı engeller,
- Deney koşullarını kolayca tekrar etmeyi sağlar.

---

## 9) Proje Yapısı (Repository Structure)

Özet klasör yapısı:

```
DRG/
├── drg/                       # Core library (monolithic codebase)
│   ├── ingestion/             # Ingestion & normalization
│   ├── chunking/              # Chunking strategies + validators
│   ├── embedding/             # Embedding provider abstraction
│   ├── retrieval/             # Retrieval helpers (NOT a serving RAG framework)
│   ├── graph/                 # KG core, schema generation, community reports, visualization adapters
│   ├── clustering/            # Louvain/Leiden/Spectral + summarization
│   ├── optimizer/             # DSPy optimizer experiments
│   ├── api/                   # FastAPI server + UI templates
│   ├── schema.py              # Schema definitions (EnhancedDRGSchema, EntityType, RelationGroup, etc.)
│   ├── extract.py             # DSPy extraction logic
│   └── cli.py                 # CLI
├── docs/                      # Documentation (NO CODE)
├── examples/                   # Example scripts (full pipeline, API server)
├── tests/                      # Unit/integration tests
├── outputs/                    # Generated outputs (KG, schema, reports)
├── pyproject.toml              # Project configuration
└── README.md                   # Entry point for users
```

Not: `docs/` klasörü sadece dokümantasyon içindir; kod taşımamalıdır.

---

## 10) Tipik Kullanım Senaryoları

### 10.1 “Metinden KG üret”
- Metni ver
- Schema’yı üret veya yükle
- Chunking + extraction ile KG üret
- Çıktıları JSON olarak al

### 10.2 “Graph analizi”
- Clustering çalıştır
- Cluster summarization + community report üret
- Graph kalite metrikleri/heuristic’ler ile değerlendirme yap

### 10.3 “UI ile incele”
- `examples/api_server_example.py` ile server aç
- KG’yi görsel olarak incele
- Deterministic query ile ilişkileri hızlı doğrula

---

## 11) DRG’nin Diğer Projelerden Farkı (Kısa Özet)

- **RAG/serving değil**: “Cevap veren sistem” değil, “graph çıkaran pipeline”.
- **DSPy + declarative**: Extraction davranışı şema ile tanımlanır; prompt karmaşası minimize edilir.
- **Dataset‑agnostic**: Aynı sistem farklı domain’lerde tekrar kullanılabilir.
- **Enhanced schema + metadata**: EntityType/RelationGroup + description/detail alanları ile zengin temsil.
- **Graph‑first analiz**: Clustering/community report gibi graph‑native çıktılar.


