# DRG — Declarative Relationship Generation

DRG, metinlerden **bilgi grafiği (Knowledge Graph)** çıkarımı yapan, **DSPy
tabanlı declarative** bir Python kütüphanesidir. Şemayı tanımlarsın, DRG entity
ve relation extraction'ı üstlenir; üzerine clustering, community report ve
görselleştirme katmanları ekler.

> **⚠️ Alpha:** Bu repo `0.1.0a0` sürümündedir. API değişiklikleri olabilir.

---

## Proje Hakkında Detaylı Genel Bakış

Projeyi hiç bilmeyen biri için **mimari ve felsefe** dokümanı:

- [`docs/project_overview.md`](docs/project_overview.md)

Bu doküman özellikle şunları netleştirir:

- DRG'nin **DSPy tabanlı, declarative** extraction yaklaşımı
- DRG'nin **bir RAG/serving framework olmadığı** (UI query: deterministic KG lookup)
- Dataset‑agnostic tasarım ve Enhanced schema yaklaşımı
- Pipeline akışı, UI ve repo yapısı

---

## Özellikler

- **Declarative Schema** — Entity tip ve ilişkilerini tanımla, gerisini DRG halletsin
- **DSPy Entegrasyonu** — TypedPredictor ile yapılandırılmış extraction
- **Enhanced Schema** — `EntityType`, `RelationGroup`, `EntityGroup`, `PropertyGroup` ile zengin tanımlar
- **Otomatik Schema Üretimi** — `generate_schema_from_text()` ile metinden şema
- **Chunk‑Based Extraction** — Uzun metinler için bağlam‑aware chunking
- **Knowledge Graph Katmanı** — `EnhancedKG` (KGNode, KGEdge, Cluster)
- **Clustering & Community Reports** — Louvain / Leiden / Spectral + summarization
- **API Server + UI** — FastAPI + Cytoscape.js tabanlı interaktif görselleştirme
- **Çoklu LLM Desteği** — OpenAI, Gemini, Anthropic, Perplexity, OpenRouter, Ollama
- **Neo4j Export (opsiyonel)** — Graph persistence

---

## Kurulum

```bash
git clone https://github.com/helindincel/drg-kg.git
cd drg-kg

# Sadece çekirdek
pip install -e .

# Tüm opsiyonel özellikler ile (api, embedding, clustering, vs.)
pip install -e ".[all]"

# Developer modu (test + lint + type-check araçları ile)
pip install -e ".[dev]"
```

Detaylı kurulum ve sorun giderme için: [`docs/setup.md`](docs/setup.md)

### Gereksinimler

- Python `>= 3.10`
- `dspy >= 2.5.0, < 3.0.0`
- `pydantic >= 2.0.0`

---

## Konfigürasyon

DRG, davranışını environment variable'larla yönetir:

```bash
cp .env.example .env
# .env dosyasını editleyip ilgili API key'i doldur.
```

Tipik değişkenler:

| Değişken | Varsayılan | Açıklama |
|----------|------------|----------|
| `DRG_MODEL` | `openai/gpt-4o-mini` | DSPy/LiteLLM model id |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / ... | — | İlgili provider için API key |
| `DRG_TEMPERATURE` | `0.0` | LLM temperature |
| `DRG_MAX_TOKENS` | `1500` | LLM output bütçesi |
| `DRG_BASE_URL` | — | Ollama / self-hosted gateway |

Tüm değişkenler için: [`docs/setup.md`](docs/setup.md)

---

## Hızlı Başlangıç

### Basit kullanım (legacy schema)

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

### Enhanced schema (önerilen)

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

---

## CLI

```bash
# Dosyadan çıkarım
drg extract input.txt -o output.json

# Standart girişten
echo "Apple released iPhone 16" | drg extract - -o output.json

# Özel model ile
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

Detaylar ve endpoint listesi: [`docs/api_server.md`](docs/api_server.md)

---

## API Referansı (özet)

### Şema sınıfları

| Sınıf | Kullanım |
|-------|----------|
| `DRGSchema` | Legacy, basit `Entity` + `Relation` |
| `EnhancedDRGSchema` | Önerilen, `EntityType` + `RelationGroup` ile zengin |

### Extraction

```python
extract_typed(text, schema)
# -> (entities, triples)
#    entities: List[Tuple[str, str]]            # (entity_name, entity_type)
#    triples:  List[Tuple[str, str, str]]       # (source, relation, target)

extract_triples(text, schema)
# Yalnızca triples (geriye dönük uyumluluk için).
```

### Knowledge Graph

```python
kg = KG.from_typed(entities, triples)
kg = KG.from_triples(triples)
print(kg.to_json(indent=2))
```

Zengin KG (`EnhancedKG`, `KGNode`, `KGEdge`, `Cluster`) için `drg.graph`
modülüne bakın.

---

## Proje Yapısı

```
DRG/
├── drg/                       # Ana paket (monolithic codebase)
│   ├── __init__.py            # Public API + lazy loading
│   ├── schema.py              # EnhancedDRGSchema, EntityType, RelationGroup, ...
│   ├── protocols.py           # Structural interfaces (KGExtractor / Embedding / Clustering / LLM)
│   ├── errors.py              # Typed exception hierarchy (DRGError + 11 subclasses)
│   ├── config.py              # LMConfig — DSPy LM kurulumu (env-driven)
│   ├── extract/               # DSPy extraction paketi (KGExtractor + cross-chunk + heuristics)
│   ├── coreference_resolution/# Pronoun resolution (strategy pattern: NLP + heuristic)
│   ├── entity_resolution/     # Entity merging (strategy pattern: String + Hybrid)
│   ├── chunking/              # Token / sentence / semantic chunker'lar
│   ├── embedding/             # Provider abstraction (OpenAI / Gemini / Local / OpenRouter)
│   ├── graph/                 # EnhancedKG, schema_generator, community_report,
│   │                          # relationship_model/ (paket), visualization_adapter/ (paket),
│   │                          # hub_mitigation, query_engine, auto_clusters, neo4j_exporter
│   ├── clustering/            # Louvain / Leiden / Spectral + summarization
│   ├── optimizer/             # DSPy optimizer + metrics
│   ├── api/                   # FastAPI server + Cytoscape UI
│   ├── mcp_api.py             # MCP entegrasyonu
│   ├── cli.py                 # `drg` CLI giriş noktası
│   └── utils/                 # env_loader, llm_throttle, strict, logging, cache (shared LRU)
├── docs/                      # Dokümantasyon (KOD YOK)
├── examples/                  # full_pipeline_example, api_server_example, ...
├── tests/                     # Unit + integration + multi_dataset evaluation
├── outputs/                   # Üretilen artifact'lar (gitignored)
├── inputs/                    # Örnek metin dosyaları
├── pyproject.toml             # Tek doğruluk kaynağı (deps + tooling)
└── README.md
```

---

## Test ve Geliştirme

```bash
# Developer kurulumu
pip install -e ".[dev]"

# Non-integration testler (API key gerekmez)
pytest -m "not integration"

# Coverage ile
pytest -m "not integration" --cov=drg --cov-report=term-missing

# Lint + format
ruff check drg tests examples
ruff format drg tests examples

# Type check
mypy drg

# Pre-commit hook'larını kur
pre-commit install
```

---

## Desteklenen Modeller

DRG, DSPy/LiteLLM üzerinden aşağıdaki provider'ları destekler. Model ID'leri
`provider/model` formatındadır:

- **OpenAI** — `openai/gpt-4o-mini`, `openai/gpt-4`, ...
- **Google Gemini** — `gemini/gemini-2.0-flash-exp`, ...
- **Anthropic** — `anthropic/claude-3-5-sonnet`, ...
- **Perplexity** — `perplexity/llama-3.1-sonar-large-128k-online`, ...
- **OpenRouter** — `openrouter/<model>`
- **Ollama (local)** — `ollama_chat/llama3`, `ollama_chat/mistral`, ...

Model `DRG_MODEL` environment variable'ı ile seçilir.

---

## Opsiyonel Bağımlılıklar

Modüler bağımlılık yapısı:

| Extra | İçerik |
|-------|--------|
| `api` | FastAPI, uvicorn |
| `neo4j` | Neo4j driver |
| `openai` / `gemini` / `openrouter` | LLM/embedding client'ları |
| `local` | sentence-transformers (local embedding) |
| `louvain` / `leiden` / `spectral` | Clustering backend'leri |
| `networkx` | Graph processing |
| `coreference` | spaCy + coreferee |
| `dev` | pytest, ruff, mypy, pre-commit, pytest-cov |
| `all` | Yukarıdakilerin hepsi |

---

## Belgelendirme

- [`docs/project_overview.md`](docs/project_overview.md) — Mimari + felsefe
- [`docs/setup.md`](docs/setup.md) — Detaylı kurulum
- [`docs/api_server.md`](docs/api_server.md) — API + UI kullanımı
- [`docs/pipeline_overview.md`](docs/pipeline_overview.md) — Pipeline akışı
- [`docs/schema_design.md`](docs/schema_design.md) — Şema tasarım prensipleri
- [`docs/chunking_strategy.md`](docs/chunking_strategy.md) — Chunking stratejileri
- [`docs/relationship_model.md`](docs/relationship_model.md) — İlişki modeli
- [`docs/clustering_summarization.md`](docs/clustering_summarization.md) — Clustering
- [`docs/optimizer_design.md`](docs/optimizer_design.md) — DSPy optimizer
- [`docs/mcp_integration.md`](docs/mcp_integration.md) — MCP entegrasyonu

---

## Lisans

MIT — Detaylar için `LICENSE` dosyasına bakın.
