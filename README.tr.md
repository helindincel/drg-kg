# DRG - Declarative Relationship Generation

[![CI](https://github.com/helindincel/drg-kg/actions/workflows/ci.yml/badge.svg)](https://github.com/helindincel/drg-kg/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/drg-kg.svg)](https://pypi.org/project/drg-kg/)
[![Python versions](https://img.shields.io/pypi/pyversions/drg-kg.svg)](https://pypi.org/project/drg-kg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typed: PEP 561](https://img.shields.io/badge/typed-PEP%20561-blue.svg)](https://peps.python.org/pep-0561/)

> **Kurulum:** `pip install drg-kg` — Python'da `import drg`. PyPI'deki `drg`
> paketi farklı bir projeye (Medicare DRG grouper) aittir.

DRG bir **şema odaklı Knowledge Graph extraction framework**'tür.
Yapılandırılmamış metni, deklaratif şemalar ve DSPy tabanlı extraction ile
açıklanabilir Knowledge Graph artifact'larına dönüştürür.

Ana vaat extraction'dır: domain şeması tanımlayın veya metinden türetin, typed
entity ve relation çıkarın, ardından incelenebilir, doğrulanabilir,
deterministik sorgulanabilir, versiyonlanabilir, değerlendirilebilir veya
export edilebilir bir `EnhancedKG` oluşturun. DRG bir GraphRAG framework
değildir, bir RAG framework değildir, retrieval veya chat application stack'i
değildir. Query, API, MCP, Neo4j, optimizer ve evaluation modülleri çıkarılmış
graph'ların etrafındaki destek katmanlarıdır.

İngilizce dokümantasyon için [`README.md`](README.md) dosyasına bakın.

## Alpha Durumu

DRG şu anda alpha aşamasındadır. Ana extraction API'leri alpha serisi için
stabil kabul edilir; JSON formatları, CLI flag'leri ve opsiyonel entegrasyon
yüzeyleri `v1.0` öncesinde değişebilir. Ciddi deneyler için versiyon pin'leyin
ve yükseltme öncesinde [`CHANGELOG.md`](CHANGELOG.md) dosyasını inceleyin.

## Neden DRG?

Metin, LLM ve graph ekosistemindeki çoğu araç problemin tek bir katmanını
çözer. DRG ise KG'nin tüm yaşam döngüsüne odaklanır: graph'ı oluşturmak,
zenginleştirmek, güncellemek, versiyonlamak, değerlendirmek, sorgulamak ve
downstream sistemlere açmak.

Birçok proje "metinden triple çıkarma" noktasında durur. DRG extraction'ı,
graph evrimi, kalite, izlenebilirlik ve entegrasyon etrafındaki daha uzun bir
mühendislik sürecinin ilk adımı olarak ele alır.

| Araç | En güçlü olduğu alan | DRG'nin farkı |
|:---|:---|:---|
| LangChain | LLM uygulama orkestrasyonu ve chain'ler | DRG KG lifecycle'a odaklanır: şema kontrollü extraction, provenance, versioning, evaluation ve deterministik graph query. |
| LlamaIndex | Doküman indeksleme ve LLM workflow yardımcıları | DRG graph-native yapı kurar; chat ve generation katmanları bu repo'nun ürün kapsamı dışındadır. |
| Neo4j | Kalıcı graph database ve Cypher sorguları | DRG KG'leri oluşturur, doğrular, zenginleştirir, versiyonlar, değerlendirir ve export eder; Neo4j downstream storage hedefi olabilir. |
| NetworkX | In-memory graph algoritmaları | DRG extraction, schema validation, provenance, temporal metadata, entity resolution, lifecycle operasyonları, CLI/API ve export workflow'ları ekler. |
| Raw DSPy programları | Typed LLM programları | DRG DSPy extraction'ı tam bir KG lifecycle'a sarar: chunking, schema generation, graph build, update, reasoning, evaluation ve serving. |

## Lifecycle Kapsamı

DRG'nin ana soyutlaması sadece "KG çıktısı" değildir. KG etrafındaki yaşam
döngüsünü paketler:

| Lifecycle aşaması | DRG'nin sorumluluğu |
|:---|:---|
| Tasarım | Domain için extraction şeması tanımlamak veya metinden türetmek. |
| Oluşturma | Entity/relation çıkarmak ve `EnhancedKG` oluşturmak. |
| Güven | Provenance, evidence, confidence ve validation sonuçlarını taşımak. |
| Evrim | Yeni dokümanları merge etmek, entity resolution yapmak, snapshot diff almak ve version tutmak. |
| Reasoning | Deterministik graph query, temporal lookup, multi-hop traversal ve rule-based inference çalıştırmak. |
| Değerlendirme | Extraction, graph query davranışı, graph structure ve performans regresyonlarını ölçmek. |
| Entegrasyon | CLI/API/MCP üzerinden sunmak, JSON export etmek ve Neo4j'ye sync etmek. |

## DRG Nedir / Ne Değildir?

DRG şunları hedefler:

- Bir Knowledge Graph extraction framework'ü olmak.
- Metinden typed entity ve relation çıkaran schema-first bir kütüphane olmak.
- Metinden gelen entity ve relation'lar için graph construction ve enrichment
  toolkit'i sağlamak.
- `EnhancedKG` etrafında deterministik query, evaluation, versioning,
  provenance ve export katmanı sunmak.
- Lokal deneyler ve entegrasyon prototipleri için CLI/API/MCP paketi olmak.

DRG şunlar değildir:

- Genel amaçlı bir LLM uygulama framework'ü.
- GraphRAG framework'ü.
- RAG framework'ü.
- Chatbot framework'ü.
- Database veya search backend.
- Retrieval framework.
- Vector veya hybrid search katmanı; embedding'ler serving index değil,
  entity resolution/export için opsiyonel yardımcı sinyallerdir.
- Neo4j, NetworkX, LangChain, LlamaIndex veya DSPy yerine geçen bir araç.
- Henüz hosted product veya tam stabil production platform.

## Mimari

```text
Yapılandırılmamış Metin
      |
      v
Schema / Auto-Schema
      |
      v
Chunking + DSPy Extraction
      |
      v
EnhancedKG
      |
      +--> Provenance / Confidence / Validation
      |
      +--> Entity Resolution / Incremental Updates / Versioning
      |
      +--> Temporal Metadata / Multi-Document Reasoning
      |
      v
Query + Reasoning + Evaluation
      |
      v
CLI / FastAPI UI / MCP / Neo4j Export / JSON
```

## Feature Matrix

| Alan | Özellik | Durum | Not |
|:---|:---|:---|:---|
| Extraction | Declarative schemas | Mevcut | Entity type, relation group, örnek ve metadata. |
| Extraction | Auto-schema generation | Mevcut | Ham metinden başlangıç şeması üretir. |
| Graph core | `EnhancedKG` | Mevcut | Typed node, edge, cluster ve JSON export. |
| Güven | Provenance | Mevcut | Evidence/source metadata node ve edge'lerle taşınabilir. |
| Güven | Confidence scoring | Mevcut | Confidence metadata ve filtering stratejileri. |
| Lifecycle | Versioning | Mevcut | Graph evrimi için snapshot, diff ve rollback yardımcıları. |
| Lifecycle | Incremental updates | Mevcut | Yeni dokümanları mevcut KG ile merge eder. |
| Intelligence | Entity resolution | Mevcut | Canonical entity merge ve alias yönetimi. |
| Intelligence | Temporal query | Mevcut | Timeline yardımcıları ve kompakt temporal lookup'lar. |
| Intelligence | Multi-document reasoning | Mevcut | Graph path ve bridge'ler üzerinde rule-based inference. |
| Integration | FastAPI + Cytoscape UI | Mevcut | Lokal graph exploration ve API endpoint'leri. |
| Integration | Neo4j export | Mevcut | Graph verisini Neo4j'ye sync/export eder. |
| Integration | MCP server | Mevcut | KG operasyonlarını MCP uyumlu client'lara açar. |
| Quality | Evaluation framework | Mevcut | Extraction, graph-query, structural ve performans metrikleri. |

## Experimental Özellikler

Aşağıdaki modüller araştırma ve entegrasyon deneyleri için kullanışlıdır, ancak
henüz dondurulmuş core extraction API'nin parçası değildir:

- Optimizer integration: DSPy optimizer workflow'ları experimentaldir; training
  data ve metrik pratikleri oturdukça değişebilir.
- Confidence calibration: confidence strategy'leri ve calibrated score'lar,
  kendi labelled verinizle fit edilmedikçe heuristic/experimentaldir.
- Long-document optimization: chunking, cross-chunk relation recovery ve
  windowed relation extraction gelişmeye devam etmektedir.
- API/UI/MCP entegrasyon detayları: dokümante komut ve endpoint'lerin çalışması
  hedeflenir, ancak internal response shape ve UI implementation detayları
  değişebilir.

## Kullanım Alanları

- Haber analizi: haberlerden kişi, şirket, olay, satın alma, çatışma ve
  timeline bilgisi çıkarmak.
- Kurumsal dokümanlar: politika, rapor, sözleşme ve iç notları açıklanabilir
  graph yapılarına dönüştürmek.
- Araştırma raporları: makale veya teknik dokümanlar arasında bulgu, yöntem,
  entity, dataset ve citation bağlantıları kurmak.
- Çok dokümanlı bilgi birleştirme: birçok kaynaktan gelen parçalı bilgileri
  provenance ile tek graph altında toplamak.
- Knowledge graph operasyonları: çıkarılan fact'leri query edilebilir,
  versiyonlanabilir, açıklanabilir ve downstream graph database ya da analytics
  kullanımlarına hazır tutmak.

## Roadmap

`v0.2` hedefleri:

- Top-level Python extraction API'sini alpha serisi boyunca stabil tutmak.
- API key gerektirmeyen ve mock'lu demoları genişleterek yeni kullanıcıların
  DRG'yi hızlı değerlendirmesini sağlamak.
- Extraction, temporal metadata ve graph query için evaluation kapsamını
  iyileştirmek.
- API, MCP, Neo4j ve benchmark flow'ları için opsiyonel integration testlerini
  güçlendirmek.

`v1.0` hedefleri:

- Stabil public API sınırları ve migration policy taahhüt etmek.
- Production-ready package metadata ve release workflow yayınlamak.
- Generated API reference docs ve daha net architecture decision record'ları
  sağlamak.
- Daha geniş regression ve benchmark kapsamıyla graph correctness güvenini
  yükseltmek.

## Related Work

DRG birkaç ekosistemdeki fikirlerin üzerine inşa edilir:

- [DSPy](https://github.com/stanfordnlp/dspy): typed LLM programları ve
  optimization.
- [LangChain](https://github.com/langchain-ai/langchain): LLM uygulama
  orkestrasyonu.
- [LlamaIndex](https://github.com/run-llama/llama_index): data source'lar
  üzerinde indexing.
- [Neo4j](https://neo4j.com/): graph persistence ve Cypher query.
- [NetworkX](https://networkx.org/): in-memory graph algoritmaları.

## Kurulum

DRG Python 3.10, 3.11, 3.12 ve 3.13 üzerinde desteklenir.

```bash
# Kaynak kod checkout'ı, minimal graph/query kullanımı
pip install -e .

# Geliştirme araçları
pip install -e ".[dev]"

# Odaklı opsiyonel kurulumlar
pip install -e ".[extract]"  # DSPy extraction
pip install -e ".[api]"      # FastAPI UI
pip install -e ".[mcp]"      # MCP server
pip install -e ".[neo4j]"    # Neo4j export
```

Public PyPI release sonrasında ihtiyacınız olan extra'larla kurulum:

```bash
pip install drg-kg                  # Çekirdek graph/query (LLM extraction yok)
pip install "drg-kg[extract]"       # DSPy extraction + tiktoken chunking
pip install "drg-kg[all]"           # Tüm opsiyonel entegrasyonlar
```

## Hızlı Başlangıç

Tam ilk çalıştırma rehberi için
[`docs/getting_started.md`](docs/getting_started.md) dosyasına bakın.

İlk çalıştırılabilir örnek API key veya LLM gerektirmez:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
drg --help
python examples/query_layer_example.py
```

Canlı extraction için bir model provider yapılandırılmalıdır:

```bash
pip install -e ".[extract]"
cp .env.example .env

export DRG_MODEL=openai/gpt-4o-mini
export OPENAI_API_KEY=sk-...

# Gemini
export DRG_MODEL=gemini/gemini-2.0-flash-exp
export GEMINI_API_KEY=...

# Lokal Ollama
export DRG_MODEL=ollama_chat/llama3
export DRG_BASE_URL=http://localhost:11434
```

Production extraction için LM yoksa hızlı hata almak önerilir. `DRG_REQUIRE_LM=1`
veya `DRG_STRICT=1` ayarlandığında eksik LM konfigürasyonu `LLMConfigError`
üretir; bu bayraklar yokken Python API testler ve offline örnekler için boş
extraction döndürmeye devam eder.

Hikaye odaklı bir CLI extraction örneği:

```bash
drg extract hikaye.txt --auto-schema --output-format enhancedkg -o hikaye_kg.json
drg validate hikaye_kg.json
```

İlk yayın senaryosu için yaklaşık 20 sayfalık bir hikaye dokümanı ile
başlayın. DRG dokümanı örnekler, entity type, example, property, relation group
ve relation example içeren detaylı bir `EnhancedDRGSchema` oluşturur; ardından
bu şemayı DSPy’ye structured input olarak verip KG çıkarır. Çıktı
doğrulanabilir, diff alınabilir, version'lanabilir, deterministik sorgulanabilir
veya export edilebilir bir graph artifact'ıdır.

Python API ile kullanım:

```python
from drg import EnhancedDRGSchema, EntityType, Relation, RelationGroup, extract_typed
from drg.graph.builders import build_enhanced_kg

schema = EnhancedDRGSchema(
    entity_types=[
        EntityType(
            name="Place",
            description="Hikayedeki mekanlar ve anlamlı sahneler",
            examples=["eski liman", "arşiv kulesi"],
            properties={"atmosphere": "duygu veya duyusal nitelik", "role": "hikayedeki işlev"},
        ),
        EntityType(
            name="Artifact",
            description="Olay örgüsü, hafıza veya sembolik anlam taşıyan nesneler",
            examples=["gümüş pusula", "mühürlü mektup"],
            properties={"condition": "gözlenen durum", "significance": "neden önemli olduğu"},
        ),
        EntityType(
            name="Conflict",
            description="Gerilimler, engeller veya çözülmemiş hikaye problemleri",
            examples=["kapalı geçit", "bozulmuş ittifak"],
            properties={"stakes": "kaybedilebilecek şey", "status": "açık/çözülmüş vb."},
        ),
    ],
    relation_groups=[
        RelationGroup(
            name="narrative_structure",
            description="Hikaye öğelerinin birbirini nasıl şekillendirdiği",
            relations=[
                Relation(
                    "reveals",
                    "Artifact",
                    "Conflict",
                    description="Bir nesne bir çatışmayı açığa çıkarır veya netleştirir",
                ),
                Relation(
                    "located_in",
                    "Artifact",
                    "Place",
                    description="Bir nesne bir mekanda bulunur veya saklanır",
                ),
            ],
        )
    ],
)

text = "Arşiv kulesindeki mühürlü mektup, liman kapısının neden kilitli olduğunu gösterdi."
entities, triples = extract_typed(text, schema)
kg = build_enhanced_kg(entities_typed=entities, triples=triples, schema=schema, source_text=text)

print(kg.to_json())
```

API key ayarlamadan deterministik repo demosunu denemek için:

```bash
python examples/query_layer_example.py
```

## Public API Stabilitesi

Tercih edilen Python API isimleri alpha serisi için dondurulmuş kabul edilir:
`extract_typed()`, `extract_from_chunks()`, `extract_typed_async()`,
`extract_from_chunks_async()` ve backward-compatible `extract_triples()`
wrapper'ı. Bu isimler ve temel return davranışları changelog girdisi ve
migration notu olmadan değiştirilmemelidir.

`DRGSchema`, `EnhancedDRGSchema`, `EntityType`, `Relation` gibi schema tipleri
ve `build_enhanced_kg()` builder'ı da dokümante kullanıcı yüzeyinin parçasıdır.
Optimizer, API, MCP, UI ve prompt internallerinden deep import'lar experimental
kalmaya devam eder. Detaylar için [`docs/public_api.md`](docs/public_api.md)
dosyasına bakın.

## Örnek Galerisi

| Örnek | Ne gösterir? |
|:---|:---|
| [`examples/quickstarts/01_wikipedia_article.py`](examples/quickstarts/01_wikipedia_article.py) | Inline schema ile küçük biyografik/ansiklopedik extraction. |
| [`examples/quickstarts/02_financial_news.py`](examples/quickstarts/02_financial_news.py) | Kurumsal ve finans haberlerinden entity/relation extraction. |
| [`examples/quickstarts/03_biomedical.py`](examples/quickstarts/03_biomedical.py) | Biyomedikal ilaç, hastalık ve gen graph extraction. |
| [`examples/full_pipeline_example.py`](examples/full_pipeline_example.py) | Chunking, extraction, graph build, clustering ve report içeren uçtan uca pipeline. |
| [`examples/api_server_example.py`](examples/api_server_example.py) | Lokal FastAPI server ve Cytoscape graph UI. |
| [`examples/incremental_update_example.py`](examples/incremental_update_example.py) | Yeni dokümanları mevcut graph'a merge etme. |
| [`examples/temporal_query_example.py`](examples/temporal_query_example.py) | Temporal metadata ve timeline tarzı sorgular. |
| [`examples/query_layer_example.py`](examples/query_layer_example.py) | Deterministik graph query layer kullanımı. |
| [`examples/multi_document_reasoning_example.py`](examples/multi_document_reasoning_example.py) | Cross-document reasoning ve inferred graph bridge'ler. |
| [`examples/event_extraction_example.py`](examples/event_extraction_example.py) | Event-oriented extraction pipeline. |
| [`examples/evaluation_framework_example.py`](examples/evaluation_framework_example.py) | Evaluation metrikleri ve rapor üretimi. |
| [`examples/mcp_demo.py`](examples/mcp_demo.py) | MCP entegrasyon akışı. |
| [`examples/optimizer_demo.py`](examples/optimizer_demo.py) | Extraction etrafında DSPy optimizer deneyleri. |

Input -> output artifact walkthrough için:
[`docs/input_output_examples.tr.md`](docs/input_output_examples.tr.md).

## CLI

| Komut | Amaç |
|:---|:---|
| `drg extract` | Dosyadan veya stdin'den KG çıkarır. |
| `drg validate` | Kaydedilmiş KG JSON dosyasını doğrular. |
| `drg diff` | İki KG snapshot'ını karşılaştırır. |
| `drg versions list` | Graph version snapshot'larını listeler. |
| `drg versions diff` | Graph versiyonlarını karşılaştırır. |
| `drg versions rollback` | Önceki graph versiyonuna döner. |
| `drg eval run` | Benchmark dataset'i çalıştırır. |
| `drg eval list` | Bundled benchmark dataset ve adapter'ları listeler. |
| `drg eval compare` | Evaluation report'larını karşılaştırır. |

Incremental update örneği:

```bash
drg extract yeni_makale.txt --update global_kg.json --infer
drg validate global_kg.json
drg diff onceki_kg.json global_kg.json --json
```

## API, UI, MCP ve Evaluation

```bash
# Etkileşimli Cytoscape UI
python examples/api_server_example.py

# Neo4j sync önizlemesi
curl -X POST "http://localhost:8000/api/neo4j/sync?dry_run=true"

# Cursor / Claude Desktop için MCP server
python -m drg.mcp_server

# Benchmark çalıştırma
drg eval run examples/benchmarks/corporate_acquisition_benchmark.json \
  --measure-performance \
  -o reports/current.json \
  --markdown-output reports/current.md
```

Detaylar için [`docs/api_server.md`](docs/api_server.md),
[`docs/mcp_integration.md`](docs/mcp_integration.md) ve
[`docs/evaluation_framework.md`](docs/evaluation_framework.md) dosyalarına
bakın.

## Proje Haritası

```text
drg/
├── schema.py              # Enhanced schema tanımları
├── extract/               # DSPy tabanlı extraction
├── chunking/              # Token ve sentence chunker'lar
├── graph/                 # EnhancedKG, provenance, diff, versioning
├── query/                 # Deterministik query ve analytics katmanı
├── temporal/              # Zamansal reasoning ve timeline yardımcıları
├── reasoning/             # Çoklu doküman inference
├── evaluation/            # Metrikler, raporlar, benchmark adapter'ları
├── api/                   # FastAPI server ve Cytoscape UI
├── events/                # Event extraction pipeline'ı
└── cli.py                 # CLI giriş noktası
```

## Dokümantasyon

- İlk çalıştırma: [`docs/getting_started.md`](docs/getting_started.md)
- Kurulum ve konfigürasyon: [`docs/setup.md`](docs/setup.md)
- Mimari: [`docs/project_overview.md`](docs/project_overview.md)
- Pipeline: [`docs/pipeline_overview.md`](docs/pipeline_overview.md)
- Input/output walkthrough: [`docs/input_output_examples.tr.md`](docs/input_output_examples.tr.md)
- Şema tasarımı: [`docs/schema_design.md`](docs/schema_design.md)
- Public API: [`docs/public_api.md`](docs/public_api.md)
- Benchmarking: [`docs/benchmarking.md`](docs/benchmarking.md)
- Quickstart scriptleri: [`examples/quickstarts/README.md`](examples/quickstarts/README.md)

## Known Limitations

- DRG Knowledge Graph artifact'ları çıkarır ve bu artifact'lar üzerinde işlem
  yapar; GraphRAG veya RAG serving stack'i sağlamaz.
- Optimizer integration experimentaldir ve stabil production tuning katmanı
  değil, research workflow olarak ele alınmalıdır.
- Long-document optimization gelişmektedir; chunking ve cross-chunk relation
  recovery domain'e göre tuning gerektirebilir.
- Confidence calibration, domain'iniz için labelled data ile kalibre edilmedikçe
  heuristic/experimentaldir.
- Live extraction kalitesi kullanılan LLM'e, schema kalitesine ve provider
  davranışına bağlıdır.

## Geliştirme

```bash
pip install -e ".[dev]"
pytest tests/
ruff check .
mypy drg
```

Katkı yönergeleri [`CONTRIBUTING.md`](CONTRIBUTING.md), güvenlik bildirimi
süreci [`SECURITY.md`](SECURITY.md) içinde yer alır.

## Lisans

MIT © [Helin Dinçel](https://github.com/helindincel)
