# DRG-KG — Staff-Level Code Review

> **Tarih:** 2026-06-25  
> **Yöntem:** Gerçek kaynak kodu incelendi. README hiçbir zaman doğru kabul edilmedi.  
> Her bulgu bir kod referansına dayanır.

---

## Executive Summary

DRG-KG, alpha aşamasında, mimarisi sağlam ve konsepti güçlü bir Python paketidir.
Schema tasarımı, graph core, query layer, provenance sistemi ve test altyapısı gerçekten
iyi işlenmiş. Ancak repository'nin **iki P0 runtime crash'i** var: `drg/reasoning/` ve
`drg/evaluation/` modülleri CLI'dan referans edilmekte ama disk üzerinde bulunmamaktadır.
Bu durum belgelenen birden fazla CLI komutunu tamamen işlevsiz kılmaktadır. CI workflow'ları
da var olmayan GitHub Actions major version'larını referans etmektedir. Bu blokerlerin
ötesinde API server'da önemli thread-safety sorunları ve README'nin vaat ettiklerinin
ile gerçek implementasyon arasında ciddi bir uçurum bulunmaktadır.

---

## 1. Feature Completeness

| Özellik | Durum | Kanıt |
|:---|:---|:---|
| Declarative schemas | ✅ Tam | `drg/schema.py` — `DRGSchema`, `EnhancedDRGSchema`, `from_dict`, `to_dict`, `load_schema_from_json` |
| Auto-schema generation | ✅ Tam | `drg/extract/_schema_gen.py` — `generate_schema_from_text` via DSPy TypedPredictor |
| Chunking | ✅ Tam | `drg/chunking/strategies.py` — `TokenBasedChunker`, `SentenceBasedChunker`; `drg/chunking/validators.py` |
| DSPy extraction | ✅ Tam | `drg/extract/__init__.py` — `extract_typed`, `extract_from_chunks`, `KGExtractor(dspy.Module)` |
| EnhancedKG | ✅ Tam | `drg/graph/kg_core.py` — `KGNode`, `KGEdge`, `Cluster`, `EnhancedKG`; JSON/JSON-LD/enriched export |
| Provenance | ✅ Tam | `drg/graph/provenance.py` — `ProvenanceRecord`, `attach_provenance`, `find_text_provenance` |
| Confidence scoring | ✅ Tam | `drg/confidence/` — `ConfidenceStrategy`, `DefaultConfidenceStrategy`, `ConfidenceScore` |
| Entity resolution | ✅ Tam | `drg/entity_resolution/` — strategy-based, string/embedding similarity, adaptive threshold |
| Coreference resolution | ✅ Tam | `drg/coreference_resolution/` — heuristic + NLP stratejileri |
| Versioning | ✅ Tam | `drg/graph/versioning.py` — `create_snapshot`, `list_versions`, `diff_versions`, `rollback_to_version` |
| Incremental updates | ✅ Tam | `drg/graph/incremental.py` — `GraphMerger`, `KGDiff`, `MergeStrategy` |
| Temporal query | ✅ Tam | `drg/temporal/` — `TemporalScope`, `is_active_at`, `detect_conflicts`, `build_timeline` |
| Graph query layer | ✅ Tam | `drg/query/` — `GraphQuery`, `InMemoryBackend`, `bfs_neighborhood`, `find_paths`, `explain` |
| Clustering | ✅ Tam | `drg/clustering/` — Louvain, Leiden, Spectral; optional deps gerektirir |
| Community reports | ✅ Tam | `drg/graph/community_report.py` |
| FastAPI + Cytoscape UI | ✅ Tam | `drg/api/server.py` — `/api/graph`, `/api/query`, Neo4j sync, healthz/readyz |
| CLI | 🟡 Kısmi | `extract`, `validate`, `diff`, `versions` çalışıyor; **`eval` ve `--infer` crash** |
| MCP server | ✅ Tam | `drg/mcp_server.py` — `drg_define_schema`, `drg_extract`, `drg_build_kg`, `drg_get_kg`, `drg_export_kg` |
| Neo4j export | ✅ Tam | `drg/graph/neo4j_exporter.py` — `Neo4jExporter`, `build_neo4j_sync_plan` |
| Events | ✅ Tam | `drg/events/` — `EventTypeRegistry`, `extract_events`, KG graph mapping |
| Embedding providers | ✅ Tam | `drg/embedding/providers.py` — OpenAI, Gemini, OpenRouter, Local |
| Evaluation framework | 🔴 **EKSİK** | `drg/evaluation/` disk üzerinde yok; `drg eval run/compare/list` import'ta crash |
| Multi-document reasoning | 🔴 **EKSİK** | `drg/reasoning/` disk üzerinde yok; `drg extract --infer` import'ta crash |
| DSPy optimizer | 🔴 **EKSİK** | `drg/optimizer/` disk üzerinde yok; README'nin `optimizer_demo.py`'si de yok |
| Benchmarking | 🔴 **EKSİK** | `drg/evaluation/`'a bağımlı; tüm eval CLI yüzeyi bozuk |

---

## 2. Missing Features

| Kategori | Eksik Öğe |
|:---|:---|
| Core modül | `drg/reasoning/` — `MultiDocumentReasoner`, `ReasoningConfig` (`cli.py:506`'da import edilmiş) |
| Core modül | `drg/evaluation/` — `BenchmarkRunner`, `PipelinePrediction`, `compare_reports`, vb. (`cli.py:550`) |
| Core modül | `drg/optimizer/` — `optimizer.py`, `metrics.py` (README, coverage config, CHANGELOG'da referans) |
| Core modül | `drg/graph/schema_generator.py` (coverage omit ve CHANGELOG'da referans) |
| Örnek | `examples/multi_document_reasoning_example.py` (README tablosunda, `docs/multi_document_reasoning.md`'de) |
| Örnek | `examples/evaluation_framework_example.py` (README tablosunda) |
| Örnek | `examples/optimizer_demo.py` (README tablosunda) |
| CLI | `drg eval run` — her zaman crash: `ModuleNotFoundError: No module named 'drg.evaluation'` |
| CLI | `drg extract --infer` — her zaman crash: `ModuleNotFoundError: No module named 'drg.reasoning'` |
| API | `/api/extract` endpoint'inde rate limiting yok |
| API | `api_key` alanı access log'lara sızma riski taşıyor |
| Konfigürasyon | `DRG_CORS_ORIGINS` için dokümantasyon yok; default `*` production'da tehlikeli |
| Test | `drg/utils/env_loader.py`, `drg/utils/llm_throttle.py` için test yok |
| Test | `drg eval run` CLI path'i için integration test yok (zaten broken) |
| Dokümantasyon | `docs/evaluation_framework.md` var olmayan modülü açıklıyor |
| Dokümantasyon | `docs/multi_document_reasoning.md` — `drg/reasoning/`'ı "source of truth" olarak referans ediyor |

---

## 3. Code Quality

### 3.1 P0 Runtime Crash'ler

**`drg/cli.py` satır 506 ve 550 — var olmayan modüllerden import:**

```python
# satır 506 — _handle_extract() içinde, sadece --infer flag'i geçildiğinde ulaşılıyor
from .reasoning import MultiDocumentReasoner, ReasoningConfig   # ModuleNotFoundError

# satır 550 — _handle_eval() içinde, herhangi bir `drg eval` subcommand'ı için ulaşılıyor
from .evaluation import (
    BenchmarkRunner,
    PipelinePrediction,
    ...
)   # ModuleNotFoundError
```

Her iki import da lazy function body içinde (iyi pattern), ancak altta yatan modüller
basitçe yok. `drg eval list` çalıştıran bir kullanıcı stack trace alır.

---

### 3.2 Thread-Safety Race Condition — API Server

**`drg/api/server.py` — `_apply_extraction_env()` global `os.environ`'ı mutate ediyor:**

```python
def _apply_extraction_env(request: ExtractRequest) -> dict[str, str | None]:
    ...
    if request.model:
        os.environ["DRG_MODEL"] = request.model     # ← process-global, thread-safe değil
    if request.api_key:
        os.environ["OPENAI_API_KEY"] = request.api_key   # ← eş zamanlı isteklerle race
    return previous
```

FastAPI async event loop kullanır. Birden fazla eş zamanlı istek `os.environ` üzerinde
race condition oluşturur: bir kullanıcının API key'i başka bir kullanıcının extraction
context'ine sızabilir. Bu hem güvenlik hem de doğruluk açısından kritik bir hata.

**Kanıt:** `drg/api/server.py:270–300`; `/api/extract` ve `/api/graph/update` route'larında çağrılıyor.

---

### 3.3 CORS Misconfiguration

**`drg/api/server.py:617–624`:**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,    # default ["*"]
    allow_credentials=True,         # ← tarayıcılar wildcard origins'e credentialed CORS'u bloklar
    ...
)
```

`allow_credentials=True` ile `allow_origins=["*"]` kombinasyonu Fetch spesifikasyonu (§3.2)
tarafından reddedilir ve production'da CORS hatalarına neden olur.

---

### 3.4 Silent Exception Swallowing

**`drg/graph/builders.py:302–306`:**
```python
except Exception:
    pass  # ← evidence snippet extraction sessizce başarısız oluyor; log yok
```

**`drg/confidence/_default.py:101`:**
```python
except Exception:
    pass  # ← confidence scoring başarısızlığı sessizce default döner
```

Bu bare `except Exception: pass` blokları bug'ları gizler ve production hata ayıklamasını
imkânsız kılar.

---

### 3.5 `config.py`'de Hard `import dspy` — Guard Yok

**`drg/config.py:13`:**
```python
import dspy   # ← top-level, try/except yok
```

`dspy` opsiyonel bir bağımlılıktır (yalnızca `[extract]`/`[dspy]` extra'larında).
`drg.config` modülüne herhangi bir kod yolu `dspy` kurulu olmadan ulaşırsa
`ModuleNotFoundError` fırlatır.

---

### 3.6 God Function: `_handle_extract`

**`drg/cli.py:370–545`:** `_handle_extract()` ~200 satır uzunluğunda olup şu sorumlulukları
tek başına üstlenir: env var mutasyonu, schema yükleme, schema üretimi, chunking config,
extraction (2 farklı path), event extraction, KG build, incremental merge, snapshot oluşturma,
reasoning çağrısı, output yazma, secret redaction. SRP ihlali.

---

### 3.7 God Function: `create_app`

**`drg/api/server.py:580–1050`:** Tüm 20+ FastAPI route handler'ı `create_app()` içinde
nested closure olarak tanımlanmış. Tek bir route'u test etmek için tüm factory'yi
instantiate etmek gerekiyor.

---

### 3.8 `Cluster` / `ClusterResult` İsim Çakışması

**`drg/clustering/algorithms.py:58`:**
```python
Cluster = ClusterResult   # algoritma çıktısı: int cluster_id, list nodes, list edges
```

**`drg/graph/kg_core.py`:**
```python
@dataclass
class Cluster:            # KG storage nesnesi: str id, set node_ids, dict metadata
```

`drg.clustering.Cluster` ve `drg.graph.Cluster` tamamen farklı, uyumsuz sınıflardır.

---

### 3.9 `pyproject.toml` Coverage Omit Listesinde Stale Girişler

**`pyproject.toml` `[tool.coverage.run]` omit listesi:**
```toml
"drg/mcp_api.py",                    # dosya silindi
"drg/graph/visualization_adapter.py", # package dir, glob eşleşmiyor
"drg/optimizer/optimizer.py",         # modül yok
"drg/optimizer/metrics.py",           # modül yok
"drg/graph/schema_generator.py",      # dosya yok
```

Bu stale girişler gerçek coverage ölçümünü bozar ve 70% floor'un sahte güven vermesine yol açar.

---

### 3.10 CI Workflow Versiyonları Var Olmayan Major Version'lara İşaret Ediyor

**`.github/workflows/ci.yml`:**
```yaml
uses: actions/checkout@v6        # v6 yok; current stable v4
uses: actions/setup-python@v6   # v6 yok; current stable v5
uses: actions/upload-artifact@v7 # v7 yok; current stable v4
```

**`.github/workflows/release.yml`:**
```yaml
uses: actions/download-artifact@v8  # v8 yok; current stable v4
```

GitHub Actions bu version'ları çözemez. Her push/PR'da CI tamamen başarısız olur.

---

## 4. Architecture Review

| Sorun | Ciddiyet | Kanıt |
|:---|:---|:---|
| `_handle_extract` / `_handle_eval` — SRP ihlali | Yüksek | `cli.py:390–545` |
| `create_app` — God function | Yüksek | `api/server.py:580–1050`, 800+ satır nested closure |
| `drg/extract/__init__.py` — 1000+ satır god modül | Orta | "Refactored" package olmasına rağmen |
| `os.environ` mutasyonu per-request config için | **Kritik** | `api/server.py:_apply_extraction_env()` |
| `LMConfig._configured` class-level state — singleton | Düşük | `config.py:26–28`; test izolasyonu yok |
| `drg.graph.__init__` tüm alt modülleri eager import ediyor | Düşük | `from drg import KG` neo4j, visualization vb.'yi de çeker |
| `ClusterResult`/`Cluster` isim çakışması | Orta | `clustering/algorithms.py:58`, `graph/kg_core.py:Cluster` |

**Circular import tespit edilmedi.** `TYPE_CHECKING` guard'ları ve `drg/__init__.py`'deki
lazy `__getattr__` döngüsel bağımlılıkları temiz biçimde önlüyor.

---

## 5. Public API Review

| Bulgu | Kanıt |
|:---|:---|
| `drg.__all__` deprecated `"KG"` sınıfını first-class export olarak listeliyor | `drg/__init__.py:64` |
| `drg.__all__` legacy shim olan `"extract_triples"`'ı listeliyor | `drg/extract/__init__.py` |
| `drg.graph.__all__` implementation detail olan `"SnapshotDiff"` ve `"Neo4jSyncPlan"`'ı export ediyor | `drg/graph/__init__.py:142,155` |
| `_sample_text_for_schema_generation` test erişimi için `# noqa: F401` ile semi-public yapılmış | `drg/extract/__init__.py:49` |
| `mcp_server._schemas` / `_knowledge_graphs` module-level singleton — multi-worker ile uyumsuz | `drg/mcp_server.py:54–55` |
| `ClusterResult` `drg.__all__`'da yok ama `drg.clustering.__all__`'da var — tutarsız yüzey | `drg/__init__.py` vs `drg/clustering/__init__.py` |

---

## 6. Tests

| Modül | Coverage Durumu |
|:---|:---|
| `drg/reasoning/` | ❌ Modül yok, test yok |
| `drg/evaluation/` | ❌ Modül yok, test yok |
| `drg/optimizer/` | ❌ Modül yok, test yok |
| `drg/utils/env_loader.py` | ❌ Dedicated test yok |
| `drg/utils/llm_throttle.py` | ❌ Test yok (coverage'dan da exclude edilmiş) |
| `drg/graph/schema_generator.py` | ❌ Dosya yok, test yok |
| `drg/embedding/providers.py` | 🟡 Cache test edilmiş; provider'lar yalnızca live API key ile test edilebilir |
| `drg/api/server.py` | 🟡 `test_api_server.py` mevcut; `[api]` extra gerektirir |
| `drg/graph/builders.py` | 🟡 Integration testleri var; `extract_evidence_snippet` edge case'leri test edilmemiş |
| `drg/confidence/_default.py` | 🟡 Kısmi; None schema, nested alan path'leri test edilmemiş |
| Diğer modüller | ✅ Yeterli unit test coverage'ı |

**Spesifik test eksiklikleri:**
- `EnhancedKG.add_edge()` için node'lar mevcut değilken negatif test yok
- `test_integration.py` dspy stublandığı için `extract_triples` sessizce boş liste döner (exception bekleniyorsa)
- `drg eval list` veya `drg eval run` CLI path'lerini test eden test yok (her ikisi de broken)
- `validate_graph_data` için property-based / fuzz test yok

---

## 7. Documentation

| Bulgu | Kanıt |
|:---|:---|
| `docs/evaluation_framework.md` var olmayan tam evaluation API'sini açıklıyor | `drg/evaluation/` missing |
| `docs/multi_document_reasoning.md` `drg/reasoning/`'ı "source of truth" olarak referans ediyor | `drg/reasoning/` missing |
| README `examples/optimizer_demo.py`, `examples/multi_document_reasoning_example.py`, `examples/evaluation_framework_example.py` listeliyor | Dosyalar yok |
| README `drg eval run`'ı çalışan bir CLI komutu olarak gösteriyor | Runtime'da crash |
| `SOURCES.txt` tüm eksik modülleri hâlâ listeliyor — son `pip install -e .`'dan sonra silindiklerine işaret ediyor | `drg_kg.egg-info/SOURCES.txt:49–122` |

---

## 8. Examples

| Örnek | Durum | Sorun |
|:---|:---|:---|
| `examples/full_pipeline_example.py` | ⚠️ Live API key gerektirir | `networkx`'i direkt import ediyor |
| `examples/quickstarts/01_wikipedia_article.py` | ⚠️ Deprecated `KG` kullanıyor | Her instantiation'da `DeprecationWarning` fırlatır |
| `examples/quickstarts/02_financial_news.py` | ✅ | — |
| `examples/quickstarts/03_biomedical.py` | ✅ | — |
| `examples/api_server_example.py` | ✅ | — |
| `examples/incremental_update_example.py` | ✅ | — |
| `examples/temporal_query_example.py` | ✅ | — |
| `examples/query_layer_example.py` | ✅ | Docstring'de var olmayan `multi_document_reasoning_example.py`'ye referans |
| `examples/event_extraction_example.py` | ✅ | Live API key gerektirir |
| `examples/mcp_demo.py` | ✅ | — |
| `examples/multi_document_reasoning_example.py` | 🔴 **EKSİK** | README'de, docs'ta ve `query_layer_example.py` docstring'inde referans |
| `examples/evaluation_framework_example.py` | 🔴 **EKSİK** | README'de |
| `examples/optimizer_demo.py` | 🔴 **EKSİK** | README'de |

---

## 9. Production Readiness

| Kategori | Değerlendirme |
|:---|:---|
| **Logging** | ✅ `drg/utils/logging.py` — structured JSON logging, `with_context` adapter; modüllerin çoğu `get_logger(__name__)` kullanıyor |
| **Input validation** | ✅ API'de Pydantic modelleri, extract'ta uzunluk guard'ları, KG dosyaları için `validate_graph_data` |
| **Exception handling** | 🟡 `drg/errors.py`'de custom hiyerarşi iyi; birkaç `except Exception: pass` site onu baltalıyor |
| **Configuration** | 🟡 Environment variable driven; `config.py` model normalizasyonu ile LM config'ini karıştırıyor (SRP ihlali) |
| **Serialization** | ✅ `to_json`/`from_dict` round-trip'leri tutarlı |
| **Performance** | 🟡 `GeminiEmbeddingProvider.embed_batch` tek tek döngü yapıyor — O(n) API call |
| **Scalability** | 🔴 API server'da global `os.environ` mutasyonu — multi-worker deployment'a uyumsuz |
| **Memory** | 🟡 `KGNode.embedding: list[float]` — büyük graph'larda ciddi bellek tüketimi |
| **Concurrency** | 🔴 `_apply_extraction_env()` race condition; `LMConfig._configured` process-global class-level flag |
| **API stability** | 🟡 Alpha olarak belgelenmiş; coverage exclusion'ları gerçek coverage'ı gizliyor |

---

## 10. Release Readiness

| Öğe | Durum | Kanıt |
|:---|:---|:---|
| Packaging | ✅ `pyproject.toml` iyi yapılandırılmış; `setuptools-scm` ile git tag versiyonlama | |
| Versioning | ✅ Semantic versioning; `_version.py` build'de üretilir; `__init__.py`'de fallback | |
| CI | 🔴 Broken — `checkout@v6`, `setup-python@v6`, `upload-artifact@v7`, `download-artifact@v8` yok | `.github/workflows/` |
| Coverage floor | 🟡 `fail_under = 70` — düşük; stale omit entries görünür coverage'ı şişiriyor | `pyproject.toml:247` |
| Changelog | ✅ Bakımlı, mevcut olanlar için doğru | `CHANGELOG.md` |
| PyPI token setup | ✅ Release workflow gerekli secret'ları belgeliyor | `release.yml` |
| README | 🟡 Var olmayan feature'ları (`eval`, `--infer`, optimizer, reasoning) açıklıyor | |
| Migration policy | 🟡 Alpha uyarısı var; `KG` deprecation warning yerinde; `DEPRECATED.md` yok | |
| `py.typed` marker | ✅ Mevcut | `pyproject.toml:200` |
| Missing modules | 🔴 `drg/reasoning/`, `drg/evaluation/`, `drg/optimizer/` yok | |

---

## Top 20 Kritik Problem (Önem Sırasına Göre)

---

### Problem 1 — P0: `drg eval` CLI komutu import'ta crash ediyor

**Neden problem:** `drg/cli.py:550` — `from .evaluation import (BenchmarkRunner, PipelinePrediction, ...)`
`_handle_eval()` içinde. `drg/evaluation/` disk üzerinde yok.

**Dosya:** `drg/cli.py:550`, `drg/evaluation/` (eksik)

**Kanıt:** `drg_kg.egg-info/SOURCES.txt:49–56` silinmiş modülleri listeliyor; filesystem'de hiçbir
`drg/evaluation/*.py` dosyası yok.

**Çözüm:** `drg/evaluation/` package'ını git geçmişinden geri yükle veya `BenchmarkRunner`,
`compare_reports`, `load_benchmark_datasets`, `save_json_report`, `render_markdown_report`,
`load_evaluation_report`, `load_prediction_artifact` ve diğer bağımlılıkları yeniden implement et.

---

### Problem 2 — P0: `drg extract --infer` import'ta crash ediyor

**Neden problem:** `drg/cli.py:506` — `from .reasoning import MultiDocumentReasoner, ReasoningConfig`.
`drg/reasoning/` disk üzerinde yok.

**Dosya:** `drg/cli.py:506`, `drg/reasoning/` (eksik)

**Kanıt:** `SOURCES.txt:118–122` `drg/reasoning/__init__.py`, `_engine.py`, `_explain.py`,
`_rules.py`, `_types.py`'yi listeliyor — tümü filesystem'de yok.

**Çözüm:** `drg/reasoning/` package'ını geri yükle. `MultiDocumentReasoner` `docs/multi_document_reasoning.md`'de belgelenmiş.

---

### Problem 3 — P0: CI workflow'ları var olmayan GitHub Actions versiyonlarına referans veriyor

**Neden problem:** `actions/checkout@v6`, `actions/setup-python@v6`, `actions/upload-artifact@v7`,
`actions/download-artifact@v8` yok. Her push/PR'da CI tamamen başarısız olur.

**Dosya:** `.github/workflows/ci.yml:20,22,87`, `.github/workflows/release.yml:63`

**Kanıt:** Mevcut stable versiyonlar: `checkout@v4`, `setup-python@v5`, `upload-artifact@v4`,
`download-artifact@v4`.

**Çözüm:** Doğru major versiyonlara düşür.

---

### Problem 4 — P0: API server `os.environ` mutasyonu thread-safe değil

**Neden problem:** `_apply_extraction_env()` async FastAPI route handler içinden `os.environ`'a
(process-global) yazıyor. Eş zamanlı `/api/extract` istekleri `DRG_MODEL`, `OPENAI_API_KEY`
vb. üzerinde race condition oluşturur. Bir kullanıcının API key'i başka bir kullanıcının
extraction context'ine sızabilir.

**Dosya:** `drg/api/server.py:270–310`

**Kanıt:**
```python
os.environ["DRG_MODEL"] = request.model      # ← async handler içinden
os.environ["OPENAI_API_KEY"] = request.api_key  # ← race condition
```

**Çözüm:** Credentials'ı `os.environ` yerine doğrudan extraction fonksiyonuna argüman olarak ilet.
Per-request LM scoping için `dspy.context(lm=...)` kullan.

---

### Problem 5 — P1: `allow_credentials=True` + `allow_origins=["*"]` geçersiz CORS

**Neden problem:** Tarayıcılar Fetch spec §3.2 gereği wildcard origins'e credentialed CORS'u reddeder.
Production'da deploy edildiğinde browser-based client'lar için CORS hataları oluşur.

**Dosya:** `drg/api/server.py:617–624`

**Kanıt:**
```python
CORSMiddleware(allow_origins=["*"], allow_credentials=True, ...)
```

**Çözüm:** Açık deployment'lar için `allow_credentials=False` yap, ya da credentials
etkinleştirildiğinde `DRG_CORS_ORIGINS`'in açıkça set edilmesini zorunlu kıl.

---

### Problem 6 — P1: `drg/optimizer/` modülü eksik

**Neden problem:** README örnek galerisi `examples/optimizer_demo.py`'yi listeliyor ve
`docs/optimizer_design.md` DSPy optimizer pipeline'ını açıklıyor. Ne modül ne de örnek var.

**Dosya:** `drg/optimizer/` (eksik), `examples/optimizer_demo.py` (eksik)

**Kanıt:** `SOURCES.txt:102–104`, `pyproject.toml:230–231` `drg/optimizer/optimizer.py`,
`drg/optimizer/metrics.py`'yi referans ediyor.

**Çözüm:** Optimizer modülünü geri yükle veya README ve docs'taki tüm referansları kaldır.

---

### Problem 7 — P1: README örnek galerisinde üç dosya eksik

**Neden problem:** `examples/multi_document_reasoning_example.py`, `examples/evaluation_framework_example.py`,
`examples/optimizer_demo.py` README tablosunda listeleniyor ama disk üzerinde yok.

**Dosya:** `README.md:272,277,281`, `README.tr.md:278,282,285`

**Çözüm:** Örnekleri oluştur veya backing modüller geri yüklenene kadar README'den kaldır.

---

### Problem 8 — P1: `pyproject.toml` coverage omit listesi stale dosya path'leri içeriyor

**Neden problem:** Stale `omit` girişleri doğru coverage ölçümünü engeller. Eksik modüller
geri yüklendiğinde 70% floor gerçeği yansıtmayabilir.

**Dosya:** `pyproject.toml:225–240`

**Kanıt:** `"drg/mcp_api.py"`, `"drg/optimizer/optimizer.py"`, `"drg/optimizer/metrics.py"`,
`"drg/graph/schema_generator.py"` — hiçbiri yok. `"drg/graph/visualization_adapter.py"`
bir package directory, glob eşleşmez.

**Çözüm:** Stale girişleri kaldır; package için `drg/graph/visualization_adapter/**` kullan.

---

### Problem 9 — P2: `ClusterResult`/`Cluster` isim çakışması

**Neden problem:** `drg.clustering.Cluster` (ClusterResult alias: `int cluster_id`, `list nodes`)
ve `drg.graph.kg_core.Cluster` (`str id`, `set node_ids`) tamamen farklı, uyumsuz sınıflar.
`from drg.clustering import Cluster` ile `from drg.graph import Cluster` farklı şeyler döner.

**Dosya:** `drg/clustering/algorithms.py:58`, `drg/graph/kg_core.py:271`

**Çözüm:** Clustering sonuç tipini `CommunityResult` veya `AlgorithmCluster` olarak yeniden adlandır.

---

### Problem 10 — P2: `config.py`'de `dspy` try/except guard'sız import ediliyor

**Neden problem:** `dspy` opsiyonel bir bağımlılık. `drg/config.py:13`'te `import dspy`
module level'da, guard yok. `[extract]` extra'sı kurulu olmadan `drg.config`'e doğrudan
erişim `ModuleNotFoundError` fırlatır.

**Dosya:** `drg/config.py:13`

**Çözüm:**
```python
try:
    import dspy
except ImportError as e:
    raise ImportError("dspy gerekli. `pip install drg-kg[extract]` ile kur.") from e
```

---

### Problem 11 — P2: `GeminiEmbeddingProvider.embed_batch` O(n) API call

**Neden problem:** Method, Gemini batch embedding endpoint'ini kullanmak yerine her metin için
ayrı API call yapıyor. N metin = N API isteği = N kat gecikme.

**Dosya:** `drg/embedding/providers.py:196–205`

**Kanıt:**
```python
for text in texts:
    result = self.genai_module.embed_content(...)  # her metin için bir network call
    embeddings.append(result["embedding"])
```

**Çözüm:** `genai.embed_content(model=..., content=texts)` veya `batch_embed_contents` API'sini kullan.

---

### Problem 12 — P2: MCP server module-level singleton — multi-worker ile uyumsuz

**Neden problem:** `_schemas` ve `_knowledge_graphs` module-level dict'ler. Multi-worker
deployments'ta (`uvicorn --workers 4`) her worker kendi memory'sine sahip; bir worker'da
kaydedilen schema diğerlerine görünmez.

**Dosya:** `drg/mcp_server.py:54–55`

**Kanıt:**
```python
_schemas: dict[str, DRGSchema | EnhancedDRGSchema] = {}
_knowledge_graphs: dict[str, EnhancedKG] = {}
```

**Çözüm:** Single-worker kısıtını belgele, ya da shared backend (Redis, SQLite) enjekte et.

---

### Problem 13 — P2: İlk quickstart örneği deprecated `KG` sınıfını kullanıyor

**Neden problem:** Yeni bir kullanıcının çalıştıracağı ilk şey olan quickstart,
her instantiation'da `DeprecationWarning` fırlatan `KG`'yi kullanıyor.

**Dosya:** `examples/quickstarts/01_wikipedia_article.py:43`

**Kanıt:**
```python
from drg import KG, ...  # KG.__init__ DeprecationWarning fırlatır
```

**Çözüm:** Quickstart'ı `build_enhanced_kg` ve `EnhancedKG` kullanacak şekilde güncelle.

---

### Problem 14 — P3: `_handle_extract` SRP ihlali — 200 satırlık monolit

**Neden problem:** Tek bir fonksiyon: env var mutasyonu, schema yükleme, schema üretimi,
chunking config, extraction (2 farklı path), event extraction, KG build, incremental merge,
snapshot oluşturma, reasoning çağrısı, output yazma, secret redaction'ı üstleniyor.

**Dosya:** `drg/cli.py:370–545`

**Çözüm:** `_build_schema`, `_run_extraction`, `_apply_incremental_update`, `_write_output`
olarak ayrı fonksiyonlara böl.

---

### Problem 15 — P3: `builders.py`'de sessiz `except Exception: pass`

**Neden problem:** Evidence snippet extraction başarısızlığı hiçbir diagnostic olmadan
`None` döner. Regex veya metin erişim mantığında bug varsa "no evidence found" olarak görünür.

**Dosya:** `drg/graph/builders.py:302–306`

**Kanıt:**
```python
except Exception:
    pass  # log yok, metrik yok, görünmez başarısızlık
```

**Çözüm:** En azından `logger.debug("Evidence extraction failed", exc_info=True)` ekle.

---

### Problem 16 — P3: `create_app` 800+ satırlık god function

**Neden problem:** Tüm 20+ FastAPI route handler `create_app()` içinde nested closure olarak
tanımlanmış. `app`, `kg`, `neo4j_config` vb.'yi closure üzerinden yakalıyorlar. Tek bir
route'u unit test etmek tüm factory'yi ayağa kaldırmayı gerektiriyor.

**Dosya:** `drg/api/server.py:580–1050`

**Çözüm:** Route handler'ları `request: Request` alan module-level fonksiyonlar olarak tanımla.
`request.app.state`'den state'i oku. `create_app`'te app'e register et.

---

### Problem 17 — P3: `drg/__init__.py.__getattr__` 60+ girişli sabit kodlanmış dict

**Neden problem:** Lazy import mapping 60+ sembol→modül girişini sabit kodluyor. Her yeni
public export iki yerde (`__all__` ve `lazy_imports`) manual ekleme gerektiriyor. `__all__`'da
olup `lazy_imports`'ta olmayan semboller mümkün.

**Dosya:** `drg/__init__.py:100–230`

**Çözüm:** Package-level `__init__.py` deklarasyonlarıyla `__all__`-driven `importlib.import_module`
kullan, ya da başlangıç maliyetini kabul ederek eager import'a geç.

---

### Problem 18 — P3: `RelationGroup.description` required ama sık sık boş geçiliyor

**Neden problem:** `RelationGroup` `description: str` gerektiriyor (default yok), ancak
`mcp_server.py` gibi caller'lar `description=rg.get("description", "")` ile boş string geçiyor.
`__post_init__` boş description'ı validate etmiyor.

**Dosya:** `drg/schema.py:83–97`, `drg/mcp_server.py:96`

**Çözüm:** `description: str = ""` default ekle ya da `__post_init__`'te boş string'i reddet.

---

### Problem 19 — P4: `rollback_to_version` rollback sonrası snapshot kayıt etmiyor

**Neden problem:** Önceki bir versiyona rollback yapıldığında manifest'e yeni bir
"rollback" snapshot eklenmiyor. Sonraki snapshot'lar yanlış parent ID hesaplar ve
version history'de boşluk oluşur.

**Dosya:** `drg/graph/versioning.py:rollback_to_version`

**Çözüm:** Dosya restore edildikten sonra `operation="rollback"` ile `create_snapshot` çağır.

---

### Problem 20 — P4: `env_loader.py` ve `llm_throttle.py` için sıfır test

**Neden problem:** Bu utility'ler her extraction call'da (`load_dotenv`) ve her LLM
çağrısında (`throttle_llm_calls`) kullanılıyor. Happy path için bile smoke test yok.

**Dosya:** `test_env_loader.py` yok, `test_llm_throttle.py` yok

**Çözüm:** Geçici dosyayla `load_dotenv` testi, `DRG_LLM_MIN_INTERVAL_SECONDS=0` (no-op)
ve `DRG_LLM_MIN_INTERVAL_SECONDS=0.01` (sleep path) için basic unit test ekle.

---

## Puanlar

| Boyut | Puan | Gerekçe |
|:---|:---|:---|
| **Architecture** | 6/10 | Temiz package yapısı, iyi sorumluluk ayrımı, sağlam provenance ve temporal tasarım. CLI/API'deki god function'lar, global `os.environ` mutasyonu ve singleton MCP state puan düşürüyor. |
| **Code Quality** | 5/10 | Schema ve graph core güçlü; sessiz exception swallowing, stale coverage config, isim çakışması ve 1000 satırlık extract modülü zayıflatıyor. |
| **Documentation** | 5/10 | README kapsamlı ve iyi yazılmış ancak üç eksik modülü (`reasoning`, `evaluation`, `optimizer`) ve beş var olmayan örnek dosyayı aktif olarak reklamını yaparak kullanıcıları yanıltıyor. |
| **Tests** | 6/10 | Test sayısı yüksek (50+ dosya), fixture'lar iyi tasarlanmış, mocking stratejisi sağlam. README'nin reklamını yaptığı iki tam subsystem için sıfır coverage. |
| **Production Readiness** | 4/10 | API server'da thread-safety bug, CORS yanlış yapılandırma, extraction endpoint'inde rate limiting yok, `os.environ` race'leri. API server olmadan core pipeline daha sağlam. |
| **Release Readiness** | 3/10 | CI broken (yanlış Actions versiyonları), üç core modül eksik, README var olmayan bir ürünü kısmen açıklıyor. 70% coverage floor PyPI release için çok düşük. |
| **Overall** | 5/10 | İyi fikirlere ve sağlam core implementasyona sahip, iyi tasarlanmış bir KG lifecycle framework. Eksik modüllerin geri yüklenmesi, CI'ın düzeltilmesi ve thread-safety sorunlarının giderilmesi herhangi bir release öncesinde zorunlu. |
