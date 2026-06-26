# CODE_REVIEW.md — Bağımsız Doğrulama Raporu

> **Yöntem:** Her bulgu, önceki rapora başvurulmaksızın, yalnızca dosya sistemindeki  
> gerçek kaynak koduna dayanarak doğrulandı.  
> **Kararlar:** DOĞRU / KISMEN DOĞRU / YANLIŞ / KANIT YETERSİZ  
> **Tarih:** 2026-06-25

---

## Karar Anahtarı

| Karar | Anlamı |
|:---|:---|
| ✅ DOĞRU | Orijinal iddia, kod kanıtıyla tam örtüşüyor |
| 🟡 KISMEN DOĞRU | Ana teşhis doğru ama ayrıntılar (satır no., şiddet, kod snippet'i) yanıltıcı/hatalı |
| ❌ YANLIŞ | Orijinal iddia koda uymuyor; kanıt zıt yönü gösteriyor |

---

## Problem 1 — `drg eval` import'ta crash

**Karar: ✅ DOĞRU**

`drg/cli.py:549`'da tanımlanan `_handle_eval()` fonksiyonunun ilk satırı
`from .evaluation import (BenchmarkRunner, PipelinePrediction, ...)`.
Bu fonksiyon her `drg eval` alt komutunda çağrılır.

`drg/evaluation/` dizini dosya sisteminde bulunmuyor:

```
Get-ChildItem drg/ -Directory → 13 dizin; "evaluation" yok
```

`drg_kg.egg-info/SOURCES.txt` modülün var olduğunu belgeleyen 7 dosyayı hâlâ listeliyor.

> **Küçük düzeltme:** `_handle_eval` satır 549'da başlar; rapordaki "satır 550" tek satır yanındadır.

---

## Problem 2 — `drg extract --infer` import'ta crash

**Karar: ✅ DOĞRU**

`drg/cli.py:506`'da `if args.infer:` bloğu içinde
`from .reasoning import MultiDocumentReasoner, ReasoningConfig`.
`drg/reasoning/` dizini dosya sisteminde yok (doğrulandı).
`SOURCES.txt` `__init__.py`, `_engine.py`, `_explain.py`, `_rules.py`, `_types.py`'yi
listeliyor; hiçbiri fiziksel olarak mevcut değil. `--infer` flag'i geçildiğinde
`ModuleNotFoundError` kaçınılmazdır.

---

## Problem 3 — CI workflow'ları geçersiz GitHub Actions versiyonlarına işaret ediyor

**Karar: ✅ DOĞRU — ancak rapor bir versiyonu atladı**

`.github/workflows/ci.yml` ve `release.yml` doğrudan okunarak teyit edilen sürüm referansları:

| Dosya | Satır | Kullanılan | Mevcut stable |
|:---|:---|:---|:---|
| `ci.yml` | 19, 40, 65, 113 | `checkout@v6` | `v4` |
| `ci.yml` | 22, 43, 68, 117 | `setup-python@v6` | `v5` |
| `ci.yml` | 87 | `upload-artifact@v7` | `v4` |
| `release.yml` | 42, 50 | `checkout@v6` | `v4` |
| `release.yml` | 71 | `upload-artifact@v7` | `v4` |
| `release.yml` | 87 | `download-artifact@v8` | `v4` |

GitHub Actions bu versiyonları çözümleyemez; her push/PR'da CI tamamen başarısız olur.

> **Raporun atladığı sorun:** `ci.yml:93`'te `codecov/codecov-action@v6` da kullanılıyor.  
> Codecov action'ın güncel stable versiyonu v4'tür; v6 mevcut değil.  
> Orijinal rapor bu beşinci hatalı versiyonu **atladı**.

---

## Problem 4 — `os.environ` mutasyonu thread-safe değil

**Karar: ✅ DOĞRU**

`drg/api/server.py:240–273` — `_apply_extraction_env()` kilit olmaksızın
`os.environ["DRG_MODEL"]`, `os.environ["OPENAI_API_KEY"]` vb.'ye yazıyor.
Fonksiyon, async route handler içinden line 489 ve 555'te çağrılıyor.

```python
previous_env = _apply_extraction_env(request)   # os.environ mutate
try:
    ...
    await asyncio.to_thread(extract_typed, ...)  # ← burada yield; başka coroutine env'yi ezer
finally:
    _restore_env(previous_env)
```

`asyncio.to_thread()` await edilirken event loop başka coroutine'e kontrolü
devredebilir. İki eşzamanlı isteğin `os.environ` üzerinde race'i somut; API key
sızma senaryosu gerçek.

---

## Problem 5 — CORS yanlış yapılandırılmış

**Karar: ✅ DOĞRU**

`drg/api/server.py:344–350` doğrudan okundu:

```python
_cors_origins_env = os.getenv("DRG_CORS_ORIGINS", "*").strip()
cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,    # default ["*"]
    allow_credentials=True,        # ← Fetch spec §3.2 bu kombinasyonu reddeder
    allow_methods=["*"],
    allow_headers=["*"],
)
```

`DRG_CORS_ORIGINS` set edilmediğinde `allow_origins=["*"]` + `allow_credentials=True`
kombinasyonu tarayıcılar tarafından reddedilir; tüm browser tabanlı istemciler CORS hatası alır.

---

## Problem 6 — `drg/optimizer/` eksik

**Karar: ✅ DOĞRU**

`drg/optimizer/` dizini dosya sisteminde yok.
`pyproject.toml` omit listesi `optimizer.py` ve `metrics.py`'yi referans ediyor;
`SOURCES.txt:102–104` paketi listeliyor. Hiçbiri fiziksel olarak mevcut değil.

---

## Problem 7 — README'de üç örnek dosyası eksik

**Karar: ✅ DOĞRU**

`examples/` dizininin doğrudan listesinde bu üç dosya yok:

- `examples/multi_document_reasoning_example.py`
- `examples/evaluation_framework_example.py`
- `examples/optimizer_demo.py`

Ayrıca `examples/query_layer_example.py:5` `multi_document_reasoning_example.py`'ye
doğrudan referans veriyor — iç çapraz referans da kırık.

---

## Problem 8 — `pyproject.toml` coverage omit listesi stale

**Karar: ✅ DOĞRU**

`pyproject.toml:215–235` omit listesi bağımsız olarak doğrulandı:

| Omit girdisi | Gerçek durum |
|:---|:---|
| `"drg/mcp_api.py"` | ❌ Dosya yok (`Test-Path` → False) |
| `"drg/graph/visualization_adapter.py"` | ❌ Bu bir **dizin** (`PSIsContainer: True`); coverage.py dosya glob'u dizine eşleşmez — `visualization_adapter/` içi omit edilMEZ |
| `"drg/optimizer/optimizer.py"` | ❌ Dosya yok |
| `"drg/optimizer/metrics.py"` | ❌ Dosya yok |
| `"drg/graph/schema_generator.py"` | ❌ Dosya yok |

`visualization_adapter` paketi için doğru pattern `"drg/graph/visualization_adapter/**"` olmalı.

---

## Problem 9 — `ClusterResult`/`Cluster` isim çakışması

**Karar: 🟡 KISMEN DOĞRU**

İki ayrı sınıf gerçekten mevcut:

- `drg.clustering.algorithms.ClusterResult` → `int cluster_id`, `list nodes`, `list edges`
- `drg.graph.kg_core.Cluster` → `str id`, `set node_ids`, `dict metadata`
- `algorithms.py:58`: `Cluster = ClusterResult` alias'ı var

**Ancak** bu tasarım *kasıtlı ve belgelenmiş*:

```python
# algorithms.py:56–60
# Backward-compatibility alias.
# New code should use ClusterResult; the alias keeps existing imports working.
# To store a community in EnhancedKG use ClusterResult.to_kg_cluster() which
# returns a drg.graph.kg_core.Cluster (a separate, storage-oriented class).
Cluster = ClusterResult
```

`ClusterResult.to_kg_cluster()` dönüşüm metodu da mevcut. Orijinal raporun
"uyumsuz, sorunlu çakışma" olarak nitelendirmesi gerçeği yansıtmıyor; isimlendirme
tartışmaya açık olsa da tasarım bilerek yapılmış ve belgelenmiş.

---

## Problem 10 — `config.py`'de guard'sız `import dspy`

**Karar: 🟡 KISMEN DOĞRU**

`drg/config.py:13`'te `import dspy` module-level, try/except guard yok — **doğru**.

**Ancak:** Tüm kod tabanında `config.py`'nin import edildiği tek yer
`drg/extract/__init__.py:631`'dir ve bu import bir fonksiyonun içindedir (lazy).
`drg/__init__.py` `config`'e hiç dokunmuyor.

`import drg` çağrısı `config.py`'yi hiçbir zaman yüklemez. Risk yalnızca
`import drg.config` ile doğrudan erişimde gerçekleşir — bu, `dspy` kurulu
olmadan olağan kullanımda gerçekleşebilir bir senaryo değil.

Teşhis teknik olarak doğru; pratik şiddet orijinal rapordakinden önemli ölçüde düşük.

---

## Problem 11 — `GeminiEmbeddingProvider.embed_batch` O(n) API call

**Karar: ✅ DOĞRU**

`drg/embedding/providers.py:175–184` doğrudan okundu:

```python
def embed_batch(self, texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        result = self.genai_module.embed_content(
            model=self.model_name, content=text, task_type="semantic_similarity"
        )
        embeddings.append(result["embedding"])
    return embeddings
```

N metin → N API çağrısı. Gemini API'nin `batch_embed_contents` endpoint'i kullanılmıyor.

---

## Problem 12 — MCP server module-level singleton multi-worker ile uyumsuz

**Karar: ✅ DOĞRU**

`drg/mcp_server.py:54–55`:

```python
_schemas: dict[str, DRGSchema | EnhancedDRGSchema] = {}
_knowledge_graphs: dict[str, EnhancedKG] = {}
```

Modül düzeyinde process-local dict'ler. Çok-worker deployment'ta process izolasyonu
nedeniyle state paylaşılmaz.

---

## Problem 13 — İlk quickstart deprecated `KG` kullanıyor

**Karar: ✅ DOĞRU**

`examples/quickstarts/01_wikipedia_article.py:41`: `from drg import KG`.
`KG.from_typed()` → `cls()` → `KG.__init__()` → `warnings.warn(..., DeprecationWarning, ...)`.

Doğrulama zinciri `drg/graph/_legacy.py:39–43`'te kapandı. Her çalıştırmada
`DeprecationWarning` fırlar.

---

## Problem 14 — `_handle_extract` monolit

**Karar: 🟡 KISMEN DOĞRU**

Fonksiyon `cli.py:329`'da başlar, `548`'de biter → **220 satır**.

Orijinal raporun verdiği satır aralığı `370–545` yanlış; fonksiyon 370'te değil
**329'da** başlıyor. Satır sayısı "~200" olarak ifade edilmiş; gerçek 220 bununla
uyumlu. God function teşhisi doğru; belirtilen aralık hatalı.

---

## Problem 15 — `builders.py`'de sessiz `except Exception: pass`

**Karar: ❌ YANLIŞ — kod snippet'i gerçeği yansıtmıyor**

Orijinal rapor `drg/graph/builders.py:302–306`'da şunu gösteriyor:

```python
except Exception:
    pass  # log yok, metrik yok, görünmez başarısızlık
```

Gerçek kod:

```python
try:
    evidence_max_chars = int(os.getenv("DRG_EVIDENCE_MAX_CHARS", "240"))
except Exception:
    evidence_max_chars = 240       # ← "pass" değil; sağlıklı fallback

try:
    evidence_max_pair_distance = int(os.getenv("DRG_EVIDENCE_MAX_PAIR_DISTANCE", "2500"))
except Exception:
    evidence_max_pair_distance = 2500   # ← yine sağlıklı fallback
```

Bu bloklar env var tamsayı dönüşümü için makul fallback içeriyor.
"Evidence extraction sessizce başarısız oluyor" iddiası bu satırlar için **yanlış**.

> **Not:** `drg/confidence/_default.py:101`'de gerçek bir `except Exception: return False`
> var — bu satır log üretmeden sessizce `False` döner ve meşru biçimde eleştirilebilir.
> Ancak bu da `pass` değil; şiddet rapordakinden düşük.

---

## Problem 16 — `create_app` 800+ satırlık God function

**Karar: 🟡 KISMEN DOĞRU**

`create_app` fonksiyonu `server.py:309`'da başlar, `1022`'de biter → **714 satır**.

- Orijinal raporun verdiği satır aralığı `"580–1050"` → **yanlış** (başlangıç 309, bitiş 1022)
- "800+ satır" ifadesi → **yanlış** (714 satır, yaklaşık %12 şişirilmiş)

God function teşhisi doğru: 20+ route handler nested closure. Mimari sorun gerçek;
sayısal iddialar tutarsız.

---

## Problem 17 — `__getattr__` 60+ girişli sabit kodlanmış dict

**Karar: ✅ DOĞRU**

`drg/__init__.py` `lazy_imports` dict sayımı: confidence (3) + extraction (8) +
graph (~26) + chunking (6) + embedding (6) + clustering (7) + events (12) + query (15)
→ **~83 giriş**. "60+" iddiası muhafazakâr ama doğru.

---

## Problem 18 — `RelationGroup.description` boş geçilebiliyor

**Karar: ✅ DOĞRU**

`drg/schema.py:86`: `description: str` — default yok.
`drg/schema.py:97–100`: `__post_init__` yalnızca `name` ve `relations`'ı doğruluyor.
`drg/mcp_server.py:83`: `description=rg.get("description", "")` — boş string geçilebilir.

Boş description validation'dan geçer; downstream tüketiciler boş alan alır.

---

## Problem 19 — `rollback_to_version` snapshot kayıt etmiyor

**Karar: ❌ YANLIŞ — yanlış teşhis**

Orijinal iddia: "manifest'e yeni bir 'rollback' snapshot eklenmiyor; sonraki
snapshot'lar yanlış parent ID hesaplar."

`drg/graph/versioning.py:204–228` gerçek kodu:

```python
rollback = GraphVersion(
    version_id=f"rollback-{version_id}-{len(manifest.versions) + 1}",
    operation="rollback",
    ...
)
manifest.versions.append(rollback)   # ← manifest'e EKLENIYOR
manifest.save(graph_path, versions_dir)
```

Manifest'e yeni bir `GraphVersion` girdisi **ekleniyor**. Parent ID zinciri de
doğru çalışır; rollback kaydı manifest'tedir.

**Gerçek sorun** (raporda belirtilmemiş): `snapshot_path=target.snapshot_path` —
rollback girdisi kendi yeni snapshot dosyasını oluşturmuyor, eski snapshot'ın
yolunu reuse ediyor. Rollback öncesi state (C→A senaryosunda C state'i) kayıt
altına alınmıyor. Bu farklı bir bug.

---

## Problem 20 — `env_loader.py` ve `llm_throttle.py` test yok

**Karar: ✅ DOĞRU**

Her iki dosya da `pyproject.toml` omit listesinde. `tests/` dizininde bu modüllere
özgü test dosyası yok. Her extraction çağrısında kullanılıyorlar ama smoke test bile
mevcut değil.

---

## Özet Tablosu

| # | Problem | Karar | Temel Gerekçe |
|:---|:---|:---|:---|
| 1 | `drg eval` crash | ✅ DOĞRU | `drg/evaluation/` yok; her alt komutta crash |
| 2 | `drg extract --infer` crash | ✅ DOĞRU | `drg/reasoning/` yok |
| 3 | CI version hataları | ✅ DOĞRU | 4 hatalı versiyon; + 1 atlandı (`codecov@v6`) |
| 4 | `os.environ` race condition | ✅ DOĞRU | async + `asyncio.to_thread` + kilit yok |
| 5 | CORS yanlış yapılandırma | ✅ DOĞRU | `allow_credentials=True` + `["*"]` |
| 6 | `optimizer/` eksik | ✅ DOĞRU | Dizin yok |
| 7 | 3 örnek dosyası eksik | ✅ DOĞRU | `examples/` listesi doğrulandı |
| 8 | Stale coverage omit | ✅ DOĞRU | 5 stale giriş; `visualization_adapter` nüansı eklendi |
| 9 | Cluster isim çakışması | 🟡 KISMEN DOĞRU | Kasıtlı + belgelenmiş; şiddet abartılmış |
| 10 | `config.py` guard'sız import | 🟡 KISMEN DOĞRU | Teknik doğru; pratik risk çok düşük |
| 11 | Gemini O(n) API call | ✅ DOĞRU | Loop tek tek çağrı yapıyor |
| 12 | MCP singleton | ✅ DOĞRU | Module-level dict; multi-worker izolasyonu |
| 13 | Quickstart deprecated KG | ✅ DOĞRU | `KG()` her instantiation'da DeprecationWarning |
| 14 | `_handle_extract` monolit | 🟡 KISMEN DOĞRU | 220 satır, 329–548; rapor 370–545 veriyor |
| 15 | `builders.py` sessiz `pass` | ❌ YANLIŞ | Gerçek kod fallback değeri atıyor, `pass` değil |
| 16 | `create_app` 800+ satır | 🟡 KISMEN DOĞRU | 714 satır (309–1022), "800+" ve "580–1050" hatalı |
| 17 | `__getattr__` 60+ giriş | ✅ DOĞRU | Gerçekte ~83 giriş |
| 18 | `RelationGroup.description` | ✅ DOĞRU | Empty string doğrulamadan geçiyor |
| 19 | `rollback_to_version` | ❌ YANLIŞ | Manifest'e ekleniyor; gerçek bug farklı |
| 20 | `env_loader`/`throttle` test yok | ✅ DOĞRU | Test dosyası ve coverage kaydı yok |

---

## İstatistik

| Karar | Sayı | Oran |
|:---|:---|:---|
| ✅ DOĞRU | 14 | %70 |
| 🟡 KISMEN DOĞRU | 4 | %20 |
| ❌ YANLIŞ | 2 | %10 |
| KANIT YETERSİZ | 0 | — |

---

## Raporun Atladığı Gerçek Sorunlar

Doğrulama sürecinde orijinal CODE_REVIEW.md'de bulunmayan ancak doğrudan koddan
tespit edilen sorunlar:

1. **`codecov/codecov-action@v6`** (`ci.yml:93`) — var olmayan versiyon; Problem 3'te
   sayılmayan beşinci hatalı CI referansı.

2. **`rollback_to_version`'da snapshot path reuse** — Rollback girdisi
   `snapshot_path=target.snapshot_path` ile eski dosyayı reuse ediyor; rollback öncesi
   state korunmuyor. Problem 19'da tanımlananın *farklı* bir bug.

3. **`drg/extract/__init__.py` 1608 satır** — Rapor "1000+ satır" diyor; gerçek 1608.
   Teşhis muhafazakâr kalmış, sayı küçümsüyor.

---

## Değişmeyen P0 Blokerler

Bağımsız doğrulamadan geçen ve repository'nin temel işlevselliğini kıran sorunlar:

1. **`drg/evaluation/`** eksik → `drg eval *` her alt komutta `ModuleNotFoundError`
2. **`drg/reasoning/`** eksik → `drg extract --infer` `ModuleNotFoundError`
3. **CI workflow'ları** her push'ta başarısız → `checkout@v6`, `setup-python@v6`,
   `upload-artifact@v7`, `download-artifact@v8`, `codecov-action@v6` var olmayan versiyonlar
