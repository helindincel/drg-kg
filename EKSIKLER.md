# DRG-KG — Eksikler & Sorun Analizi

> **Oluşturulma:** 2026-06-16 · Versiyon: `0.1.0a1` · Kaynak: kod incelemesi + CI/STATUS.md analizi  
> **Amaç:** Projenin mimari, kod kalitesi ve yayın sürecindeki açıklarını tek bir yerde toplamak.

---

## 1. Mimari Sorunlar

### 1.1 İki Paralel KG Sınıfı — Yol Haritası Eksik

`drg/graph/__init__.py` içinde **legacy `KG`** sınıfı ile `drg/graph/kg_core.py` içindeki **`EnhancedKG`** yan yana yaşıyor. `KG` sınıfı `__init__.py`'nin içinde doğrudan implement edilmiş (başka bir modülden import edilmiyor), `EnhancedKG` ise ayrı bir dosyada.

**Sorunlar:**
- `KG` ne zaman kaldırılacağına dair `DeprecationWarning` yok.
- Hangi sınıfın "doğru" tercih olduğu kullanıcıya net değil.
- `KG.from_typed` → `EnhancedKG` migration helper'ı yok.

**Önerilen düzeltme:** `KG`'yi `drg/graph/_legacy.py`'ye taşı, `__init__.py`'den `DeprecationWarning` ile import et ve `docs/` altına bir migration rehberi ekle.

---

### 1.2 `Cluster` Sınıfı İki Yerde Tanımlı

`drg/clustering/algorithms.py:Cluster` ile `drg/graph/kg_core.py:Cluster` birbirinden bağımsız iki farklı `Cluster` dataclass'ı. Clustering modülünün ürettiği `Cluster`, `EnhancedKG` içinde tutulandan farklı alanlar içeriyor. Bu, veri akışında sessiz tip uyumsuzluklarına neden olabilir.

**Önerilen düzeltme:** `drg/schema.py` veya `drg/graph/kg_core.py`'deki `Cluster` tek kaynak olsun; `clustering/algorithms.py` onu import etsin.

---

### 1.3 `LMConfig` Singleton Değil — Birden Fazla DSPy Konfigürasyonu Riski

`drg/config.py:LMConfig` sınıfındaki `_configured` flag **instance-level**. Pipeline farklı noktalardan birden fazla `LMConfig()` örneği oluşturursa DSPy global ayarı birden fazla kez üzerine yazılır. Özellikle `async` / çok thread'li kullanımda yarış koşulu oluşabilir.

**Önerilen düzeltme:** `_configured` ve `_instance` flag'lerini class-level yaparak singleton pattern uygula ya da `functools.lru_cache` ile modül-level cache ekle.

---

### 1.4 `mcp_api.py` — Resmi MCP SDK Kullanılmıyor

`drg/mcp_api.py` (608 LOC), MCP protokolünü elle uyguluyor: `MCPRequest`, `MCPResponse` dataclass'ları JSON-RPC 2.0'ı taklit ediyor. Ancak resmi [`mcp` Python SDK](https://github.com/modelcontextprotocol/python-sdk) kullanılmıyor. MCP spesifikasyonu evrildiğinde bu dosya kırılacak ve bakım yükü yüksek olacak.

**Ayrıca:** Transport katmanı (stdio / HTTP+SSE) yok. MCP sunucu olarak gerçekten çalışmak için `mcp.server.fastmcp` veya `mcp.server.stdio_server` gerekiyor.

**Önerilen düzeltme:** `mcp` paketini opsiyonel bağımlılık olarak ekle (`pip install drg-kg[mcp]`), `mcp_api.py`'yi resmi SDK üzerine yeniden yaz.

---

### 1.5 Async Desteği Yok

Tüm LLM çağrıları (`extract_typed`, `KGExtractor.forward`) ve embedding çağrıları senkron. FastAPI endpoint'leri `def` olarak tanımlanmış (async değil), bu da her isteğin thread pool'u bloklaması anlamına gelir. Yoğun yük altında API sunucusu daralır.

**Önerilen düzeltme:** En azından `async def extract_typed_async()` wrapper'ı ekle, API endpoint'lerini `async def` yap ve LLM çağrılarını `asyncio.to_thread()` ile sardır.

---

### 1.6 API Sunucusunda Authentication Yok

`drg/api/server.py` CORS middleware'i var ancak hiçbir endpoint'te authentication/authorization mekanizması (API key, Bearer token, vb.) yok. Sunucu dışa açılırsa herkes `POST /api/query` çağırabilir.

**Önerilen düzeltme:** En azından opsiyonel `DRG_API_KEY` env var kontrolü ekle; docs'a deployment uyarısı yaz.

---

### 1.7 `networkx` Opsiyonel Ama Kritik Kodda Kullanılıyor

`pyproject.toml`'da `networkx` optional extras'ta. Ancak clustering algoritmaları (`LouvainClustering`, `LeidenClustering`) ve `auto_clusters.py` networkx olmadan kırılıyor. Kullanıcı `pip install drg-kg[louvain]` kursa bile `networkx`'i ayrıca kurması gerekiyor çünkü `louvain` extra'sı networkx'i içermiyor.

**Önerilen düzeltme:** `louvain`, `leiden`, `spectral` extra'larına `"networkx>=3.0"` ekle.

---

### 1.8 `leidenalg` — `igraph` Bağımlılığı Bildirilmemiş

`leidenalg` paketi `igraph`'a ihtiyaç duyuyor ancak `pyproject.toml`'daki `leiden` extra'sında `igraph` listelenmemiş. `pip install drg-kg[leiden]` başarılı gibi görünüp runtime'da `ImportError` verebilir.

**Önerilen düzeltme:**
```toml
leiden = [
    "leidenalg>=0.9.0",
    "igraph>=0.10.0",
]
```

---

## 2. Kod Kalitesi Sorunları

### 2.1 Schema Sınıfları `dataclass` Kullanıyor, Pydantic Değil

`drg/schema.py` tüm schema sınıfları (`Entity`, `Relation`, `EntityType`, `RelationGroup`, vb.) Python `dataclass` ile tanımlanmış. `drg/api/server.py`'deki request/response modelleri ise `pydantic.BaseModel` kullanıyor. İki farklı validation sistemi aynı kodbase'de. Hatalar `ValueError` fırlatıyor, `SchemaError` değil.

**Önerilen düzeltme:** Schema sınıflarını Pydantic `BaseModel`'e migrate et veya en azından `__post_init__` hataları `SchemaError` (typed hierarchy) fırlatsın.

---

### 2.2 Turkish Comments / Docstrings in English Codebase

Kod İngilizce ama şu dosyalarda Türkçe yorum/string var:

| Dosya | Yer |
|---|---|
| `drg/config.py` | `# Environment variable'lardan otomatik oku` |
| `drg/cli.py` | `"""Varsayılan şema: Company -> Product"""` |
| `drg/schema.py` | `Relation.description: str = ""  # Bağlantı sebebi...` |
| `drg/schema.py` | `Relation.detail: str = ""  # Bağlantı detayı...` |

Bu, İngilizce konuşan katkıcıları zorluyor ve PEP 8 ruhuna aykırı.

---

### 2.3 `mypy` Gradual Modda — 8 Hata Kodu Devre Dışı

`pyproject.toml`'da `mypy` şu ignore'larla çalışıyor: `assignment`, `arg-type`, `var-annotated`, `index`, `misc`, `override`, `truthy-function`, `attr-defined`. Bu, gerçek tip hatalarının CI'da gizlenmesi anlamına gelir. Her disabled kural, potansiyel bir runtime hatasının işaretidir.

**Durum:** `STATUS.md`'de takip ediliyor fakat milestone/milestone tarihi yok.

---

### 2.4 `coreferee` Bağımlılığı Kırık Olabilir

`pyproject.toml`'daki `coreference` extra'sı:
```toml
coreference = [
    "spacy>=3.7.0",
    "coreferee>=1.0.0",
]
```
`coreferee` kütüphanesi 2023'ten beri bakımsız ve spaCy 3.7+ ile uyumsuzluk sorunları bildiriliyor. `all` extra'sı da `coreferee`'yi **içermiyor** (`spacy` var ama `coreferee` yok). Yani `pip install drg-kg[all]` ile coreference çalışmıyor.

**Önerilen düzeltme:** `all` extra'sına `coreferee` ekle veya `coreferee` desteğini `spacy-experimental`/`huggingface` tabanlı alternatifle değiştir.

---

### 2.5 `extract/__init__.py` — 528 LOC Tek Dosya, %14 Coverage

Projenin kalbi olan extraction loop tek bir 528 satırlık `__init__.py`'de. Bu dosya:
- En düşük test coverage'ına sahip core modül (%14).
- Test edilmesi en zor kısımlar içeriyor (LLM çağrıları) ama mock'lanabilir alt-fonksiyonlar da var.
- `_configure_llm_auto`, `resolve_entities_and_relations`, `resolve_coreferences` mock-friendly olarak exposed edilmiş ama kapsamlı test yok.

---

### 2.6 `__getattr__` ile Lazy Loading — IDE Desteği Kırık

`drg/__init__.py:__getattr__` ile yapılan lazy loading, PyCharm/VSCode gibi IDE'lerde autocomplete'i bozuyor. `extract_typed`, `KGExtractor` gibi en çok kullanılan semboller statik import olarak görünmüyor.

**Önerilen düzeltme:** `TYPE_CHECKING` bloğu ile statik tip bilgisi ver:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .extract import extract_typed, KGExtractor
```

---

### 2.7 Pydantic Üst Sınır Yok

`dependencies = ["pydantic>=2.0.0"]` — Pydantic `3.0` çıktığında breaking change içerebilir. `"pydantic>=2.0.0,<3.0.0"` kullanmak güvenli olur.

---

## 3. Test & Coverage Sorunları

### 3.1 Coverage Gate Çok Düşük (%50)

Mevcut gate `50%`, gerçek coverage `~60%`. Alpha bir kütüphane için minimum `70%`, release öncesi `80%` hedeflenmeli.

### 3.2 Kritik Modüllerin Coverage Durumu

| Modül | Satır | Coverage | Neden Kritik |
|---|---:|---:|---|
| `drg/extract/__init__.py` | ~528 | ~14% | Core extraction loop |
| `drg/graph/kg_core.py` | ~172 | ~38% | `EnhancedKG` ana sınıfı |
| `drg/mcp_api.py` | ~608 | hariç | MCP entegrasyon noktası |
| `drg/cli.py` | — | hariç | Kullanıcı giriş noktası |
| `drg/extract/_parsing.py` | ~70 | ~12% | Pure Python, LLM bağımlılığı yok |
| `drg/extract/_relations.py` | ~70 | ~26% | Pure Python, LLM bağımlılığı yok |
| `drg/api/__init__.py` | 2 | 0% | Server smoke test eksik |

### 3.3 CLI Smoke Test Yok

`drg --help` ve `drg extract -` için en azından bir `subprocess` testi olmalı. CLI `dev` extras tarafından da kapsanmıyor.

### 3.4 API Server Contract Test Yok

`drg/api/server.py` için `fastapi.testclient.TestClient` ile en azından `GET /api/graph` ve `POST /api/query` happy-path testleri yok.

### 3.5 `mcp_api.py` Contract Test Yok

608 satırlık MCP implementasyonu için hiç test yok ve coverage konfigürasyonundaki `omit` listesinde de yok — tamamen görünmez.

---

## 4. Yayın (Publish) Sorunları

### 4.1 Gerçek PyPI'da Yok

`STATUS.md`'e göre paket sadece `test.pypi.org`'da. `release.yml` hazır ve sırası gelince `PYPI_API_TOKEN` secret'ı GitHub'a eklenmesi yeterli. Ama henüz yapılmamış.

**Checklist:**
- [ ] `pypi.org`'da `drg-kg` ismi rezerve edildi mi? (30 saniye, ücretsiz)
- [ ] `PYPI_API_TOKEN` secret GitHub'a eklendi mi?
- [ ] `CODECOV_TOKEN` secret GitHub'a eklendi mi?

---

### 4.2 `_version.py` — Editable Install'da Versiyon "0.0.0+unknown"

`setuptools_scm` `drg/_version.py` dosyasını **build sırasında** üretiyor. Bu dosya `.gitignore`'da. Git tag'i olmayan dirty tree'de veya `pip install -e .` yapıldığında `__version__ = "0.0.0+unknown"` görünür. Bu, `pip show drg-kg` ve `importlib.metadata.version("drg-kg")`'nin doğru sonuç verememesi demek.

**Önerilen düzeltme:** CI'da `pip install -e .` öncesi `git tag v0.0.0` koy veya `fallback_version = "0.0.0.dev0"` ekle:
```toml
[tool.setuptools_scm]
fallback_version = "0.0.0.dev0"
```

---

### 4.3 README'deki Bağlantılar PyPI'da Kırılacak

`README.md` göreli bağlantılar kullanıyor:
- `[docs/setup.md](docs/setup.md)`
- `[examples/quickstarts/](examples/quickstarts/)`

Bunlar GitHub'da çalışır ama PyPI'da render edilmez. PyPI `README.md`'yi raw HTML olarak gösterir, göreli yollar 404 verir.

**Önerilen düzeltme:** `pyproject.toml`'a canonical URL ekle:
```toml
[project.urls]
Documentation = "https://github.com/helindincel/drg-kg/tree/main/docs"
```
ve README'deki göreli linkleri absolute GitHub URL'lerine çevir.

---

### 4.4 `SECURITY.md` PyPI Metadata'sında Eksik

`SECURITY.md` dosyası var ama `pyproject.toml`'daki `[project.urls]` bölümünde `"Bug Tracker"` veya `"Security"` girişi yok. PyPI sayfasında güvenlik açığı raporlama yolu görünmez.

**Önerilen düzeltme:**
```toml
[project.urls]
"Bug Tracker" = "https://github.com/helindincel/drg-kg/issues"
"Security"    = "https://github.com/helindincel/drg-kg/security/advisories"
```

---

### 4.5 `License` Classifier Eksik

`pyproject.toml`'da `license = "MIT"` var ama `classifiers` listesinde `"License :: OSI Approved :: MIT License"` yok. PyPI'da lisans arama filtresi çalışmaz.

**Önerilen düzeltme:**
```toml
classifiers = [
    ...
    "License :: OSI Approved :: MIT License",
]
```

---

### 4.6 `all` Extra'sı `coreferee` İçermiyor

Yukarıda (2.4) belirtildi. `pip install drg-kg[all]` yaptığında coreference resolution feature'ı çalışmıyor. Kullanıcılar bunu fark etmesi zor.

---

## 5. Dokümantasyon Sorunları

### 5.1 `docs/` Klasörünün Büyük Çoğunluğu Türkçe

`STATUS.md`'de listeleniyor ama somut ilerleme yok:

| Doküman | EN | TR |
|---|:-:|:-:|
| `schema_design.md` | ✗ | ✓ |
| `chunking_strategy.md` | ✗ | ✓ |
| `relationship_model.md` | ✗ | ✓ |
| `clustering_summarization.md` | ✗ | ✓ |
| `optimizer_design.md` | ✗ | ✓ |
| `mcp_integration.md` | ✗ | ✓ |

Non-Turkish konuşan katkıcılar için deep docs erişilemez.

---

### 5.2 Otomatik API Referansı Yok

`mkdocs` + `mkdocstrings` veya `sphinx` + `autodoc` konfigürasyonu yok. `pyproject.toml`'da `[tool.mkdocs]` veya `mkdocs.yml` yok. Mevcut tek API listesi README'deki özet tablo.

---

### 5.3 CHANGELOG'da Migration Rehberi Yok

Pre-`1.0` minor bump'lar breaking change içerebilir (README ve CHANGELOG'da belirtilmiş). Ama `CHANGELOG.md`'de "eğer önceden X kullanıyordunuz, şimdi Y yapın" formatında bir migration guide yok.

---

### 5.4 Architecture Decision Records (ADR) Yok

"DSPy neden RAG yerine?", "Strategy pattern neden ABC yerine Protocol?", "Neden Neo4j opsiyonel?" gibi kararlar `project_overview.md`'de proza gömülü. ADR klasörü (`docs/adr/`) olmadığı için kararların gerekçesine ulaşmak güç.

---

## 6. CI / DevOps Sorunları

### 6.1 `pre-commit` CI'da Çalışıyor Ama Hook Versiyonu Senkronize Edilmeli

`.github/workflows/ci.yml`'da `pre-commit/action@v3.0.1` kullanılıyor. `.pre-commit-config.yaml`'daki hook versiyonları CI'da cache'lendiği için yerel araçlarla kayabiliyor. Dependabot `pre-commit` source'unu takip etmiyor.

---

### 6.2 Integration Test Job Yok

`pytest -m integration` CI'da hiç çalışmıyor. Scheduled workflow (günlük/haftalık) ile mocked-LLM veya gerçek API key secrets ile integration test çalıştırılmıyor. Sessiz extraction kırılmaları fark edilmiyor.

---

### 6.3 Coverage Codecov'a Gitmiyor (Secret Eksik Olabilir)

`ci.yml`'da `codecov/codecov-action@v4` var ve `fail_ci_if_error: false`. Eğer `CODECOV_TOKEN` secret eklenmemişse yüklemeler public repo rate limit'e takılır ve badge güncellenmez. PR'larda coverage delta görünmez.

---

### 6.4 Release Workflow Trusted Publishing Kullanmıyor

`release.yml` `PYPI_API_TOKEN` secret ile `twine` kullanıyor. Daha güvenli yol [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/) (OIDC): token saklama ve rotasyon riski ortadan kalkar.

**Önerilen düzeltme:**
```yaml
- uses: pypa/gh-action-pypi-publish@release/v1
  with:
    repository-url: https://upload.pypi.org/legacy/
```
Ve PyPI tarafında GitHub OIDC trusted publisher tanımla.

---

## 7. Güvenlik Sorunları

### 7.1 CORS Wildcard Potansiyeli

`drg/api/server.py`'de `CORSMiddleware` ekleniyor. Konfigürasyonunun `allow_origins=["*"]` veya eşdeğeri olup olmadığı kontrol edilmeli. Production deployment'ta wildcard CORS açık bırakılmamalı.

### 7.2 LLM Prompt Injection

`POST /api/query` endpoint'i kullanıcı girdisini doğrudan `execute_deterministic_query`'ye iletmek için kullanılıyor. Ancak `generate_schema_from_text` ve `extract_typed` akışlarında LLM'e iletilen metin sanitize edilmiyor. Özellikle `auto_schema=True` modunda kullanıcı girdisi doğrudan DSPy promptuna giriyor.

**Önerilen düzeltme:** En azından `docs/`'ta prompt injection riski konusunda uyarı ekle ve input uzunluk/karakter limiti uygula.

---

## 8. Özet Öncelik Sırası

| Öncelik | Sorun | Etki | Efor |
|---|---|---|---|
| 🔴 Kritik | MCP SDK kullanılmıyor (`mcp_api.py`) | MCP entegrasyon kırık/bakımsız | Yüksek |
| 🔴 Kritik | Gerçek PyPI'da yayın yok | Kullanıcılar ulaşamıyor | Düşük |
| 🔴 Kritik | `all` extra'sı coreferee içermiyor | Feature sessiz bozuk | Düşük |
| 🟠 Yüksek | `leidenalg` igraph bağımlılığı eksik | Runtime ImportError | Düşük |
| 🟠 Yüksek | `extract/__init__.py` %14 coverage | Core loop güvensiz | Orta |
| 🟠 Yüksek | API auth yok | Güvenlik riski | Orta |
| 🟠 Yüksek | README göreli linkleri PyPI'da kırık | Kötü kullanıcı deneyimi | Düşük |
| 🟡 Orta | `Cluster` sınıfı çift tanımı | Sessiz tip uyumsuzluğu | Orta |
| 🟡 Orta | `LMConfig` singleton değil | Thread safety | Düşük |
| 🟡 Orta | Async desteği yok | API ölçeklenemiyor | Yüksek |
| 🟡 Orta | `mypy` 8 hata kodu disabled | Tip güvensizliği | Yüksek |
| 🟡 Orta | `docs/` büyük çoğunluğu Türkçe | Katkıcı engeli | Orta |
| 🟢 Düşük | Otomatik API referansı yok | Geliştirici DX | Orta |
| 🟢 Düşük | License classifier eksik PyPI'da | Görünürlük | Düşük |
| 🟢 Düşük | Trusted Publishing kullanılmıyor | Güvenlik best practice | Düşük |
| 🟢 Düşük | IDE lazy load autocomplete sorunu | Geliştirici DX | Düşük |
