# DRG-KG — Kalan Eksikler

> **Güncelleme:** 2026-06-16 · Versiyon: `0.1.0a1`
> Bu dosya `EKSIKLER.md`'den giderilen maddeler çıkarıldıktan sonra kalanları listeler.

---

## ✅ Tamamlananlar (bu oturumda giderildi)

| # | Madde |
|---|---|
| 1.1 | `KG` legacy sınıfı `_legacy.py`'ye taşındı, `DeprecationWarning` eklendi |
| 1.2 | `Cluster` çift tanımı: `ClusterResult` alias'ı ve açıklayıcı yorum |
| 1.3 | `LMConfig._configured` / `_lock` class-level yapıldı (singleton) |
| 1.4 | `mcp_server.py` (resmi SDK) oluşturuldu; `mcp_api.py` deprecated |
| 1.5 | `extract_typed_async`, `extract_from_chunks_async`, `asyncio.to_thread` eklendi |
| 1.6 | `DRG_API_KEY` env var + `X-API-Key` header authentication |
| 1.7/1.8 | `networkx`, `igraph` bağımlılıkları `louvain`/`leiden` extra'larına eklendi |
| 2.1 | `DRGSchema` / `EnhancedDRGSchema._validate` → `SchemaError` kullanıyor |
| 2.2 | Türkçe yorum/log mesajları İngilizce'ye çevrildi |
| 2.3 | mypy `var-annotated` + `attr-defined` düzeltildi; disable listesi 8 → 6 |
| 2.4 | spaCy ≥3.7 tespitinde `coreferee` uyumsuzluk uyarısı eklendi |
| 2.5 | `tests/test_extract_core.py` oluşturuldu (22 test: input guard, mock-mode, async, schema filtering) |
| 2.6 | `drg/__init__.py` `TYPE_CHECKING` bloğu genişletildi (tüm public semboller) |
| 2.7 | Pydantic üst sınır: `"pydantic>=2.0.0,<3.0.0"` |
| 3.1 | Coverage gate `70%`'e yükseltildi |
| 3.3 | CLI smoke testleri (`test_cli_smoke.py`) zaten mevcuttu |
| 3.4 | `tests/test_api_server.py` oluşturuldu (34+ test) |
| 3.5 | `tests/test_mcp_api_contract.py` zaten mevcuttu |
| 4.2 | `fallback_version = "0.0.0.dev0"` setuptools_scm'e eklendi |
| 4.3 | README göreli linkleri mutlak GitHub URL'lerine çevrildi |
| 4.4 | `SECURITY.md` PyPI metadata'sına eklendi |
| 4.5 | MIT License classifier eklendi |
| 4.6 | `all` extra'sına `coreferee>=1.0.0` eklendi |
| 7.1 | CORS wildcard → `DRG_CORS_ORIGINS` env var ile yapılandırılabilir |
| 7.2 | Prompt injection: `extract_typed`, `generate_schema_from_text`, `QueryRequest` input limitleri |

---

## ⬜ Kalan Eksikler

### 3.2 Kritik Modüllerin Coverage'ı Hâlâ Düşük

`extract/__init__.py` iyileştirildi (`test_extract_core.py`) ama diğer kritik modüller hâlâ düşük:

| Modül | Tahmini Coverage | Neden Kritik |
|---|---|---|
| `drg/graph/kg_core.py` | ~38% | `EnhancedKG` ana sınıfı |
| `drg/extract/_parsing.py` | ~12% | Pure Python, LLM bağımlılığı yok |
| `drg/extract/_relations.py` | ~26% | Pure Python, LLM bağımlılığı yok |

**Yapılacak:** Her modül için ayrı test dosyaları ekle:
- `tests/test_kg_core_extended.py` — `EnhancedKG.add_node/edge`, `to_json`, community queries
- `tests/test_extract_parsing.py` — `_parse_entity_output`, `_parse_relation_output` edge-case'leri
- `tests/test_extract_relations.py` — `_build_relation_filter`, `_apply_confidence_filter`

---

### 4.1 Gerçek PyPI'da Yayın Yok

Kod hazır, `release.yml` hazır. Sadece GitHub Actions secret'ları eksik.

**Yapılacak (kullanıcı işlemi):**
1. [pypi.org](https://pypi.org) üzerinde `drg-kg` paket adını rezerve et
2. `PYPI_API_TOKEN` secret'ını repo'nun Settings → Secrets → Actions'a ekle
3. `git tag v0.1.0a1 && git push --tags` ile release workflow'u tetikle

---

### 5.1 `docs/` Türkçe Belgeler İngilizce'ye Çevrilmeli

| Dosya | Durum |
|---|---|
| `docs/schema_design.md` | Türkçe |
| `docs/chunking_strategy.md` | Türkçe |
| `docs/relationship_model.md` | Türkçe |
| `docs/clustering_summarization.md` | Türkçe |
| `docs/optimizer_design.md` | Türkçe |
| `docs/mcp_integration.md` | Türkçe |

**Yapılacak:** Her dosyayı İngilizce'ye çevir (içerik korunacak, sadece dil değişecek). `pipeline_overview.tr.md` ve `project_overview.tr.md` TR kopyası olarak bırakılabilir.

---

### 5.2 Otomatik API Referansı Yok

Proje düzeyinde `mkdocs.yml` veya `sphinx/conf.py` yok. Tüm API dokümantasyonu yalnızca README özet tablosundan oluşuyor.

**Yapılacak:**
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
```
Ardından `mkdocs.yml` oluştur:
```yaml
site_name: drg-kg
theme:
  name: material
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
nav:
  - Home: index.md
  - API Reference: api/
  - Guides: docs/
```

---

### 5.3 CHANGELOG'da Migration Rehberi Yok

`CHANGELOG.md` sürüm notlarına `## Migration` bölümü ekle: "daha önce X kullanıyordunuz → şimdi Y yapın" formatında örneğin `KG` → `EnhancedKG` geçişi.

---

### 5.4 Architecture Decision Records (ADR) Yok

"DSPy neden seçildi?", "Protocol vs ABC?", "Neo4j neden opsiyonel?" gibi kararlar `project_overview.md` prosasına gömülü.

**Yapılacak:** `docs/adr/` klasörü oluştur, `ADR-001-dspy-over-langchain.md` gibi dosyalar ekle.

---

### 6.1 `pre-commit` Hook Versiyonları Senkronize Edilmeli

Dependabot `pre-commit` kaynaklarını takip etmiyor; `.pre-commit-config.yaml` hook versiyonları manuel güncelleniyor.

**Yapılacak:** `.github/dependabot.yml`'a pre-commit ecosystem girişi ekle:
```yaml
- package-ecosystem: "pip"
  directory: "/"
  schedule:
    interval: "weekly"
- package-ecosystem: "github-actions"
  directory: "/"
  schedule:
    interval: "weekly"
```
(pre-commit hook'ları için özel bir Dependabot desteği yok; alternatif olarak `pre-commit autoupdate` komutunu CI'da haftalık çalıştır.)

---

### 6.2 Integration Test CI Job'u Yok

`pytest -m integration` CI'da çalışmıyor. Mock-LLM modunda bile scheduled test yok.

**Yapılacak:** `.github/workflows/integration.yml` oluştur:
```yaml
name: Integration Tests
on:
  schedule:
    - cron: '0 4 * * 1'  # Pazartesi 04:00 UTC
  workflow_dispatch:
jobs:
  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev,api,mcp]"
      - run: pytest -m integration --timeout=120
        env:
          DRG_USE_MOCK: "1"
```

---

### 6.3 `CODECOV_TOKEN` Secret Eksik Olabilir

`ci.yml`'da `${{ secrets.CODECOV_TOKEN }}` referansı var ama secret ayarlanmamışsa Codecov yüklemeleri sessizce başarısız oluyor ve PR'larda coverage delta görünmüyor.

**Yapılacak (kullanıcı işlemi):**
1. [codecov.io](https://codecov.io) → Settings → Upload token'ı kopyala
2. GitHub repo → Settings → Secrets → `CODECOV_TOKEN` olarak ekle

---

### 6.4 Release Workflow Trusted Publishing Kullanmıyor

`release.yml` `pypa/gh-action-pypi-publish` kullanıyor ama hâlâ `password: ${{ secrets.PYPI_API_TOKEN }}` ile — OIDC Trusted Publishing değil.

**Yapılacak:**
1. pypi.org → Account Settings → Publishing → Trusted Publishers → GitHub repo ekle
2. `release.yml`'dan `password:` satırını kaldır (OIDC otomatik çalışır):
```yaml
- uses: pypa/gh-action-pypi-publish@release/v1
  with:
    repository-url: https://upload.pypi.org/legacy/
    # password satırı kaldırılır — OIDC kullanılır
```
3. Workflow'a `permissions: id-token: write` ekle

---

## Öncelik Sırası

| Öncelik | Madde | Efor | Bloke? |
|---|---|---|---|
| 🔴 Yüksek | **4.1** PyPI publish | Düşük | Kullanıcı işlemi (secret) |
| 🔴 Yüksek | **6.3** CODECOV_TOKEN | Düşük | Kullanıcı işlemi (secret) |
| 🟠 Orta | **3.2** kg_core + parsing coverage | Orta | Hayır |
| 🟠 Orta | **5.1** Türkçe docs çevirisi | Yüksek | Hayır |
| 🟠 Orta | **6.2** Integration test CI job | Düşük | Hayır |
| 🟡 Düşük | **6.4** Trusted Publishing | Düşük | 4.1 sonrası |
| 🟡 Düşük | **5.2** mkdocs kurulumu | Orta | 5.1 sonrası |
| 🟡 Düşük | **5.3** CHANGELOG migration guide | Düşük | Hayır |
| 🟡 Düşük | **5.4** ADR klasörü | Düşük | Hayır |
| 🟢 Bilgi | **6.1** pre-commit autoupdate | Düşük | Hayır |
