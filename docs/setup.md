# DRG — Kurulum Rehberi

Bu doküman DRG'nin kurulum, konfigürasyon ve ilk çalıştırma adımlarını anlatır.
Repo'nun **genel mimarisi ve felsefesi** için `docs/project_overview.md`'ye bakın.

---

## 1. Gereksinimler

| Bileşen | Sürüm |
|---------|-------|
| Python  | 3.10, 3.11 veya 3.12 |
| pip     | son sürüm |

Sadece **core** kullanım için `pydantic` ve entity-resolution yolunda kullanılan
`numpy` kurulur. DSPy destekli extraction, API server, embedding provider'ları
ve clustering gibi yüzeyler opsiyonel extra'larla yüklenir.

---

## 2. Kurulum

### 2.1. Virtual environment (önerilir)

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# veya
.venv\Scripts\activate       # Windows
```

### 2.2. Paketi kur

```bash
# Sadece çekirdek
pip install -e .

# DSPy destekli extraction / `drg extract`
pip install -e ".[extract]"

# Developer modu (test + lint + type-check araçları ile)
pip install -e ".[dev]"

# Tüm opsiyonel özellikler ile (api, clustering, embedding, vs.)
pip install -e ".[all]"
```

> İpucu: `pyproject.toml` içinde `[project.optional-dependencies]` altında tüm
> extra grupları (extract, api, neo4j, openai, gemini, local, louvain, leiden,
> networkx, coreference, ...) tanımlıdır. Sadece kullandığını yükle.

---

## 3. Konfigürasyon

DRG, davranışını **environment variable**'larla yönetir. `.env.example` dosyasını
kopyalayıp kendi `.env` dosyanı oluştur:

```bash
cp .env.example .env
# Sonra .env dosyasını editleyip ilgili API key'leri doldur.
```

Önemli değişkenler:

| Değişken | Varsayılan | Açıklama |
|----------|------------|----------|
| `DRG_MODEL` | `openai/gpt-4o-mini` | DSPy/LiteLLM provider-prefixed model id |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / ... | — | İlgili provider için API key |
| `DRG_TEMPERATURE` | `0.0` | LLM temperature |
| `DRG_MAX_TOKENS` | `1500` | LLM output token bütçesi |
| `DRG_BASE_URL` | — | Self-hosted gateway / Ollama URL'i |

`.env` dosyası **kesinlikle commit edilmemelidir** (`.gitignore` zaten engelliyor).

---

## 4. Doğrulama

```bash
# Paket içe aktarılıyor mu?
python -c "import drg; print(drg.__version__)"

# CLI çalışıyor mu?
drg --help

# Non-integration testleri çalıştır
pytest -m "not integration"
```

API key gerektirmeyen smoke örneği (mock mode):

```bash
python examples/query_layer_example.py
```

---

## 5. Yaygın Sorunlar

### "ModuleNotFoundError: dspy"
Extraction extra'sı yüklenmemiş. Kaynak checkout için `pip install -e ".[extract]"`,
PyPI paketi için `pip install "drg-kg[extract]"` ile düzelt.

### "No API key configured"
İlgili `*_API_KEY` environment variable'ı setlenmemiş.
`.env` dosyasını yükleyen bir araç (`direnv`, `dotenv-cli`) kullanmıyorsan,
DRG kendi `utils/env_loader.py` modülü ile `.env`'i okuyabilir.

### "Multiple top-level packages discovered"
Eski bir setuptools sürümü. `pip install --upgrade pip setuptools` ile düzelt.

### Gemini "404 models/models/..."
`DRG_MODEL` değerinde `models/` prefix'i bırakmış olabilirsin. `drg/config.py`
çoğu format için otomatik normalize eder; sorun devam ederse `gemini/<isim>`
formatını kullan (örn. `gemini/gemini-2.0-flash-exp`).

---

## 6. Sonraki Adımlar

- **İlk kullanım akışı**: `docs/getting_started.md`
- **Pipeline akışı**: `docs/pipeline_overview.md`
- **Şema tasarımı**: `docs/schema_design.md`
- **API server**: `docs/api_server.md`
- **Chunking stratejileri**: `docs/chunking_strategy.md`
- **Clustering**: `docs/clustering_summarization.md`
