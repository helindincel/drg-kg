# DRG — Declarative Relationship Generation

[![CI](https://github.com/helindincel/drg-kg/actions/workflows/ci.yml/badge.svg)](https://github.com/helindincel/drg-kg/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/drg-kg.svg)](https://pypi.org/project/drg-kg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

DRG, yapılandırılmamış metinlerden derin, sorgulanabilir ve açıklanabilir **Bilgi Grafikleri (Knowledge Graph)** oluşturmak için tasarlanmış, **gelişmiş bir DSPy tabanlı framework**'tür. Vektör tabanlı aramaya dayanan geleneksel RAG sistemlerinin aksine DRG, karmaşık akıl yürütme (reasoning), zamansal analiz ve çoklu doküman sentezini destekleyen deterministik grafik yapıları kurar.

> 🇬🇧 **English readers:** [`README.md`](README.md) | 🗺 **Yol Haritası:** [`STATUS.md`](STATUS.md)

---

## 🚀 Temel Özellikler

### 🧠 Akıllı Çıkarım (Extraction)
- **Declarative Schema**: Alan modelinizi tanımlayın; DRG, **DSPy TypedPredictors** üzerinden çıkarım mantığını otomatik yönetir.
- **Enhanced Schema**: Açıklamalar, örnekler ve özellik grupları içeren zengin tip tanımları.
- **Otomatik Şema Üretimi**: Ham metinden başlangıç şemasını kendisi oluşturur—manuel modelleme gerektirmez.
- **Chunk-Aware İşleme**: Uzun dokümanları bağlam duyarlı parçalama ve parçalar arası ilişki birleştirme ile yönetir.

### 🕸 Grafik Zekası
- **Incremental Ingestion**: Yeni dokümanları, otomatik varlık çözümleme (entity resolution) ve ilişki tekilleştirme ile mevcut grafiğe ekleyin.
- **Temporal KG (Zamansal Grafik)**: `valid_from`/`valid_to` metadata desteği, kısmi tarihler ve zaman çizelgesi (timeline) oluşturma.
- **Çoklu Doküman Akıl Yürütme (Reasoning)**: Dokümanlar arası köprüleri keşfeden kural tabanlı çıkarım motoru (örn. A, B'yi tanıyor; B, C'yi tanıyor → A ve C bağlantılıdır).
- **Clustering & Communities**: Louvain ve Leiden algoritmaları ile otomatik topluluk tespiti ve LLM destekli grup özetleme.

### 🛠 Üretime Hazır (Production Ready)
- **Sorgu ve Akıl Yürütme Katmanı**: Çok adımlı yol bulma (path finding) ve kanıta dayalı (provenance-backed) deterministik grafik sorgulama.
- **Değerlendirme Sistemi (Evaluation)**: Çıkarım kalitesi için entegre metrikler (P/R/F1, NDCG, Hits@K) ve regresyon testi.
- **API ve UI**: Yerleşik FastAPI sunucusu ve etkileşimli Cytoscape.js web arayüzü.
- **Çoklu LLM Desteği**: OpenAI, Gemini, Anthropic, Ollama ve daha fazlası ile uyumlu.
- **MCP Entegrasyonu**: Grafik yeteneklerini Model Context Protocol üzerinden dışarı açar.

---

## 📦 Kurulum

```bash
# Çekirdek paket
pip install drg-kg

# Geliştirme araçları ve tüm opsiyonel özellikler
pip install "drg-kg[all]"
```

---

## ⚡ Hızlı Başlangıç

### 1. Tanımla ve Çıkar (Enhanced Schema)
```python
from drg import EnhancedDRGSchema, EntityType, Relation, extract_typed, EnhancedKG

schema = EnhancedDRGSchema(
    entity_types=[
        EntityType(name="Company", description="Şirketler ve kurumlar"),
        EntityType(name="Person", description="Kişiler")
    ],
    relations=[Relation("founded_by", "Company", "Person")]
)

text = "TechCorp, 2015 yılında Jane Doe tarafından kuruldu."
entities, triples = extract_typed(text, schema)

kg = EnhancedKG.from_typed(entities, triples, schema=schema)
print(kg.to_json())
```

### 2. Otomatik Şema Girişi
```bash
# Şemayı metinden türet ve grafiğe dönüştür
drg extract ornek.txt --auto-schema -o cikti_kg.json
```

---

## 🛠 Modüller ve CLI

### CLI Komutları
| Komut | Açıklama |
|:---|:---|
| `drg extract` | Dosya veya standart girdiden varlık/ilişki çıkarımı yap. |
| `drg eval run` | Altın standart (gold-standard) dataset üzerinden benchmark çalıştır. |
| `drg eval compare` | İki çalışma arasındaki kalite değişimlerini tespit et. |

### Artımlı Güncelleme ve Reasoning
```bash
# Mevcut grafiğe yeni doküman ekle
drg extract yeni_makale.txt --update global_kg.json --infer
```
*`--update` bayrağı verileri mevcut grafikle birleştirir. `--infer` bayrağı yeni bağlantıları keşfetmek için reasoning katmanını çalıştırır.*

---

## 📊 Değerlendirme Sistemi (Evaluation)

Pipeline kalitenizi ayrıntılı metriklerle ölçün:
- **Extraction**: Varlık ve İlişki P/R/F1 skorları.
- **Retrieval**: RAG değerlendirmesi için NDCG, Recall@K ve Hits@K.
- **Yapısal**: Grafik yoğunluğu, kapsama oranı (coverage) ve yetim düğüm oranları.

```bash
drg eval run benchmarks/kurumsal.json -o raporlar/sonuc.json
```

---

## ⏳ Zamansal (Temporal) Destek

DRG, zamanı grafik içinde birinci sınıf bir vatandaş olarak görür.
- **Durum Geçişi**: Bir varlığın özelliklerinin zamanla nasıl değiştiğini izleyin.
- **Zaman Çizelgesi**: Herhangi bir düğüm için kronolojik geçmiş üretin.
- **Çatışma Tespiti**: Zamansal çelişkileri (örn. bir kişinin aynı anda iki farklı rakip firmada CEO olması) belirleyin.

---

## 🏗 Proje Yapısı

```text
drg/
├── schema.py           # Enhanced Schema tanımları
├── extract/            # DSPy tabanlı extraction mantığı
├── graph/              # EnhancedKG ve grafik işlemleri
├── temporal/           # Zamansal reasoning ve timeline'lar
├── reasoning/          # Dokümanlar arası çıkarım
├── evaluation/         # Metrikler ve benchmarking
├── query/              # Deterministik sorgu motoru
├── api/                # FastAPI ve Cytoscape UI
└── cli.py              # Birleşik CLI giriş noktası
```

---

## 🤝 Katkıda Bulunma

Bağımlılık yönetimi için **uv**, testler için **pytest** kullanıyoruz.
```bash
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 Lisans

MIT © [Helin Dinçel](https://github.com/helindincel)
