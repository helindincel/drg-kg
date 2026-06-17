# DRG-KG — Sıfırdan Başlangıç Kılavuzu

> Bu dosya, projeyi ilk defa görenler için yazıldı. Programlama bilmen
> gerekiyor (Python kurulumu, terminal kullanımı), ama **Knowledge Graph,
> LLM, DSPy gibi kavramları bilmesen de** sorun yok — hepsini açıklıyorum.
>
> Daha hızlı, daha teknik bir özet için: [`README.tr.md`](README.tr.md).
> Mimari için: [`docs/project_overview.md`](docs/project_overview.md).

---

## 1. Bu proje ne işe yarıyor?

**Tek cümleyle:** Düz metin alıp ondan bilgi grafiği (knowledge graph)
çıkaran bir Python kütüphanesi.

**Daha somut bir örnek:** Apple Inc. hakkında 1 sayfalık bir Wikipedia
yazısı versen, DRG sana şunu üretir:

```text
Apple Inc.       --PRODUCES-->     iPhone
Apple Inc.       --PRODUCES-->     MacBook Pro
Apple Inc.       --HAS_CEO-->      Tim Cook
Steve Jobs       --FOUNDED-->      Apple Inc.
Apple Park       --LOCATED_IN-->   Cupertino, California
Tim Cook         --SUCCEEDED_AS_CEO--> Steve Jobs
...
```

Bu listenin tamamına "knowledge graph" diyoruz. Düz metni okuyup, içindeki
**varlıkları** (kişi, şirket, ürün vs.) ve aralarındaki **ilişkileri**
(üretir, kuruldu, çalışır, vb.) çıkarıyor; sonra bunu hem JSON dosyası
olarak kaydediyor hem de tarayıcıda interaktif graph olarak gösteriyor.

### Ne işe yarar?

- **Araştırmacılar:** 100 sayfa makaleyi taratıp "hangi ilaç hangi geni
  etkiliyor" sorusunu çıkarabilir.
- **Şirketler:** Şirket içi dokümanları KG'ye dönüştürüp arama / soru-cevap
  altyapısı kurabilir.
- **Veri gazetecileri:** Yıllarca biriken haberlerden "X kişi Y şirketle
  ne zaman ilişkilendirildi" gibi bağlantıları çıkarabilir.
- **Öğrenci/hobici:** Wikipedia gibi metinlerden özet graph'lar üretip
  görselleştirebilir.

### Ne işe yaramaz?

- **ChatGPT alternatifi değil.** Sana cevap üretmez, metinden yapı çıkarır.
- **Soru-cevap aracı değil.** Çıkardığı KG'yi başka araçlarla (Neo4j,
  RAG sistemleri vs.) sorgulayabilirsin ama DRG'nin kendisi sorgu motoru
  değil.
- **Kategorize/etiketleme aracı değil.** Hedef cümle/paragraf değil,
  birbirine bağlı varlıklar.

---

## 2. Birkaç kavram (şart olan minimum)

### 2.1. Knowledge Graph (Bilgi Grafiği)

Düğümler (nodes) ve onları birbirine bağlayan kenarlar (edges) olan
yapı. Düğüm = bir varlık (örn. "iPhone"). Kenar = iki varlık arası ilişki
(örn. "iPhone --RUNS_ON--> iOS"). Üçlüye **(kaynak, ilişki, hedef)**
"triple" denir. Tüm KG, böyle triple'ların toplamıdır.

### 2.2. LLM (Large Language Model)

ChatGPT, Gemini, Claude gibi yapay zekâ modelleri. DRG, metinden varlık
ve ilişki çıkarmak için bir LLM'e prompt gönderiyor. Yani çalışması için
mutlaka bir LLM **API key**'i (veya lokal Ollama) gerekiyor.

### 2.3. Schema (Şema)

LLM'e "ne aramayı bilmesi gerektiğini" söyleyen şablon. Örneğin
biyografik metinler için:

- **Entity types:** `Person`, `Organization`, `Place`, `Award`, `Discovery`
- **Relations:** `worked_at`, `born_in`, `received` …

Schema'yı sen elle yazabilir veya DRG'ye "metinden otomatik üret" dedirtirsin.

### 2.4. DSPy

Python kütüphanesi. LLM prompt'larını manuel yazmak yerine, schema'dan
otomatik üretmeyi sağlıyor. DRG bunu kapağın altında kullanıyor; sen
DSPy'i bilmek zorunda değilsin.

### 2.5. Pipeline (Boru hattı)

Metni KG'ye dönüştürmek tek adımda olmuyor. DRG'nin pipeline'ı:

```
Metin
  ↓ Chunking         (uzun metni parçalara böl, LLM'in context window'una sığsın)
  ↓ Schema gen       (schema yoksa metinden otomatik tipler/ilişkiler türet)
  ↓ Extraction       (her chunk'tan entity ve triple çıkar)
  ↓ Build EnhancedKG (parçaları birleştir, tekrarları temizle)
  ↓ Hub-split        ("Apple Inc. her şeye bağlı" tarzı süper hub'ları yumuşat)
  ↓ Clustering       (Louvain ile birbirine yakın düğümleri grupla)
  ↓ Community report (her cluster için özet üret)
  ↓ JSON / Neo4j / FastAPI UI
```

Tüm bu adımlar [`examples/full_pipeline_example.py`](examples/full_pipeline_example.py)
içinde tek dosyada zincirleniyor.

---

## 3. Kurulum (adım adım)

### 3.1. Gereksinimler

- **Python 3.10 veya üstü.** Kontrol:
  ```bash
  python3 --version
  ```
  Eğer "Python 3.10.x" veya üstü görüyorsan tamam. Değilse
  [python.org](https://www.python.org/downloads/) veya `brew install python@3.12`.

- **Git** (repoyu klonlamak için).

- **Bir LLM API key'i** veya **Ollama** (lokal model). En kolayı:
  [Google AI Studio](https://aistudio.google.com/apikey)'den ücretsiz
  Gemini key. Free tier günlük ~250 istek veriyor, küçük metinlerde
  fazlasıyla yeter.

### 3.2. Repoyu indir

```bash
git clone https://github.com/helindincel/drg-kg.git
cd drg-kg
```

### 3.3. Virtual environment oluştur (öneri)

Sistemde Python paketlerini kirletmemek için izole bir ortam:

```bash
python3 -m venv .venv
source .venv/bin/activate     # macOS / Linux
# Windows: .venv\Scripts\activate
```

Aktif olduğunda terminalin başında `(.venv)` görürsün.

### 3.4. Paketi kur

DRG'yi geliştirici modunda (yapacağın değişiklikler anında etkili olsun
diye), Gemini desteği ile beraber kur:

```bash
pip install -e ".[gemini]"
```

Diğer sağlayıcılar için: `".[openai]"`, `".[anthropic]"`, `".[ollama]"`,
veya hepsi: `".[all]"`.

### 3.5. .env dosyası oluştur

API key'ini ve hangi modeli kullanmak istediğini DRG'ye söylemenin yolu:

```bash
cp .env.example .env
```

Şimdi `.env`'yi açıp düzenle:

```dotenv
# Hangi modeli kullanacağız?
DRG_MODEL=gemini/gemini-2.5-flash

# Gemini API key (https://aistudio.google.com/apikey)
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Çıktı uzunluğu sınırı (uzun metinlerde 8192 öner)
DRG_MAX_TOKENS=8192
```

> **Önemli:** `.env` dosyası `.gitignore`'da, asla commit'lenmez. Key'in
> yanlışlıkla GitHub'a düşmez. Ama yine de key'i kimseyle paylaşma.

> **Shell export'u var mı?** Eğer `.zshrc` / `.bashrc` içinde
> `export GEMINI_API_KEY=...` satırın varsa, `.env` onu **ezmez**
> (override=False). O satırı kaldır veya `unset GEMINI_API_KEY` çalıştır.

### 3.6. Test çalıştırması

Her şey tamam mı? En küçük örneği (Marie Curie quickstart):

```bash
python examples/quickstarts/01_wikipedia_article.py
```

Beklenen çıktı (~5-10 saniye):

```text
Schema: 5 entity types, 7 relations
Input text (471 chars): Marie Curie was a Polish-born ...
Extracted 7 entities and 6 triples.
Entities:
  - Marie Curie  [Person]
  - Pierre Curie [Person]
  ...
KG written to: examples/quickstarts/01_wikipedia_article.json
```

Hata alırsan bölüm 7'ye (sorun giderme) bak.

---

## 4. İlk gerçek pipeline çalıştırması

Quickstart küçük, hard-coded bir metni işliyor. Asıl pipeline'ı tam
özellikleriyle (chunking, auto-schema, clustering, reports) görmek için:

```bash
python examples/full_pipeline_example.py 1example
```

`1example` = `inputs/1example_text.txt` dosyası (Apple Inc. profili,
4.8 KB). Bu komut ~2-3 dakika sürer ve şunu yapar:

1. Metni okur (4.8 KB).
2. Schema'yı LLM'e otomatik ürettirir (8 entity tipi, 4 relation grubu).
3. Metni 768-token'lık chunk'lara böler.
4. Her chunk için entity + triple çıkarır.
5. Apple Inc. süper hub olduğu için onu 9 alt-proxy'ye böler.
6. Louvain algoritmasıyla 8 community bulur.
7. Her community için özet üretir.
8. Sonuçları `outputs/1example_*.json` olarak kaydeder.

### Ürettiği dosyalar

```
outputs/
├── 1example_schema.json              # Otomatik üretilen schema
├── 1example_kg.json                  # Asıl KG (nodes + edges + clusters)
├── 1example_community_reports.json   # 8 cluster özeti
└── 1example_summary.json             # Üst düzey istatistikler
```

### Tarayıcıda görmek

KG'yi interaktif graph olarak görmek için:

```bash
python examples/api_server_example.py 1example
```

Sonra tarayıcıdan: <http://localhost:8000>

Cytoscape.js tabanlı UI'da node'ları sürükleyebilir, cluster'lara
zoom yapabilir, ilişkilerin "evidence" cümlelerini görebilirsin. Server'ı
durdurmak için terminale `Ctrl+C`.

---

## 5. Kendi metninle çalıştırma

3 yol var:

### 5.1. CLI ile (en hızlısı)

```bash
drg extract benim_metnim.txt -o cikti.json
```

Tek komutla metni KG JSON'a çevirir. Schema yoksa otomatik üretir.
Detay: `drg extract --help`.

### 5.2. `inputs/` klasörüne dosya at, full_pipeline kullan

Metnini `inputs/benimproje_text.txt` adıyla kaydet, sonra:

```bash
python examples/full_pipeline_example.py benimproje
```

Tüm pipeline (clustering, reports dâhil) çalışır, çıktılar
`outputs/benimproje_*.json` olarak yazılır.

### 5.3. Python kodu içinde

```python
from drg import extract_typed, EnhancedDRGSchema, EntityType, Relation, RelationGroup

schema = EnhancedDRGSchema(
    entity_types=[
        EntityType(name="Person", description="Bir kişi"),
        EntityType(name="Company", description="Bir şirket"),
    ],
    relation_groups=[
        RelationGroup(
            name="iş_ilişkileri",
            description="Çalışma ilişkileri",
            relations=[
                Relation("works_at", "Person", "Company"),
            ],
        ),
    ],
)

metin = "Tim Cook Apple'da CEO olarak çalışıyor."
entities, triples = extract_typed(metin, schema)

print(entities)  # [("Tim Cook", "Person"), ("Apple", "Company")]
print(triples)   # [("Tim Cook", "works_at", "Apple")]
```

Daha karmaşık örnekler: [`examples/quickstarts/`](examples/quickstarts/).

---

## 6. Çıktıyı anlamak

`outputs/1example_kg.json` aşağı yukarı şöyle:

```json
{
  "nodes": [
    { "id": "Apple Inc.", "type": "Company" },
    { "id": "iPhone", "type": "Product" },
    { "id": "Tim Cook", "type": "Person" }
  ],
  "edges": [
    {
      "source": "Apple Inc.",
      "target": "iPhone",
      "relationship_type": "PRODUCES",
      "relationship_detail": "Apple manufactures iPhone",
      "metadata": {
        "evidence": "Apple's product portfolio includes the iPhone..."
      }
    }
  ],
  "clusters": [
    {
      "id": "cluster_0",
      "node_ids": ["iPhone", "iPad", "MacBook Pro"],
      "metadata": { "algorithm": "louvain" }
    }
  ]
}
```

- **`nodes`**: KG'deki her bir varlık. `type` alanı schema'dan gelir.
- **`edges`**: İki varlık arası ilişki. `evidence` metnin hangi cümlesinden
  bu ilişkinin çıkarıldığını gösterir — provenance/güven için kritik.
- **`clusters`**: Louvain'in bulduğu topluluklar. Her cluster, birbirine
  daha sıkı bağlı node'ların kümesi.

`community_reports.json` her cluster için istatistik (top actors,
top relationships, density). Şu an narrative LLM özeti içermiyor (alpha
limiti — bkz. README "Known Limitations").

---

## 7. Sık karşılaşılan sorunlar

### "OPENAI_API_KEY bulunamadı" uyarısı

`.env`'de `DRG_MODEL=gemini/...` olduğundan emin ol. Eğer "openai/gpt-4o-mini"
seçili görünüyorsa (default), `.env` ya yüklenmiyor ya `DRG_MODEL` env'de
boş/yanlış set edilmiş.

```bash
echo $DRG_MODEL              # Boşsa .env okunmamış
cat .env | grep DRG_MODEL    # Doğru satırı görmeli
```

### "API key expired" / 400 INVALID_ARGUMENT

Key süresi dolmuş veya format yanlış. [AI Studio](https://aistudio.google.com/apikey)'den
yeni key oluştur.

> Yeni Google AI Studio key'leri `AQ.` ile başlıyor (53 karakter,
> Service-Account-bound). Eskiler `AIzaSy...` (39 karakter). İkisi de
> geçerli; bazı eski client'lar `AQ.` formatını kabul etmiyor olabilir,
> LiteLLM güncel sürümde sorun yok.

### 429 RESOURCE_EXHAUSTED / "quota exceeded"

Free tier limitlerinden birine takıldın:

| Limit | gemini-2.5-flash | gemini-2.0-flash |
|---|---|---|
| RPM (req/dakika) | 10 | 15 |
| TPM (token/dakika input) | 250K | 1M |
| RPD (req/gün) | 250 | 1500 |

- **TPM/RPM dolduysa:** 1-2 dk bekle.
- **RPD dolduysa:** Pasifik gece yarısına kadar bekle (İstanbul'da ~10:00
  sabah). Veya Ollama'ya geç (lokal, kotasız).

### "JSON parse failed" / truncated schema

LLM cevabı `DRG_MAX_TOKENS` sınırına takılıp kesilmiş. `.env`'de bu
değeri artır:

```dotenv
DRG_MAX_TOKENS=12000
```

### Pipeline çok yavaş / pahalı

- Daha hızlı model: `gemini/gemini-2.0-flash` (2.5'tan hızlı, kalite biraz
  daha düşük).
- Tamamen offline: Ollama (bkz. aşağıdaki bölüm).

### Ollama kullanma (lokal, ücretsiz, sınırsız)

```bash
brew install ollama
ollama serve &
ollama pull llama3.1:8b
```

`.env`:

```dotenv
DRG_MODEL=ollama_chat/llama3.1:8b
DRG_BASE_URL=http://localhost:11434
GEMINI_API_KEY=
```

Mac'in M-series ise 8B modeller akıcı. Kalite Gemini'den düşük ama
prototip / development için yeterli.

---

## 8. Bilinen kısıtlar (alpha)

DRG-KG şu an **v0.1.0a1 alpha**. Tamamlanmış değil. Tam liste:
[`README.md` Known Limitations](README.md#known-limitations-v010a1).

Özet:

- Schema'da örnek olarak verdiğin ama metinde de geçen entity'ler bazen
  çıkmıyor (recall %100 değil).
- Sayısal/tarihsel fact'ler (revenue, kuruluş tarihi vs.) çıkarılmıyor.
- Soyut isimler bazen yanlış tipleniyor ("hardware sales" → Product).
- Bazı ters yönlü ilişkiler tekrar ediliyor (`RUNS_ON` ↔ `POWERS_DEVICE`).
- Community report'lar şu an istatistik şablonu, LLM-özet değil.

Bunlar v0.1.0a2'de düzeltilmek üzere takipte (bkz.
[`STATUS.md`](STATUS.md) bölüm 8).

---

## 9. Daha fazla okuma

| Dosya | Ne anlatıyor |
|---|---|
| [`README.md`](README.md) | Tam teknik referans (CLI, API, model listesi) |
| [`README.tr.md`](README.tr.md) | README'nin Türkçe teknik versiyonu |
| [`STATUS.md`](STATUS.md) | Proje durumu, gap analizi, roadmap |
| [`CHANGELOG.md`](CHANGELOG.md) | Sürüm notları |
| [`docs/project_overview.md`](docs/project_overview.md) | Mimari ve felsefe |
| [`examples/quickstarts/`](examples/quickstarts/) | 3 farklı domain'de çalışan örnekler |
| [`examples/full_pipeline_example.py`](examples/full_pipeline_example.py) | Tüm aşamaları tek dosyada gösteren referans |

---

## 10. Yardım / katkı

- Bug / öneri: [GitHub Issues](https://github.com/helindincel/drg-kg/issues)
- Pull request'lere açık (önce `STATUS.md`'deki "suggested order of attack"
  bölümüne bakman önerilir).
- Kütüphanenin lisansı MIT; ticari/akademik kullanımda özgürsün.

İyi çalışmalar!
