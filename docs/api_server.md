# DRG — API Server Kılavuzu

DRG, üretilen Knowledge Graph'ı incelemek için FastAPI tabanlı bir REST + UI
sunucusu içerir. Bu doküman onun kullanımını anlatır.

> **Tasarım notu (önemli):** UI'daki query endpoint'i **deterministic KG
> lookup** yapar; LLM ile cevap üretmez (`docs/project_overview.md` §2 ve §7).

> **UI hedefi:** Bu yüzey bir production dashboard değil, read-only KG inspection
> aracıdır. Öncelik node/edge arama, filtreleme, metadata inceleme, export ve
> boş/hata durumlarını anlaşılır göstermektir.

---

## 1. Kurulum

API server opsiyonel bir extra olarak gelir:

```bash
pip install -e ".[api]"
```

`POST /api/extract` veya `POST /api/graph/update` gibi DSPy/LLM destekli
extraction endpoint'lerini kullanacaksan extraction extra'sını da kur:

```bash
pip install -e ".[api,extract]"
```

---

## 2. Çalıştırma

```bash
# Varsayılan örnek (input2) — outputs/output2_kg.json
python examples/api_server_example.py

# Belirli bir örnek
python examples/api_server_example.py 1
python examples/api_server_example.py 2

# Env ile
DRG_EXAMPLE=1 python examples/api_server_example.py
DRG_EXAMPLE=2 python examples/api_server_example.py
```

API key gerektiren LLM tabanlı pipeline'ı çalıştırmadan UI'ı denemek istersen,
önce mock mode ile bir KG üret, sonra server'ı aç (örnek script aynı işi yapar).

---

## 3. UI ve Endpoint'ler

| URL | Açıklama |
|-----|----------|
| http://localhost:8000 | Cytoscape tabanlı interaktif KG UI |
| http://localhost:8000/docs | OpenAPI / Swagger dokümantasyonu |
| `GET /healthz` | Liveness probe |
| `GET /readyz` | KG ve opsiyonel Neo4j config readiness durumu |
| `GET /api/graph` | Tam graph datası |
| `GET /api/graph/stats` | Graph istatistikleri |
| `POST /api/extract` | Metinden KG çıkar ve opsiyonel olarak server state'e yükle |
| `GET /api/communities` | Tüm community/cluster verileri |
| `GET /api/communities/{cluster_id}` | Belirli bir community raporu |
| `GET /api/visualization/{format}` | `cytoscape` \| `vis-network` \| `d3` |
| `GET /api/visualization/communities/{format}` | Cluster renk kodlamalı view |

UI tarafında beklenen temel inceleme akışı:

- graph istatistiklerini kontrol et
- node veya edge araması yap
- büyük graph'larda filtreleri kullan
- seçili node/edge metadata panelini incele
- gerekiyorsa graph JSON/export endpoint'leriyle çıktıyı indir

### Query endpoint'i

`POST /api/query` çağrısı **LLM kullanmaz**. Yalnızca:

- entity string matching
- (opsiyonel) relation filter
- seed entity etrafında neighborhood expansion

yapar. Generation isteyen kullanıcılar için ayrı bir uygulama katmanı gerekir.

### Extract endpoint'i

`POST /api/extract`, CLI'daki text → extraction → `EnhancedKG` yolunun REST
karşılığıdır. Bu endpoint DSPy/LLM provider kullanır; cloud model için API key
veya lokal model konfigürasyonu ve `[extract]` extra'sı gerekir.

```bash
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "TechCorp was founded by Jane Doe.",
    "schema": {
      "entity_types": [
        {"name": "Company", "description": "Companies"},
        {"name": "Person", "description": "People"}
      ],
      "relation_groups": [
        {
          "name": "founding",
          "relations": [
            {"name": "founded_by", "src": "Company", "dst": "Person"}
          ]
        }
      ]
    },
    "store_graph": true
  }'
```

Yanıt `entities`, `triples`, `counts` ve `graph` alanlarını döndürür.
`store_graph: true` ise sonuç `/api/graph`, `/api/graph/stats` ve UI
tarafından hemen kullanılabilir hale gelir.

---

## 4. Neo4j Entegrasyonu (Opsiyonel)

```bash
pip install -e ".[neo4j]"

export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
```

İlgili endpoint'ler:

- `GET /api/neo4j/test` — config ve bağlantı testi
- `POST /api/neo4j/sync` — KG'yi Neo4j'e senkronize et
- `POST /api/neo4j/sync?dry_run=true` — yazma yapmadan sync planını gör
- `GET /api/neo4j/stats` — Neo4j tarafındaki istatistikler

### Demo akışı

Önce API server'ı örnek KG ile aç:

```bash
python examples/api_server_example.py
```

Sonra ayrı bir terminalde bağlantı ve planı kontrol et:

```bash
curl http://localhost:8000/api/neo4j/test

curl -X POST "http://localhost:8000/api/neo4j/sync?dry_run=true"
```

`dry_run=true` hiçbir yazma yapmaz; node, edge, cluster ve ilişki tipi sayısını
döndüren bir sync planı üretir. Demo veya ilk kurulumda bu adımı geçmeden gerçek
sync çalıştırma.

Plan doğru görünüyorsa gerçek sync:

```bash
curl -X POST "http://localhost:8000/api/neo4j/sync"
```

`clear_existing=true` yalnızca disposable demo database üzerinde kullanılmalıdır;
mevcut graph datasını temizleyebilir. Public demo için ayrı bir Neo4j database
ve önce `dry_run=true` önerilir.

---

## 5. Hub-Mitigation (UI-only)

Bazı metinler tek merkezli (star-shape) graph üretir. UI'da bunun için
**proxy-node temelli anti-hub** seçeneği vardır. Bu özellik KG datasını
**değiştirmez**, sadece görsel layout'u rahatlatır
(`drg/graph/hub_mitigation.py`).

---

## 6. Yaygın Sorunlar

### Port zaten kullanımda
`api_server_example.py` içindeki `server.run(host=..., port=...)` parametresini
değiştir veya farklı bir port set et:

```python
server.run(host="0.0.0.0", port=8080)
```

### API key hatası
Embedding/extraction adımı API key gerektirir. `.env` dosyanı veya
environment variable'ları kontrol et. Bkz. `docs/setup.md`.

### Neo4j bağlantı hatası
Neo4j'i kullanmıyorsan Neo4j endpoint'leri çalışmaz; bu **normaldir** ve graph
görselleştirme için gerekli değildir.

---

## 7. Güvenlik Notları

- API key'leri **asla** kodda hardcode etme; `.env` veya secret manager kullan.
- `.env` dosyaları `.gitignore` ile dışlanmıştır; yeni `*.env` veya hassas
  dosyalar eklerken bu pattern'i bozma.
- Production'da `CORSMiddleware` ayarlarını domain'lerine göre daraltmayı unutma.
- Production gözlemlenebilirliği için `X-Request-ID` header'ı desteklenir ve
  response'a geri yazılır; upstream gateway'in ürettiği request ID korunur.
