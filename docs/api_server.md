# DRG — API Server Kılavuzu

DRG, üretilen Knowledge Graph'ı incelemek için FastAPI tabanlı bir REST + UI
sunucusu içerir. Bu doküman onun kullanımını anlatır.

> **Tasarım notu (önemli):** UI'daki query endpoint'i **deterministic KG
> lookup** yapar — LLM ile cevap üretmez. DRG bilinçli olarak bir RAG/serving
> framework değildir (`docs/project_overview.md` §2 ve §7).

---

## 1. Kurulum

API server opsiyonel bir extra olarak gelir:

```bash
pip install -e ".[api]"
```

---

## 2. Çalıştırma

```bash
# Varsayılan örnek (1example) — inputs/1example_text.txt
python examples/api_server_example.py

# Belirli bir örnek
python examples/api_server_example.py 3
python examples/api_server_example.py 4

# Env ile
DRG_EXAMPLE=3example python examples/api_server_example.py
```

API key gerektiren LLM tabanlı pipeline'ı çalıştırmadan UI'ı denemek istersen,
önce mock mode ile bir KG üret, sonra server'ı aç (örnek script aynı işi yapar).

---

## 3. UI ve Endpoint'ler

| URL | Açıklama |
|-----|----------|
| http://localhost:8000 | Cytoscape tabanlı interaktif KG UI |
| http://localhost:8000/docs | OpenAPI / Swagger dokümantasyonu |
| `GET /api/graph` | Tam graph datası |
| `GET /api/graph/stats` | Graph istatistikleri |
| `GET /api/communities` | Tüm community/cluster verileri |
| `GET /api/communities/{cluster_id}` | Belirli bir community raporu |
| `GET /api/visualization/{format}` | `cytoscape` \| `vis-network` \| `d3` |
| `GET /api/visualization/communities/{format}` | Cluster renk kodlamalı view |

### Query endpoint'i

`POST /api/query` çağrısı **LLM kullanmaz**. Yalnızca:

- entity string matching
- (opsiyonel) relation filter
- seed entity etrafında neighborhood expansion

yapar. RAG/Generation isteyen kullanıcılar için DRG uygun değildir.

---

## 4. Neo4j Entegrasyonu (Opsiyonel)

```bash
pip install -e ".[neo4j]"

export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
```

İlgili endpoint'ler:

- `POST /api/neo4j/sync` — KG'yi Neo4j'e senkronize et
- `GET /api/neo4j/stats` — Neo4j tarafındaki istatistikler

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
