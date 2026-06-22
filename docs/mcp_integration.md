# MCP Entegrasyonu

DRG, resmi MCP Python SDK üzerine kurulu bir server sağlar. Yeni entegrasyonlarda
önerilen yüzey `drg.mcp_server` modülüdür; eski `drg.mcp_api` JSON-RPC shim'i
yalnızca geriye dönük contract testleri ve migration için korunur.

## Kurulum

```bash
pip install "drg-kg[mcp]"
```

Geliştirme ortamında:

```bash
pip install -e ".[mcp]"
```

## Çalıştırma

Stdio transport, Claude Desktop ve çoğu MCP client için varsayılan yoldur:

```bash
python -m drg.mcp_server
```

HTTP/SSE transport:

```bash
python -m drg.mcp_server --http --port 8765
```

## Cursor ve Claude Bağlantısı

Yerel geliştirme checkout'ından çalıştıracaksan önce MCP extra'yı kur:

```bash
pip install -e ".[mcp]"
```

Cursor için repo veya kullanıcı seviyesindeki MCP config'e şu server'ı ekle:

```json
{
  "mcpServers": {
    "drg-kg": {
      "command": "python",
      "args": ["-m", "drg.mcp_server"],
      "cwd": "/absolute/path/to/drg-kg",
      "env": {
        "DRG_MODEL": "openai/gpt-4o-mini",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

Claude Desktop için aynı stdio server şu formatla kullanılabilir:

```json
{
  "mcpServers": {
    "drg-kg": {
      "command": "python",
      "args": ["-m", "drg.mcp_server"],
      "cwd": "/absolute/path/to/drg-kg",
      "env": {
        "DRG_MODEL": "openai/gpt-4o-mini",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

Paketlenmiş kurulumda `cwd` zorunlu değildir; editable checkout kullanırken
repo kökünü vermek import ve `.env` davranışını daha öngörülebilir yapar.
Canlı extraction tool'u (`drg_extract`) DSPy/LLM konfigürasyonu ister.
`drg_define_schema`, `drg_build_kg`, `drg_get_kg`, `drg_export_kg`,
`drg_list_schemas` ve `drg_list_kgs` ise LLM çağrısı yapmadan contract demo
için kullanılabilir.

## Araçlar

Server şu MCP tool'larını expose eder:

| Tool | Açıklama |
| --- | --- |
| `drg_define_schema` | Legacy veya Enhanced DRG schema tanımlar. |
| `drg_extract` | Kayıtlı schema ile metinden entity/triple çıkarır. |
| `drg_build_kg` | Entity/triple listesinden `EnhancedKG` oluşturur. |
| `drg_get_kg` | Kayıtlı graph'ı JSON olarak döndürür. |
| `drg_export_kg` | Graph'ı `json`, `jsonld` veya `enriched` formatta export eder. |
| `drg_list_schemas` | Kayıtlı schema'ları listeler. |
| `drg_list_kgs` | Kayıtlı knowledge graph'ları listeler. |

## Python'dan Embed Etme

```python
from drg.mcp_server import create_mcp_server

server = create_mcp_server()
server.run()  # stdio
```

HTTP/SSE:

```python
from drg.mcp_server import create_mcp_server

server = create_mcp_server()
server.run(transport="sse")
```

## Tool Payload Örnekleri

Enhanced schema:

```json
{
  "schema_id": "company_schema",
  "schema": {
    "entity_types": [
      {"name": "Company", "description": "Business organizations"},
      {"name": "Product", "description": "Commercial products"}
    ],
    "relation_groups": [
      {
        "name": "commercial",
        "relations": [
          {"name": "produces", "src": "Company", "dst": "Product"}
        ]
      }
    ]
  }
}
```

KG build payload:

```json
{
  "kg_id": "demo_kg",
  "entities": [["Apple", "Company"], ["iPhone", "Product"]],
  "triples": [["Apple", "produces", "iPhone"]]
}
```

## Migration Notu: `drg.mcp_api`

`drg.mcp_api`, DRG'nin MCP öncesi JSON-RPC benzeri adapter'ıdır. Şu durumlar
dışında yeni kodda kullanılmamalıdır:

- Eski in-process contract testleri.
- Mevcut JSON-RPC payload'larını kısa süreli koruma ihtiyacı.
- Migration sırasında eski agent entegrasyonlarını karşılaştırma.

Yeni entegrasyonlarda `python -m drg.mcp_server` veya `create_mcp_server()`
kullanılmalıdır.

## Operasyonel Notlar

- Server state'i process içi memory'dedir; schema ve KG kayıtları server
  oturumu bitince kaybolur.
- `drg_extract` canlı DSPy/LLM extraction yolunu kullanır; provider API key veya
  lokal model konfigürasyonu gerekir.
- `drg_build_kg`, `drg_get_kg` ve `drg_export_kg` LLM çağrısı yapmaz; contract
  testleri için deterministik yüzeydir.
- Çok kullanıcılı production senaryosunda persistent storage, auth ve concurrency
  stratejisi ayrıca tasarlanmalıdır.
