# MCP Entegrasyonu

DRG, resmi MCP Python SDK üzerine kurulu bir server sağlar. Desteklenen MCP
yüzeyi `drg.mcp_server` modülüdür.

## Kurulum

```bash
pip install "drg-kg[mcp]"
```

`[mcp]` extra'sı MCP SDK'nın yanında DSPy-backed extraction için gereken
runtime bağımlılıklarını da kurar; `drg_extract` tool'u yine bir model
provider/API key veya lokal model konfigürasyonu ister.

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

## Operasyonel Notlar

- Server state'i process içi memory'dedir; schema ve KG kayıtları server
  oturumu bitince kaybolur.
- `drg_extract` canlı DSPy/LLM extraction yolunu kullanır; provider API key veya
  lokal model konfigürasyonu gerekir.
- `drg_build_kg`, `drg_get_kg` ve `drg_export_kg` LLM çağrısı yapmaz; contract
  testleri için deterministik yüzeydir.
- Çok kullanıcılı production senaryosunda persistent storage, auth ve concurrency
  stratejisi ayrıca tasarlanmalıdır.
