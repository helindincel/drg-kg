# MCP Entegrasyonu: Agent Arayüz Tasarımı

## Genel Bakış

DRG, AI agent'larının sistem ile etkileşim kurabilmesi için Model Context Protocol (MCP) tarzı bir API arayüzü sağlar. Bu arayüz, agent'ların DRG'nin knowledge graph extraction yeteneklerini programatik olarak kullanmasına olanak tanır.

## Tasarım Prensipleri

### 1. MCP-Style API

MCP API, JSON-RPC 2.0 protokolünü temel alan bir request/response formatı kullanır:

- **Standardized Format**: Tüm istekler ve yanıtlar tutarlı bir JSON formatında
- **Error Handling**: Detaylı hata kodları ve mesajları
- **Type Safety**: Parametreler ve dönüş değerleri için net tip tanımlamaları
- **Extensibility**: Yeni metodlar kolayca eklenebilir

### 2. Agent-Centric Design

API, agent'ların ihtiyaçlarını ön planda tutar:

- **Declarative Operations**: Agent'lar ne istediklerini tanımlar, sistem nasıl yapılacağını halleder
- **State Management**: Schema ve knowledge graph'lar ID'lerle yönetilir
- **Multiple Formats**: Export işlemleri için çoklu format desteği (JSON, JSON-LD, enriched)

### 3. Modular Architecture

MCP API, DRG'nin mevcut modüllerini sarmalar:

- **Schema Management**: Schema tanımlama ve yönetimi
- **Extraction Pipeline**: Entity ve relation extraction
- **Knowledge Graph Building**: Graph oluşturma ve yönetimi
- **Export Capabilities**: Farklı formatlarda export

## API Referansı

### Request Format

Tüm istekler şu formatta olmalıdır:

```json
{
  "jsonrpc": "2.0",
  "method": "drg/<method_name>",
  "params": {
    // Method-specific parameters
  },
  "id": 1
}
```

### Response Format

Başarılı yanıtlar:

```json
{
  "jsonrpc": "2.0",
  "result": {
    // Method-specific result data
  },
  "id": 1
}
```

Hata yanıtları:

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": "error_code",
    "message": "Error message",
    "data": {
      // Optional additional error data
    }
  },
  "id": 1
}
```

### Error Codes

- `invalid_request`: İstek formatı geçersiz
- `method_not_found`: Belirtilen metod bulunamadı
- `invalid_params`: Parametreler geçersiz veya eksik
- `internal_error`: Sistem içi hata
- `schema_error`: Schema ile ilgili hata
- `extraction_error`: Extraction işlemi sırasında hata

## Metodlar

### 1. `drg/list_tools`

Sistemin sağladığı tüm araçları/metodları listeler.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "drg/list_tools",
  "params": {},
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "drg/define_schema",
        "description": "Define a schema for entity and relation extraction",
        "inputSchema": {
          // JSON Schema for parameters
        }
      },
      // ... diğer araçlar
    ]
  },
  "id": 1
}
```

### 2. `drg/define_schema`

Entity ve relation extraction için bir schema tanımlar.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "drg/define_schema",
  "params": {
    "schema_id": "my_schema",
    "schema": {
      "entities": [
        {"name": "Company"},
        {"name": "Product"}
      ],
      "relations": [
        {"name": "produces", "src": "Company", "dst": "Product"}
      ]
    }
  },
  "id": 2
}
```

**Enhanced Schema (Opsiyonel):**
```json
{
  "jsonrpc": "2.0",
  "method": "drg/define_schema",
  "params": {
    "schema_id": "enhanced_schema",
    "schema": {
      "entity_types": [
        {
          "name": "Company",
          "description": "Business organizations",
          "examples": ["Apple", "Google"],
          "properties": {}
        }
      ],
      "relation_groups": [
        {
          "name": "production",
          "description": "Company-product relationships",
          "relations": [
            {"name": "produces", "src": "Company", "dst": "Product"}
          ]
        }
      ]
    }
  },
  "id": 2
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "schema_id": "my_schema",
    "status": "defined",
    "entity_types": {
      "entity_types": ["Company", "Product"],
      "relations": 1
    }
  },
  "id": 2
}
```

### 3. `drg/extract`

Metinden entity ve relation'ları çıkarır.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "drg/extract",
  "params": {
    "text": "Apple released the iPhone 16 in September 2025.",
    "schema_id": "my_schema"
  },
  "id": 3
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "entities": [
      {"name": "Apple", "type": "Company"},
      {"name": "iPhone 16", "type": "Product"}
    ],
    "triples": [
      {"source": "Apple", "relation": "produces", "target": "iPhone 16"}
    ],
    "counts": {
      "entities": 2,
      "triples": 1
    }
  },
  "id": 3
}
```

### 4. `drg/build_kg`

Çıkarılan entity ve relation'lardan bir knowledge graph oluşturur.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "drg/build_kg",
  "params": {
    "kg_id": "my_kg",
    "entities": [
      ["Apple", "Company"],
      ["iPhone 16", "Product"]
    ],
    "triples": [
      ["Apple", "produces", "iPhone 16"]
    ]
  },
  "id": 4
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "kg_id": "my_kg",
    "status": "built",
    "statistics": {
      "total_nodes": 2,
      "total_edges": 1,
      "total_clusters": 0,
      "node_types_count": {
        "Company": 1,
        "Product": 1
      },
      "edge_types_count": {
        "produces": 1
      }
    }
  },
  "id": 4
}
```

### 5. `drg/get_kg`

Bir knowledge graph'ı ID ile alır.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "drg/get_kg",
  "params": {
    "kg_id": "my_kg"
  },
  "id": 5
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "nodes": [
      {
        "id": "Apple",
        "type": "Company",
        "properties": {},
        "metadata": {}
      },
      {
        "id": "iPhone 16",
        "type": "Product",
        "properties": {},
        "metadata": {}
      }
    ],
    "edges": [
      {
        "source": "Apple",
        "target": "iPhone 16",
        "relationship_type": "produces",
        "relationship_detail": "Apple produces iPhone 16",
        "metadata": {}
      }
    ],
    "clusters": []
  },
  "id": 5
}
```

### 6. `drg/export_kg`

Knowledge graph'ı belirtilen formatta export eder.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "drg/export_kg",
  "params": {
    "kg_id": "my_kg",
    "format": "json"
  },
  "id": 6
}
```

**Desteklenen Formatlar:**
- `json`: Standart JSON formatı
- `jsonld`: JSON-LD formatı (semantic web uyumlu)
- `enriched`: Zenginleştirilmiş export (entities + relationships + communities)

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "format": "json",
    "data": {
      // Format-specific data structure
    }
  },
  "id": 6
}
```

### 7. `drg/list_schemas`

Tanımlı tüm schema'ları listeler.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "drg/list_schemas",
  "params": {},
  "id": 7
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "schemas": [
      {
        "schema_id": "my_schema",
        "summary": {
          "entity_types": ["Company", "Product"],
          "relations": 1
        }
      }
    ]
  },
  "id": 7
}
```

### 8. `drg/get_schema`

Bir schema'yı ID ile alır.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "drg/get_schema",
  "params": {
    "schema_id": "my_schema"
  },
  "id": 8
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "schema_id": "my_schema",
    "type": "legacy",
    "entities": [
      {"name": "Company"},
      {"name": "Product"}
    ],
    "relations": [
      {"name": "produces", "src": "Company", "dst": "Product"}
    ]
  },
  "id": 8
}
```

## Kullanım Senaryoları

### Senaryo 1: Basit Entity Extraction

1. Schema tanımla (`drg/define_schema`)
2. Metinden extract et (`drg/extract`)
3. Knowledge graph oluştur (`drg/build_kg`)
4. Export et (`drg/export_kg`)

### Senaryo 2: Çoklu Schema Yönetimi

1. Birden fazla schema tanımla
2. Farklı metinler için farklı schema'lar kullan
3. Her schema için ayrı knowledge graph'lar oluştur
4. Schema'ları listele ve yönet (`drg/list_schemas`, `drg/get_schema`)

### Senaryo 3: Agent Workflow Integration

1. Agent, domain'e özgü schema tanımlar
2. Agent, kullanıcı girdisini veya dokümanları işler
3. Agent, extraction sonuçlarını analiz eder
4. Agent, knowledge graph'ı farklı formatlarda export eder
5. Agent, sonuçları kullanıcıya veya başka sistemlere iletir

## Python Kullanımı

### Basit Kullanım

```python
from drg.mcp_api import DRGMCPAPI, MCPRequest, create_mcp_api

# API instance oluştur
api = create_mcp_api()

# Schema tanımla
request = MCPRequest(
    method="drg/define_schema",
    params={
        "schema_id": "company_schema",
        "schema": {
            "entities": [{"name": "Company"}, {"name": "Product"}],
            "relations": [{"name": "produces", "src": "Company", "dst": "Product"}],
        },
    },
    id=1,
)
response = api.handle_request(request)
print(response.result)

# Extract
extract_request = MCPRequest(
    method="drg/extract",
    params={
        "text": "Apple released the iPhone 16.",
        "schema_id": "company_schema",
    },
    id=2,
)
response = api.handle_request(extract_request)
print(response.result)
```

### Gelişmiş Kullanım

```python
from drg.mcp_api import DRGMCPAPI, MCPRequest

api = DRGMCPAPI()

# Enhanced schema tanımla
schema_request = MCPRequest(
    method="drg/define_schema",
    params={
        "schema_id": "enhanced_schema",
        "schema": {
            "entity_types": [
                {
                    "name": "Company",
                    "description": "Business organizations",
                    "examples": ["Apple", "Google"],
                }
            ],
            "relation_groups": [
                {
                    "name": "production",
                    "description": "Company-product relationships",
                    "relations": [
                        {"name": "produces", "src": "Company", "dst": "Product"}
                    ],
                }
            ],
        },
    },
    id=1,
)
response = api.handle_request(schema_request)

# Full pipeline
text = "Apple released the iPhone 16. Samsung produces the Galaxy S24."
extract_response = api.handle_request(
    MCPRequest(
        method="drg/extract",
        params={"text": text, "schema_id": "enhanced_schema"},
        id=2,
    )
)

if extract_response.result:
    entities = [[e["name"], e["type"]] for e in extract_response.result["entities"]]
    triples = [
        [t["source"], t["relation"], t["target"]]
        for t in extract_response.result["triples"]
    ]
    
    build_response = api.handle_request(
        MCPRequest(
            method="drg/build_kg",
            params={"kg_id": "my_kg", "entities": entities, "triples": triples},
            id=3,
        )
    )
    
    export_response = api.handle_request(
        MCPRequest(
            method="drg/export_kg",
            params={"kg_id": "my_kg", "format": "enriched"},
            id=4,
        )
    )
```

## Entegrasyon Notları

### State Management

MCP API, schema'lar ve knowledge graph'lar için in-memory state management kullanır. Production ortamında, bu state'in persistent storage'a kaydedilmesi gerekebilir.

### Error Handling

Tüm metodlar, hata durumlarında detaylı error response döndürür. Agent'lar bu error response'ları parse ederek uygun aksiyon almalıdır.

### Performance Considerations

- Schema'lar ve knowledge graph'lar ID'lerle yönetilir, bu sayede büyük veri setlerinde performans korunur
- Extraction işlemleri, DSPy pipeline'ı kullandığı için LLM API çağrıları gerektirir
- Knowledge graph export işlemleri, graph boyutuna bağlı olarak zaman alabilir

### Extension Points

Yeni metodlar eklemek için:

1. `DRGMCPAPI` sınıfına yeni bir `_<method_name>` metodu ekle
2. `handle_request` metodunda yeni method'u route et
3. `_list_tools` metoduna yeni tool'u ekle
4. Dokümantasyonu güncelle

## Trade-off'lar

### Artılar

- **Standardized Interface**: Tüm agent etkileşimleri tutarlı bir formatta
- **Type Safety**: Parametreler ve dönüş değerleri için net tip tanımlamaları
- **Extensibility**: Yeni metodlar kolayca eklenebilir
- **Error Handling**: Detaylı hata mesajları ve kodları

### Eksiler

- **In-Memory State**: Production ortamında persistent storage gerekebilir
- **API Overhead**: Her işlem için JSON-RPC formatında request/response oluşturulması
- **Limited Concurrency**: Şu anki implementasyon thread-safe değil (gerekirse eklenebilir)

## Gelecek Geliştirmeler

1. **Persistent Storage**: Schema ve knowledge graph'lar için database entegrasyonu
2. **Batch Operations**: Çoklu işlemleri tek request'te yapabilme
3. **Streaming Support**: Büyük extraction işlemleri için streaming response
4. **Authentication/Authorization**: Güvenlik için API key veya token desteği
5. **Caching**: Sık kullanılan schema ve extraction sonuçları için cache mekanizması
6. **WebSocket Support**: Real-time agent etkileşimleri için WebSocket desteği

