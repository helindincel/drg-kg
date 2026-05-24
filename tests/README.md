# DRG Test Suite

## Test Kategorileri

### 1. Unit Tests (Mock-based) - `test_basic.py`, `test_extract_mock.py`
- **Amaç**: Logic'i test etmek, external dependencies gerektirmez
- **API Keys**: Gerekmez
- **Çalıştırma**: `pytest tests/test_basic.py tests/test_extract_mock.py -v`

**Test Edilenler:**
- Mock-based extraction logic
- Schema validation
- KG construction from triples
- Entity resolution logic (unit level)

**Sorun**: Bu testler sadece mock'ları test ediyor, gerçek extraction işlevselliğini test etmiyor.

### 2. Functional Tests - `test_extraction_functionality.py` ⭐ YENİ
- **Amaç**: Gerçek extraction işlevselliğini test etmek
- **API Keys**: Gerekir (integration testler için)
- **Çalıştırma**: `pytest tests/test_extraction_functionality.py -v -m integration`

**Test Edilenler:**
- ✅ Entity extraction from real text
- ✅ Relation extraction from real text
- ✅ Cross-chunk relationship discovery
- ✅ Coreference resolution
- ✅ Entity resolution (similar name merging)
- ✅ Schema generation
- ✅ Negation detection
- ✅ Temporal information extraction
- ✅ Reverse relations handling
- ✅ Implicit relationships (possessive forms)
- ✅ Schema validation
- ✅ Edge case handling (empty text, no matches)

**Önemli**: Bu testler gerçek LLM çağrıları yapar ve API key gerektirir.

### 3. Integration Tests - `test_integration.py`
- **Amaç**: End-to-end pipeline testi (farklı LLM provider'ları ile)
- **API Keys**: Gerekir
- **Çalıştırma**: `pytest tests/test_integration.py -v`

**Test Edilenler:**
- OpenAI provider ile extraction
- Gemini provider ile extraction

## Test Çalıştırma

### Tüm Testleri Çalıştır (Mock Tests Only)
```bash
pytest tests/ -v
```

### Functional Tests (API Key Gerekir)
```bash
# Environment variable'ı ayarla
export OPENAI_API_KEY=sk-...
# veya
export OPENROUTER_API_KEY=sk-or-v1-...

# Functional testleri çalıştır
pytest tests/test_extraction_functionality.py -v -m integration
```

### Sadece Belirli Test Kategorisi
```bash
# Sadece unit tests (mock-based)
pytest tests/test_basic.py tests/test_extract_mock.py -v

# Sadece functional tests
pytest tests/test_extraction_functionality.py -v -m integration

# Sadece integration tests
pytest tests/test_integration.py -v
```

## Test Sorunları ve Çözümler

### Önceki Sorunlar (Düzeltildi)
1. ❌ Testler sadece mock'ları test ediyordu - gerçek işlevsellik test edilmiyordu
2. ❌ Integration testler çok basitti - sadece "entity var mı?" kontrolü yapıyordu
3. ❌ Kritik işlevler test edilmiyordu:
   - Cross-chunk relationships
   - Coreference resolution
   - Entity resolution
   - Schema generation
   - Negation detection

### Çözümler (Yeni Testler)
1. ✅ `test_extraction_functionality.py` eklendi - gerçek işlevsellik testleri
2. ✅ 13 farklı functional test eklendi
3. ✅ Kritik işlevler için detaylı testler eklendi
4. ✅ Edge case testleri eklendi

## Test Coverage

### Şu An Test Edilen İşlevler:
- ✅ Entity extraction
- ✅ Relation extraction
- ✅ Cross-chunk relationships
- ✅ Coreference resolution
- ✅ Entity resolution
- ✅ Schema generation
- ✅ Negation detection
- ✅ Temporal information
- ✅ Reverse relations
- ✅ Implicit relationships
- ✅ Schema validation
- ✅ Edge cases

### Henüz Test Edilmeyen İşlevler (İleride Eklenebilir):
- ⚠️ Confidence scores (KGEdge'de var ama extraction test edilmiyor)
- ⚠️ Temporal fields (start_date, end_date) extraction
- ⚠️ Optimizer functionality
- ⚠️ Chunking strategies
- ⚠️ Vector store operations
- ⚠️ Clustering algorithms

## Test Best Practices

1. **Mock Tests**: Hızlı, API key gerektirmez, CI/CD için uygun
2. **Functional Tests**: Gerçek işlevselliği test eder, API key gerekir, manuel test için
3. **Integration Tests**: End-to-end pipeline testi, farklı provider'lar ile

## Notlar

- Functional testler API key gerektirdiği için CI/CD pipeline'ında isteğe bağlı olabilir
- Mock testler her zaman çalıştırılmalı (fast feedback)
- Integration testler farklı provider'ları test etmek için kullanılır

