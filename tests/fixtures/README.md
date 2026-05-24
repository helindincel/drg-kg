# Test Fixtures

Bu klasör, deterministic regression test'leri için kullanılan **küçük, elle
hazırlanmış** referans veri dosyalarını içerir. Burada üretilmiş gerçek LLM
çıktıları **tutulmamalıdır**; canlı extraction çıktıları `outputs/` klasörüne
yazılır ve gitignored'dır.

## Dosyalar

- `minimal_schema.json` — En küçük geçerli `EnhancedDRGSchema` örneği.
  Schema parsing, serialization ve doğrulama testlerinde kullanılır.
- `minimal_kg.json` — `minimal_schema.json` ile üretilebilecek olası bir
  Knowledge Graph'ın elle yazılmış referansı. KG roundtrip ve `KG.from_typed`
  semantik test'leri için kullanılır.

## Kullanım

```python
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())
```

## Genişletme Kuralları

- Fixture'lar **küçük** ve **anlamlı** olmalı. Tek bir özelliği test etmek için
  uygun olmalı.
- Gerçek LLM çıktısını fixture olarak commit etme; bu repo deterministic kalmalı.
- Şema değişirse fixture'ları **manuel** olarak güncelle (otomatik regen yok).
