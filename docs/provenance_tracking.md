# Provenance Tracking

DRG-KG stores source metadata for graph facts under `metadata.provenance` while
keeping legacy `source_ref`, `source_documents`, and `evidence` keys for
backward compatibility.

## Metadata Contract

```json
{
  "metadata": {
    "source_ref": "doc_1",
    "source_documents": ["doc_1"],
    "evidence": "Alice works at Acme.",
    "provenance": {
      "document_id": "doc_1",
      "sentence_id": "s0",
      "source_span": [0, 20],
      "snippet": "Alice works at Acme.",
      "extracted_at": "2026-06-18T20:00:00+00:00",
      "extractor_version": "0.1.0"
    }
  }
}
```

Fields are best-effort. Existing graphs without `metadata.provenance` still
load and query correctly; query helpers fall back to the legacy fields.

## Querying Evidence

```python
from drg.query import GraphQuery

gq = GraphQuery.from_json("outputs/global_kg.json")
bundle = gq.evidence_for("Alice", "works_at", "Acme")
print(bundle.source_documents)
print(bundle.evidence[0].snippet)
```

## API

```bash
curl "http://localhost:8000/api/provenance/entity/Alice"
curl "http://localhost:8000/api/provenance/edge?source=Alice&relationship_type=works_at&target=Acme"
```

The API never stores request secrets such as `api_key` in provenance metadata.
