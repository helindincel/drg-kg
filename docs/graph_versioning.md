# Graph Versioning

DRG-KG supports snapshot-based versioning for persisted `EnhancedKG` JSON files.
The graph file remains the source artifact, and snapshots live beside it in a
manifest directory.

## CLI

Incremental updates create a version snapshot automatically:

```bash
drg extract docs/article_2.txt --auto-schema --update outputs/global_kg.json
drg versions list outputs/global_kg.json
drg versions diff outputs/global_kg.json v1 v2 --json
drg versions rollback outputs/global_kg.json v1
```

The manifest is stored by default under:

```text
outputs/.global_kg_versions/manifest.json
```

Each version records:

- `version_id`
- `parent_version_id`
- `created_at`
- `operation`
- `document_id`
- `diff_summary`
- `snapshot_path`

## Python API

```python
from drg.graph.versioning import create_snapshot, diff_versions, list_versions

version = create_snapshot(kg, "outputs/global_kg.json", operation="merge")
versions = list_versions("outputs/global_kg.json")
diff = diff_versions("outputs/global_kg.json", versions[0].version_id, version.version_id)
```

## REST API

The FastAPI server keeps in-memory versions for updates performed through the
API:

```bash
curl -X POST http://localhost:8000/api/graph/update \
  -H "Content-Type: application/json" \
  -d '{"text":"Alice works at Acme.","model":"ollama_chat/llama3","document_id":"doc_1"}'

curl http://localhost:8000/api/graph/versions
curl http://localhost:8000/api/graph/versions/v2/diff
curl -X POST http://localhost:8000/api/graph/versions/v1/rollback
```

## Scope

This is snapshot rollback, not a transactional multi-user storage engine.
Branching, lock management, and Neo4j as source-of-truth are intentionally left
for a later backend-focused milestone.
