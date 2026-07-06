# Community Launch Checklist

Use after the first PyPI release (`drg-kg` on https://pypi.org/project/drg-kg/).

## GitHub repository

- [ ] **About** description: "Schema-driven Knowledge Graph extraction framework (DSPy)"
- [ ] **Topics:** `knowledge-graph`, `knowledge-graph-extraction`, `dspy`, `nlp`,
  `extraction`, `graph`, `python`
- [ ] **Website:** PyPI project URL or GitHub Pages (MkDocs) when enabled
- [ ] Pin **GitHub Release** `v0.1.1` with notes from `CHANGELOG.md`

## Demo assets

Capture before announcing:

1. **API UI screenshot** — run `python examples/api_server_example.py`, load graph,
   show entity inspector with provenance panel.
2. **Optional GIF** — query panel + graph zoom (keep under 5 MB for README).
3. Store under `docs/assets/` (add to git; not shipped in the wheel).

Deterministic demo (no API key):

```bash
pip install drg-kg
python examples/query_layer_example.py
```

## Announcement outline

**Problem (one sentence):** Turn unstructured text into explainable, queryable
knowledge graphs with declarative schemas.

**Highlights:**

- Schema-first extraction with DSPy
- `EnhancedKG` lifecycle: provenance, versioning, evaluation, Neo4j export
- CLI, FastAPI UI, and MCP server
- Install: `pip install "drg-kg[extract]"`

**Links:**

- GitHub: https://github.com/helindincel/drg-kg
- PyPI: https://pypi.org/project/drg-kg/
- Docs: https://github.com/helindincel/drg-kg/tree/main/docs

## MkDocs / GitHub Pages (v0.2)

```bash
pip install mkdocs
mkdocs serve
mkdocs gh-deploy
```

Configuration: [`mkdocs.yml`](../mkdocs.yml)
