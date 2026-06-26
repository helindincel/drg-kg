# Open Source Launch Checklist

Use this checklist for the TestPyPI dry-run and the public announcement flow.

## TestPyPI

1. Confirm the working tree contains no secrets:
   ```bash
   git status --short
   ```

2. Build and check package metadata locally:
   ```bash
   python -m pip install --upgrade build twine
   python -m build
   python -m twine check dist/*
   ```

3. Verify package contents before upload:
   ```bash
   python - <<'PY'
   import zipfile
   from pathlib import Path

   wheel = next(Path("dist").glob("*.whl"))
   names = set(zipfile.ZipFile(wheel).namelist())
   assert "drg/py.typed" in names
   assert "drg/api/templates/index.html" in names
   print("wheel contents ok:", wheel)
   PY
   ```

4. Run the GitHub release workflow manually with `target=testpypi`.

5. Smoke install from TestPyPI in a clean environment:
   ```bash
   python -m venv /tmp/drg-testpypi
   source /tmp/drg-testpypi/bin/activate
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "drg-kg[all]"
   python -c "import drg; print(drg.__version__)"
   drg --help
   python -m drg.mcp_server --help
   python examples/query_layer_example.py
   ```

6. If an LLM provider is configured, run one small extraction smoke:
   ```bash
   drg extract inputs/1example_text.txt --auto-schema --output-format enhancedkg -o /tmp/drg-smoke-kg.json
   drg validate /tmp/drg-smoke-kg.json
   ```

## Demo Script

1. Start the API/UI demo:
   ```bash
   python examples/api_server_example.py
   ```

2. Open `http://localhost:8000`, load the full graph, then show query and
   community views.

3. If Neo4j credentials are configured, show the safe preview first:
   ```bash
   curl -X POST "http://localhost:8000/api/neo4j/sync?dry_run=true"
   ```

4. Show MCP without live extraction by running:
   ```bash
   python examples/mcp_demo.py
   ```

5. For Cursor or Claude, use the stdio config in `docs/mcp_integration.md` and
   demonstrate `drg_define_schema`, `drg_build_kg`, and `drg_export_kg`.

## LinkedIn Post Outline

- One-sentence problem: deterministic knowledge graphs from unstructured text.
- Feature chain: extract, visualize, query, MCP, Neo4j export, incremental update.
- Short demo screenshot or GIF from the built-in UI.
- Call out that MCP and Neo4j demos can run without a live LLM by using the
  deterministic tool/demo paths.
- Link to GitHub, docs, and TestPyPI/PyPI package once available.
