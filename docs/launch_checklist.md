# Open Source Launch Checklist

Use this checklist for the TestPyPI dry-run and the public announcement flow.

Configure GitHub secrets and environments first — see
[`docs/release_setup.md`](release_setup.md).

## TestPyPI

1. Confirm the working tree contains no secrets:
   ```bash
   git status --short
   ```

2. Move the changelog entry out of `Unreleased`, choose the release tag, and
   confirm the tag format:
   - Final PyPI release: `vX.Y.Z`
   - TestPyPI prerelease: `vX.Y.ZaN`, `vX.Y.ZbN`, or `vX.Y.ZrcN`

3. Build and check package metadata locally:
   ```bash
   python -m pip install --upgrade build twine
   rm -rf dist build *.egg-info
   python -m build
   python -m twine check dist/*
   ```

4. Verify package contents before upload:
   ```bash
   python - <<'PY'
   import tarfile
   import zipfile
   from pathlib import Path

   wheel = next(Path("dist").glob("*.whl"))
   sdist = next(Path("dist").glob("*.tar.gz"))

   wheel_names = set(zipfile.ZipFile(wheel).namelist())
   with tarfile.open(sdist) as archive:
       sdist_names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}

   required = {
       "drg/py.typed",
       "drg/api/templates/index.html",
       "LICENSE",
       "README.md",
       "pyproject.toml",
   }
   forbidden_prefixes = (
       ".github/",
       "docs/",
       "examples/",
       "inputs/",
       "outputs/",
       "reports/",
       "scripts/",
       "tests/",
   )

   assert not (required & {"drg/py.typed", "drg/api/templates/index.html"} - wheel_names)
   assert not (required - sdist_names)
   assert not [name for name in sdist_names if name.startswith(forbidden_prefixes)]
   print("distribution contents ok:", wheel, sdist)
   PY
   ```

5. Run the GitHub release workflow manually with `target=testpypi`, or push a
   prerelease tag. Prerelease tags must stop at TestPyPI; only final `vX.Y.Z`
   tags may continue to PyPI.

6. Smoke install from TestPyPI in a clean environment:
   ```bash
   python -m venv /tmp/drg-testpypi
   source /tmp/drg-testpypi/bin/activate
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "drg-kg[all]"
   python -c "import drg; print(drg.__version__)"
   drg --help
   python -m drg.mcp_server --help
   python examples/query_layer_example.py
   ```

7. If an LLM provider is configured, run one small extraction smoke:
   ```bash
   drg extract inputs/input2.txt --auto-schema --output-format enhancedkg -o /tmp/drg-smoke-kg.json
   drg validate /tmp/drg-smoke-kg.json
   ```

8. For the final release, push the final tag only after the TestPyPI smoke test
   passes:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
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
