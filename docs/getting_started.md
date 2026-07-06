# DRG — Getting Started

This guide is the canonical first-run path for a new DRG checkout or package
install. It keeps the happy path small, then points to optional surfaces such as
the API server, MCP, Neo4j, and evaluation framework.

## 1. Create an Environment

DRG is tested on Python 3.10, 3.11, 3.12, and 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install from PyPI:

```bash
# Graph-only workflows (validation, query, versioning — no LLM extraction)
pip install drg-kg

# Extraction + chunking (`tiktoken`, DSPy, `drg extract`)
pip install "drg-kg[extract]"
```

> **Package name:** install **`drg-kg`** from PyPI (`pip install drg-kg`), then
> `import drg` in Python. The PyPI name `drg` belongs to an unrelated project.

For a source checkout:

```bash
# Graph-only workflows
pip install -e .

# `drg extract` and Python extraction APIs
pip install -e ".[extract]"
```

Install optional extras only when you need them:

```bash
pip install -e ".[extract]"  # DSPy extraction
pip install -e ".[api]"      # FastAPI server and UI
pip install -e ".[mcp]"      # MCP server
pip install -e ".[neo4j]"    # Neo4j export
pip install -e ".[all]"      # Everything, useful for local demos
```

## 2. Verify the Install

```bash
python -c "import drg; print(drg.__version__)"
drg --help
```

The top-level CLI help should list `extract`, `validate`, `diff`, `versions`,
and `eval`.

## 3. Configure a Model

Live extraction uses DSPy/LiteLLM. Choose one provider before running
`drg extract` or Python extraction APIs.

```bash
cp .env.example .env

# Default cloud setup
export DRG_MODEL=openai/gpt-4o-mini
export OPENAI_API_KEY=sk-...

# Gemini
export DRG_MODEL=gemini/gemini-2.0-flash-exp
export GEMINI_API_KEY=...

# Local Ollama
export DRG_MODEL=ollama_chat/llama3
export DRG_BASE_URL=http://localhost:11434
```

If you only want to verify the repo without an API key, run the deterministic
demo instead:

```bash
python examples/query_layer_example.py
```

## 4. Extract and Validate a Small Graph

```bash
echo "TechCorp was founded by Jane Doe in 2015." > sample.txt
drg extract sample.txt --auto-schema -o output_kg.json
drg validate output_kg.json
```

Use `drg extract --help` for extraction-specific options such as model
overrides, incremental updates, reasoning, event extraction, and output format.

## 5. Explore the Graph

For a local UI demo:

```bash
pip install -e ".[api]"
python examples/api_server_example.py
```

Open `http://localhost:8000`, then load the full graph. The UI query box is a
deterministic graph lookup, not an LLM answer generator.

## 6. Next Steps

- Long documents and tokenization: [`docs/chunking_strategy.en.md`](chunking_strategy.en.md)
  (install `drg-kg[extract]` for `tiktoken` + DSPy chunking)
- Schema design: `docs/schema_design.md`
- Query layer: `docs/query_layer.md`
- Incremental updates: `docs/incremental_updates.md`
- Graph versioning: `docs/graph_versioning.md`
- API server and UI: `docs/api_server.md`
- MCP integration: `docs/mcp_integration.md`
- Evaluation framework: `docs/evaluation_framework.md`
- Example scripts: `examples/quickstarts/README.md`
