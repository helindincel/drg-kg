# DRG Quickstart Examples

Three self-contained, single-file scripts that demonstrate the **schema →
extract → KG** loop on different domains. Each script defines its own schema
inline, runs on a short embedded sample text, and writes a JSON KG next to
itself.

| # | Script | Domain | Best for |
|---|--------|--------|----------|
| 1 | [`01_wikipedia_article.py`](01_wikipedia_article.py) | Biographical / encyclopedic text | General-purpose, easiest entry point |
| 2 | [`02_financial_news.py`](02_financial_news.py) | Corporate / financial news | M&A, funding rounds, market intelligence |
| 3 | [`03_biomedical.py`](03_biomedical.py) | Biomedical abstract | Drug-disease-gene research graphs |

## Prerequisites

- DRG installed (`pip install -e .` from the repo root)
- An LLM provider API key in the environment:
  - **OpenAI** (default): `export OPENAI_API_KEY=sk-...`
  - **Gemini**: `export GEMINI_API_KEY=... && export DRG_MODEL=gemini/gemini-2.0-flash-exp`
  - **Local Ollama**: `export DRG_MODEL=ollama_chat/llama3 && export DRG_BASE_URL=http://localhost:11434`

See [`../../README.md`](../../README.md) for the full configuration matrix.

## Running

```bash
# From repo root
python examples/quickstarts/01_wikipedia_article.py
python examples/quickstarts/02_financial_news.py
python examples/quickstarts/03_biomedical.py
```

Each script prints a human-readable summary and writes a `.json` KG dump
beside itself (e.g. `01_wikipedia_article.json`).

## Next step

When you're ready to move past hand-defined schemas and small inputs, see:

- [`../full_pipeline_example.py`](../full_pipeline_example.py) — chunking,
  cross-chunk extraction, embeddings, clustering, community reports
- [`../api_server_example.py`](../api_server_example.py) — interactive UI
  over the resulting KG
- [`../../docs/schema_design.md`](../../docs/schema_design.md) — designing
  richer schemas
