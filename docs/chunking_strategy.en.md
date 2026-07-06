# Chunking Strategy — Token Windows and Overlap (English Summary)

Full Turkish design notes: [`chunking_strategy.md`](chunking_strategy.md).

DRG splits long documents into repeatable chunks before DSPy extraction. Chunk
boundaries affect entity continuity and cross-chunk relation recovery.

## Install note

Chunking and extraction require the **`extract`** extra (ships `tiktoken` and DSPy):

```bash
pip install "drg-kg[extract]"
```

Graph-only workflows (`pip install drg-kg`) do not install tokenizers.

## Token window defaults

| Setting | Default | Range | Purpose |
|---------|---------|-------|---------|
| Window size | 768 tokens | 512–1024 | Balance between local context and cost |
| Overlap | 15% | 10–20% | Reduce entity/relation loss at boundaries |

**Short documents** (< 512 tokens): single chunk, no overlap.

**Medium / long documents**: fixed window with overlap and sequence indexing.

## Tokenizer selection

DRG abstracts tokenizers so chunk budgets align with your embedding or LLM
provider when possible:

| Provider / stack | Typical tokenizer |
|------------------|-------------------|
| OpenAI models | `tiktoken` (`cl100k_base`) via `[extract]` |
| Gemini | provider-specific counting in chunk validators |
| Local / HuggingFace | model-specific tokenizer when configured |

Mismatch between chunk tokenizer and model tokenizer can skew window sizes;
prefer the tokenizer that matches your configured `DRG_MODEL`.

## Python entry points

```python
from drg import extract_from_chunks

chunks = [
    {"text": "Paragraph one...", "chunk_id": "c0"},
    {"text": "Paragraph two...", "chunk_id": "c1"},
]
entities, triples = extract_from_chunks(chunks, schema)
```

Use `extract_from_chunks` (not `extract_typed`) for multi-chunk documents.

## Environment knobs

| Variable | Effect |
|----------|--------|
| `DRG_WINDOWED_RELATION_EXTRACTION` | `always` / `never` / `auto` cross-chunk relation recovery |
| `DRG_MAX_RELATION_CANDIDATE_PAIRS` | Cap candidate entity pairs per document |
| `DRG_MAX_RELATION_EVIDENCE_WINDOWS` | Cap evidence windows for relation passes |

See also [`getting_started.md`](getting_started.md) and [`public_api.md`](public_api.md).
