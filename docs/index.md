# DRG-KG Documentation

DRG (Declarative Relationship Generation) is a schema-driven Knowledge Graph
extraction framework built on DSPy.

## Quick links

- [Getting Started](getting_started.md)
- [Public API boundaries](public_api.md)
- [Chunking and tokenization (English)](chunking_strategy.en.md)
- [Release setup (PyPI/TestPyPI)](release_setup.md)
- [Migration policy](migration_policy.md)

## Install

```bash
pip install drg-kg
pip install "drg-kg[extract]"   # DSPy + tiktoken
```

Import in Python: `import drg`

> PyPI package name is **`drg-kg`**. The name `drg` is used by an unrelated
> Medicare DRG grouper project.

## Language note

Several design documents under `docs/` are still Turkish-only. English
translations are tracked for `v0.2`. The Python API, code comments, and error
messages are English.

See the bilingual doc table in [`README.md`](../README.md).
