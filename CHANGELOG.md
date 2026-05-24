# Changelog

All notable changes to DRG are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-`1.0` minor bumps may include breaking changes; they are still called
out under **Breaking** sections.

## [Unreleased]

## [0.1.0a0] — 2026-05-24

First public alpha. The codebase has been overhauled across five sprints
to land on a coherent, production-leaning foundation.

### Added

- `drg/errors.py` — typed exception hierarchy (`DRGError` + 11 subclasses)
  inheriting from `ValueError` / `RuntimeError` for back-compat.
- `drg/utils/logging.py` — structured logging helpers (`get_logger`,
  `with_context`, `configure_logging`, JSON formatter). Opt-in via
  `DRG_LOG_FORMAT=json`.
- `drg/utils/cache.py` — thread-safe LRU `EmbeddingCache` wrapper with
  shared-provider registry (`cached_provider`) and hit/miss stats.
- `drg/utils/strict.py` — encapsulated `DRG_STRICT` mode logic so callers
  can opt into strict error raising consistently.
- `drg/protocols.py` — structural interfaces (`KGExtractorProtocol`,
  `EmbeddingProviderProtocol`, `ClusteringAlgorithmProtocol`,
  `LLMProtocol`) using `typing.Protocol` + `runtime_checkable`.
- `tests/fixtures/` — hand-written reference data for deterministic
  regression tests.
- `.github/workflows/ci.yml` — Lint (Ruff) + type-check (MyPy) + test
  (pytest, coverage) on Python 3.10/3.11/3.12.
- `.pre-commit-config.yaml` — Ruff, MyPy, `detect-private-key`, and a
  set of standard pre-commit hooks.
- 57 new tests covering the refactored packages, typed exceptions,
  structured logging, dependency injection, and the embedding cache.
- Documentation: `docs/setup.md`, `docs/api_server.md`,
  `docs/project_overview.md` (consolidated from older READMEs).

### Changed

- **Modularised three large monolith files** into packages with strategy
  patterns, preserving public APIs:
  - `drg/extract.py` (2249 lines) → `drg/extract/` package.
  - `drg/coreference_resolution.py` (797 lines) →
    `drg/coreference_resolution/` package.
  - `drg/entity_resolution.py` (610 lines) → `drg/entity_resolution/`
    package.
  - `drg/graph/relationship_model.py` (687 lines) →
    `drg/graph/relationship_model/` package.
  - `drg/graph/visualization_adapter.py` (808 lines) →
    `drg/graph/visualization_adapter/` package.
- `KGExtractor` now accepts an optional `lm` parameter for dependency
  injection; `extract_typed` / `extract_from_chunks` thread it through.
  The global `dspy.configure` path still works as a fallback.
- Internal `RuntimeError` / `ValueError` call sites in `drg.extract`,
  `drg.config`, and schema generation now raise their typed counterparts
  (`ExtractionError`, `LLMConfigError`, `SchemaGenerationError`,
  `GraphError`). Back-compat preserved through inheritance.
- Hot-path loggers in extraction, entity resolution, and coreference
  resolution use `get_logger` + `with_context` for chunk-scoped context.
- `pyproject.toml` modernised: `[build-system]`, rich metadata,
  classifiers, `[project.optional-dependencies]` matrix, tooling configs
  for `pytest`, `pytest-cov`, `ruff`, `mypy`. Minimum coverage gate at
  50%.
- `README.md` rewritten as an installation/usage entry point with
  pointers into `docs/`.

### Removed

- `requirements.txt` — superseded by `pyproject.toml`.
- `INSTALL_SPACY.md`, `QUICK_START.md`, `README_API.md`, `SETUP.md` —
  content merged into the new `docs/` files.
- `quick_start.sh`, `restart_api_server.sh`, `start_api_server.sh` —
  scripts inlined or documented as one-liners.
- `.env.save` — accidentally committed in earlier history; removed and
  blocked by `.gitignore`.

### Security

- `.gitignore` tightened to block all `.env*` files except `.env.example`.
- `pre-commit` includes `detect-private-key` to catch accidental leaks.
- No hardcoded API keys remain in tracked files.

### Known limitations

- The graph visualization layer (`drg/graph/visualization.py`,
  `neo4j_exporter.py`) is intentionally excluded from coverage gating —
  exercised end-to-end but not unit-tested.
- Integration tests require live LLM credentials and are skipped in CI.
- DSPy 3.x is not supported; pinned to `>=2.5.0,<3.0.0`.

[Unreleased]: https://github.com/helindincel/drg-kg/compare/v0.1.0a0...HEAD
[0.1.0a0]: https://github.com/helindincel/drg-kg/releases/tag/v0.1.0a0
