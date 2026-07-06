# Changelog

All notable changes to DRG are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-`1.0` minor bumps may include breaking changes; they are still called
out under **Breaking** sections.

## [Unreleased]

### Release summary

#### Features

- DRG is positioned and documented as a schema-driven Knowledge Graph extraction
  framework: define or infer a schema, extract typed entities and relations, and
  build an `EnhancedKG` artifact for validation, deterministic query,
  evaluation, versioning, and export.
- The primary extraction APIs are treated as stable for the alpha series:
  `extract_typed()`, `extract_from_chunks()`, `extract_typed_async()`,
  `extract_from_chunks_async()`, and the backward-compatible `extract_triples()`
  wrapper.
- Package metadata, optional extras, Python 3.10-3.13 support, release
  automation, CI, typed package marker, CLI/API/MCP/evaluation examples, and
  bilingual README entry points are prepared for the first public release cycle.

#### Breaking changes

- No additional breaking changes are introduced by the release-readiness
  documentation pass.
- Pre-`1.0` users should still review this changelog before upgrading because
  alpha JSON shapes, CLI flags, and optional integration surfaces may change.

#### Experimental

- Optimizer integration is experimental and should be treated as a research
  workflow around extraction quality.
- Confidence calibration remains heuristic/experimental unless calibrated
  against labelled domain data.
- Long-document optimization, including chunking and cross-chunk/windowed
  relation recovery, is evolving.
- API/UI/MCP implementation details outside documented commands and endpoints
  are not frozen.

#### Known limitations

- DRG is not a GraphRAG framework, not a RAG framework, and not a retrieval/chat
  serving stack.
- Live extraction quality depends on the configured LLM, schema quality, and
  provider behavior.
- Integration tests that require live credentials remain outside the default CI
  path.

### Changed (documentation cleanup)

- `README.md` and `README.tr.md` were refreshed as the canonical root
  entry points: current install commands, CLI surface, API/UI/MCP/evaluation
  examples, project map, and doc links now live there directly.
- Removed stale root-level planning notes (`BASLANGIC.md`, `EKSIKLER.md`,
  `REMAINING.md`, `STATUS.md`) so the repository root keeps only stable
  project documents.

### Added (CI hardening)

- `.github/workflows/ci.yml` now uploads `coverage.xml` to
  Codecov via the official `codecov/codecov-action@v4`, gated to
  the 3.11 matrix entry so the same numbers aren't double-counted.
  `fail_ci_if_error: false` keeps CI green if Codecov is briefly
  unreachable — the test job has already provided pass/fail
  signal. The optional `CODECOV_TOKEN` secret is wired through
  for private clones / rate-limit cases.
- New `precommit` job runs `pre-commit run --all-files` via the
  official `pre-commit/action@v3.0.1` (which caches hook
  environments between runs). This catches whitespace, EOF,
  ruff, mypy, and `detect-private-key` regressions on every PR
  instead of relying on contributors to install pre-commit
  locally.
- `.github/dependabot.yml` opens weekly grouped PRs for outdated
  Python dependencies (pyproject.toml) and GitHub Actions. Major
  bumps still ship as individual PRs; DSPy 3.x is intentionally
  ignored until our extraction layer is migrated.

### Changed (mypy + ruff: bring the codebase to pre-commit-green)

- `drg/optimizer/optimizer.py::optimize` — added an explicit
  `assert self.optimized_extractor is not None` before the
  return so mypy can narrow `Optional[KGExtractor]` to
  `KGExtractor`. The three try-branches above already assign it;
  the assert documents that invariant.
- `drg/graph/schema_generator.py` — the two `import yaml`
  call sites under `try/except ImportError` now carry
  `# type: ignore[import-untyped]`. PyYAML is an optional
  runtime dependency; pulling `types-PyYAML` into the dev set
  just to silence this single import would be heavier than the
  inline ignore.
- `.pre-commit-config.yaml` — bumped the ruff-pre-commit hook
  from `v0.5.7` to `v0.15.14` to match the local ruff. The old
  hook still flagged `UP038` (`isinstance(x, (A, B))`), a rule
  Astral has since removed. Without this bump, CI would have
  failed on lints that the local `ruff check` no longer reports.
- One-off cleanup pass from `pre-commit run --all-files`:
  end-of-file newlines normalised across 4 `inputs/*.txt`,
  `tests/README.md`, several `docs/*.md` files, a few `tests/`
  modules, `drg/api/templates/index.html`, and 4 newly-converted
  `isinstance(x, A | B)` call sites (`drg/graph/visualization_adapter/_hubproxy.py`,
  `drg/graph/visualization.py`, `tests/test_chunking_strategies.py`,
  `tests/test_extraction_functionality.py`). These were pure
  whitespace / formatter changes; behaviour is unchanged.

### Added (graph coverage expansion: community_report + hub_mitigation)

- `tests/test_graph_community_report.py` (24 tests) — exhaustive
  unit coverage of `drg.graph.community_report`. Drives all five
  private helpers via the public `generate_report` API: summary
  generation (including the no-relationships, no-nodes, and
  unknown-type branches), top-actors counting (external edges
  that touch cluster members still count), top-relationships
  extraction with `max_*` capping, theme identification across
  the entity-type-centric / mapped-relation / fallback-relation
  / density (>0.5, <0.2) branches and the `max_themes` cap, and
  the density helper (single-node and empty-cluster zero
  fallbacks). `generate_all_reports` (multi-cluster + empty KG),
  `export_reports_json` (parent-dir creation), and
  `generate_report_text` (populated, metadata-omitted, and
  fully-empty variants) are all covered.
- `drg/graph/community_report.py` jumped from **10 % to 100 %**.

- `tests/test_graph_hub_mitigation.py` (21 tests) — coverage of
  `drg.graph.hub_mitigation.apply_hub_relation_proxy_split`. The
  function had a coverage of **4 %**, the lowest in the project.
  Tests cover the disabled-no-op short-circuit, threshold
  validation (`< 3` rejected, parametrised), the no-hub case,
  single-hub single-relation and multi-relation proxy creation,
  proxy edge re-routing for hub-as-source and hub-as-target,
  hub-to-hub edges (routed through the source hub's proxy),
  semantic-edge preservation (relationship_type, detail,
  metadata, start_time, end_time, confidence, is_negated all
  flow into the proxy edge), non-hub edges left untouched, the
  threshold boundary (==N hub vs N-1 not a hub), multi-hub
  scenarios, custom `proxy_node_type`/`proxy_id_prefix` knobs,
  `seen_new` deduplication via a deliberately duplicated edge,
  stats-vs-graph correctness, and orphan-endpoint backfill.
- `drg/graph/hub_mitigation.py` jumped from **4 % to 89 %** (the
  remaining branches are defensive guards on shapes the public
  API cannot produce).

- Overall project coverage: **60.32 % → 65.80 %** with 395
  passing tests (up from 295 before this work).

### Added (kg_core unit coverage)

- `tests/test_graph_kg_core.py` (55 tests) — exhaustive unit
  coverage of the most public-surface module in the library.
  Covers `KGNode` / `KGEdge` / `Cluster` dataclass validation
  (every guard clause + parametrised confidence boundaries),
  full `to_dict` ↔ `from_dict` round-trips, the
  `KGEdge.from_enriched_relationship` adapter (core fields,
  optional `source_ref`, `is_negated` fallback, nested *and*
  flat temporal metadata shapes), `EnhancedKG.add_node` /
  `add_edge` / `add_cluster` referential-integrity guards, all
  three exporters (`to_json`, `to_json_ld`, `to_enriched_format`)
  including back-compat for confidence stored in metadata, the
  `save_*` file-IO wrappers via `tmp_path`,
  `from_enriched_relationships` factory, and `add_entity_embeddings`
  with a deterministic stub provider (default and explicit
  text-mapping paths).
- `drg/graph/kg_core.py` line coverage jumped from **38 % to
  99 %**, lifting overall project coverage from 60.3 % to
  **62.3 %**. The two remaining uncovered branches are
  defensive guards on impossible inputs, kept on purpose.

### Added (English documentation)

- `docs/project_overview.md` — full English translation of the
  architecture / philosophy doc. Covers what DRG is, what it
  explicitly is *not* (not a question-answering framework, not a serving
  platform, not provider-locked), why DSPy + declarative,
  dataset-agnostic design, EnhancedDRGSchema, pipeline,
  monolithic-modular architecture, UI/query behaviour
  (deterministic, no LLM), env-driven configuration, repo
  structure, typical usage scenarios, and a comparison summary.
- `docs/pipeline_overview.md` — full English translation of the
  pipeline-flow doc. Includes the ASCII flow diagram, per-layer
  responsibilities and design decisions, metadata schemas (chunk
  / node / edge), trade-offs (chunking, embedding, query),
  extensibility points, and the evaluation methodology
  (multi-dataset, comparison framework).
- The Turkish originals are preserved as `*.tr.md` (no content
  loss) and both bilingual sets cross-link.
- `README.md` and `README.tr.md` now expose a **side-by-side
  EN/TR table** for every doc instead of an English-only list,
  so Turkish readers don't lose their entry point and the
  remaining-to-translate docs are explicit.

### Added (typing + visibility)

- `drg/py.typed` — empty PEP 561 marker so downstream `mypy` and
  `pyright` honour DRG-KG's type annotations instead of treating
  the package as untyped (`[import-untyped]` warning). Shipped in
  the wheel via `[tool.setuptools.package-data]`.
- README badges: CI status, PyPI version, supported Python
  versions, license, Ruff code style, PEP 561 typed marker.
  Bilingual: both `README.md` and `README.tr.md` get the same
  badge row and project-status pointer.
- Installation section in README now leads with `pip install
  drg-kg` (PyPI) and lists `pip install -e ".[dev]"` as the
  *source* path, reflecting the impending real-PyPI release.
- `drg/_version.py` is excluded from Ruff via `extend-exclude`
  to silence machine-generated-file lint noise.

### Added (release automation)

- `.github/workflows/release.yml` — tag-driven release pipeline. Tags
  matching `vX.Y.Z` (final), `vX.Y.Za[N]`, `vX.Y.Zb[N]`, `vX.Y.Zrc[N]`
  trigger a build + `twine check` + upload. Pre-releases land on
  TestPyPI; only final-release tags publish to PyPI. Manual
  `workflow_dispatch` is available for TestPyPI dry-runs. Both
  uploads use the official `pypa/gh-action-pypi-publish` action and
  GitHub environments (`pypi`, `testpypi`) for review gates.
- Added a root-level project status and gap analysis note covering what the
  library does, how, why it exists, and an honest list of remaining gaps
  (coverage, docs, release, CI, examples, benchmarks, governance) with a
  suggested order of attack. This note was later folded back into the canonical
  README/docs flow.

### Changed (single-source version)

- Adopted `setuptools_scm` as the single source of truth for the
  package version. `pyproject.toml` now declares `version` as
  `dynamic`, and `drg/__init__.py` reads `__version__` from the
  build-generated `drg/_version.py` (falling back to
  `importlib.metadata.version("drg-kg")`). This eliminates the
  previous dual source-of-truth between `pyproject.toml` and
  `drg/__init__.py`; bumps are now driven entirely by git tags.
- `.gitignore` now excludes `drg/_version.py` (build-generated).

### Added (test coverage expansion)

- Unit tests for `drg/graph/builders.py`, `drg/graph/auto_clusters.py`,
  `drg/graph/query_engine.py`, `drg/clustering/summarization.py`, and
  `drg/clustering/algorithms.py`. The first four run with zero external
  dependencies; the clustering algorithms tests use `pytest.importorskip`
  for each optional backend (python-louvain, leidenalg+igraph, sklearn)
  so they pass cleanly whether those packages are installed or not.

### Changed (coverage omit list)

- Removed the five sources above from `[tool.coverage.run].omit`; the
  coverage gate now reflects their real measured coverage.


### Added

- English `README.md` (Turkish version preserved as `README.tr.md`); the two
  link to each other and call out that `docs/` is currently Turkish.
- `examples/quickstarts/` — three self-contained, runnable showcase scripts
  (`01_wikipedia_article.py`, `02_financial_news.py`, `03_biomedical.py`)
  demonstrating the `schema → extract_typed → KG` loop on different domains.
- `tests/test_chunking_strategies.py` (23 tests) and
  `tests/test_chunking_validators.py` (14 tests) covering `drg.chunking`.

### Changed

- Removed `drg/chunking/strategies.py` and `drg/chunking/validators.py` from
  the coverage `omit` list now that they have dedicated unit tests; both
  files are exercised end-to-end without LLM calls via a `FakeTokenizer`.

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
  (pytest, coverage) on Python 3.10/3.11/3.12/3.13.
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
- The original alpha targeted DSPy 2.x. Current unreleased work targets
  DSPy 3.x via the `>=3.2.1,<4.0.0` optional extraction extra.

[Unreleased]: https://github.com/helindincel/drg-kg/compare/v0.1.0a0...HEAD
[0.1.0a0]: https://github.com/helindincel/drg-kg/releases/tag/v0.1.0a0
