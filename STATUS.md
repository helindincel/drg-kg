# DRG-KG — Project Status & Gap Analysis

> **Audience:** Anyone trying to answer “what is this repo, why does it exist,
> and is it ready for me to depend on?”
>
> **State as of:** 2026-05-25 · version `0.1.0a0` · 295 passing tests · ~60%
> coverage · alpha on `test.pypi.org`, not yet on real PyPI.

---

## 1. What is DRG?

**DRG (Declarative Relationship Generation)** is a Python library that turns
raw text into a **Knowledge Graph (KG)** through a *declarative* pipeline.

Concretely, it lets you:

1. **Define a schema** (entity types + relations) instead of writing prompts.
2. **Extract** nodes and edges from arbitrary text using
   [DSPy](https://github.com/stanfordnlp/dspy) `TypedPredictor` against any
   LLM provider (OpenAI / Gemini / Anthropic / OpenRouter / Ollama / …).
3. **Build an EnhancedKG** (`KGNode` + `KGEdge` + `Cluster` with metadata).
4. **Run graph analytics** on top — Louvain / Leiden / Spectral clustering,
   community reports, hub mitigation.
5. **Serve / visualise** the graph through a FastAPI server with a
   Cytoscape.js UI, or export to Neo4j / JSON.

The whole thing is **dataset-agnostic**: same pipeline ran on Wikipedia,
financial news, and biomedical abstracts in the included quickstarts.

### Explicit non-goals

- **Not a RAG framework.** The UI query path is a deterministic KG lookup,
  not LLM generation. If you want a "chat over your docs" tool, this is the
  wrong repo.
- **Not a serving / vector-DB platform.** Vector stores can be plugged in
  experimentally, but they're not the product.
- **Not provider-locked.** Provider selection is one env variable away.

---

## 2. How does it work?

```
        ┌──────────┐   ┌──────────┐   ┌────────────────┐
text →  │ Chunking │ → │ Extract  │ → │ Build          │ → EnhancedKG
        │  (token/ │   │  (DSPy   │   │  (entities +   │     ↓
        │   sent.) │   │  Typed)  │   │   relations)   │   Cluster
        └──────────┘   └──────────┘   └────────────────┘     ↓
                              ↑                          Community
                       EnhancedDRGSchema                    Report
                       (or auto-generated)                   ↓
                                                       JSON / Neo4j / UI
```

### Architectural choices that matter

| Choice | Why it matters |
|---|---|
| **DSPy `TypedPredictor`** for extraction | Schema → signature is generated declaratively; prompts are not hand-tuned per dataset. |
| **Strategy patterns** in `coreference_resolution/`, `entity_resolution/`, `chunking/`, `graph/relationship_model/`, `graph/visualization_adapter/` | Each concern is a swappable strategy behind a `Protocol`. You can override one without touching the rest. |
| **Lazy top-level imports** (`drg/__init__.py`) | Importing `drg` doesn’t pull DSPy, sklearn, leidenalg, etc. until you actually need them. CLI startup stays fast. |
| **Optional-dependency matrix** in `pyproject.toml` | `pip install drg-kg` is tiny; `pip install drg-kg[all]` is the full stack. Clustering backends, embeddings, API server, and coref are each opt-in. |
| **Env-driven config** (`drg/config.py`) | No secrets in code; reproducibility for research runs. |
| **`drg/errors.py` typed hierarchy** | `DRGError` + 11 subclasses inheriting from `ValueError` / `RuntimeError` — back-compat without losing structure. |
| **Structured logging** (`drg/utils/logging.py`) | `DRG_LOG_FORMAT=json` flips logs into JSON for ingestion; chunk-scoped context via `with_context`. |
| **Shared LRU embedding cache** (`drg/utils/cache.py`) | Avoids re-embedding identical strings across providers; thread-safe. |

### Module map (high-density modules)

```
drg/
├── extract/          ← DSPy extraction (1.1k LOC across 8 modules)
├── chunking/         ← Token + sentence chunkers (tested at 100% / 88%)
├── graph/            ← EnhancedKG, builders, clustering glue, vis adapters
├── clustering/       ← Louvain / Leiden / Spectral + cluster summarisation
├── entity_resolution/← Hybrid string + similarity merging
├── coreference_resolution/ ← NLP strategy (spaCy/coreferee) + heuristic fallback
├── embedding/        ← Provider abstraction (OpenAI / Gemini / OpenRouter / local)
├── api/              ← FastAPI + Cytoscape.js UI
├── optimizer/        ← DSPy optimiser experiments
├── mcp_api.py        ← MCP integration
└── cli.py            ← `drg extract` entrypoint
```

---

## 3. Why does this need to exist?

KG extraction in 2026 generally comes in three flavours, none of which fit
the niche DRG targets:

| Approach | What it gives you | What's wrong for KG-research workloads |
|---|---|---|
| **Raw prompting per dataset** (LangChain / LlamaIndex glue) | Quick prototypes | Brittle, dataset-locked, prompt-creep, no reproducibility |
| **Heavy graph platforms** (Neo4j NLP, KG-builder SaaS) | End-to-end product | Hard to extend; opinionated; not Python-native |
| **Generic NER + relation extraction** (spaCy / transformers) | Strong baselines | No declarative schema; relations need a separate model per relation type |

DRG fits the gap by being:

- **Schema-first** — same code, different schema → different domain.
- **DSPy-native** — extraction is a *program*, not a prompt; signatures are
  derived from the schema, so changing schema systematically changes
  extraction.
- **Graph-first** — output is a real graph object with clustering and
  community reports baked in, not just a list of triples.
- **Research-grade but production-leaning** — typed errors, structured
  logging, optional-dep matrix, gradual mypy adoption, ~60% coverage,
  CI on 3.10/3.11/3.12.

Concrete users in mind:

- **Research engineers** running KG-extraction experiments across multiple
  datasets — DRG removes the "rewrite the pipeline" tax.
- **NLP / GraphRAG teams** that need a clean KG layer *before* they bolt
  on retrieval or LLM serving themselves.
- **Open-source contributors** experimenting with new clustering /
  coreference / relation-model strategies — the strategy pattern is
  designed for this.

---

## 4. What's actually missing (gap analysis)

These are ordered roughly by how much they limit adoption *right now*. Each
bullet is something an outside contributor or downstream user can act on.

### 4.1 Tests / coverage holes

Current floor is **60.32 %**, gate at **50 %**. Quality is uneven —
some modules are at 100 %, others below 20 %. Concrete gaps:

| Module | LOC | Cov | Why it matters |
|---|---:|---:|---|
| `drg/graph/community_report.py` | 96 | 10 % | Public API surface; users *will* hit this when calling cluster reports. |
| `drg/graph/hub_mitigation.py` | 74 | 4 % | Entirely untested; UI uses it. |
| `drg/graph/kg_core.py` | 172 | 38 % | The actual `EnhancedKG` class — should be the most-tested file in the repo, isn't. |
| `drg/extract/__init__.py` | 528 | 14 % | The core extraction loop. Hardest to test (LLM dependency), but mockable patches like `tests/test_extract_mock.py` are the model — expand them. |
| `drg/extract/_parsing.py` / `_relations.py` / `_schema_gen.py` | ~210 | 12–26 % | Pure-Python helpers; no excuse not to unit-test them. |
| `drg/coreference_resolution/_scoring.py` | 62 | 21 % | No external deps needed — straight functions, easy win. |
| `drg/optimizer/__init__.py` and below | ~3 lines covered | <20 % | Excluded from omit list but barely tested. |
| `drg/api/__init__.py` | 2 | 0 % | Server smoke test missing. |
| `drg/mcp_api.py` (608 LOC) | — | **excluded** | Not in omit list either way — needs at least a contract test. |
| `drg/cli.py` (in omit list) | — | **excluded** | Should have at least one `subprocess` smoke test (`drg --help`, `drg extract -`). |

**Action:** open issues for each row; aim for floor → **70 %** before
graduating from alpha.

### 4.2 Documentation

- **Almost all of `docs/` is in Turkish.** README is bilingual, code is
  English, but `docs/project_overview.md`, `schema_design.md`,
  `chunking_strategy.md`, `clustering_summarization.md`,
  `pipeline_overview.md`, `optimizer_design.md`,
  `relationship_model.md`, `mcp_integration.md` are all Turkish.
  Non-Turkish-speaking contributors are effectively blocked from the
  deep docs.
- **No `docs/api_reference.md`.** Currently the only API surface listing
  is in the README; a generated reference (e.g. `mkdocs` + `mkdocstrings`)
  is missing.
- **No tutorial chain.** Quickstarts are good but disconnected. Need a
  "Hello world → custom schema → custom strategy → API server" progression.
- **No architecture-decision records (ADRs).** Choices like
  "no RAG", "DSPy over raw OpenAI", "strategy pattern over inheritance"
  are explained ad-hoc in `project_overview.md` but not as ADRs.
- **No CHANGELOG migration guide.** Pre-`1.0` minor bumps can break
  things; today there's no "if you used X before, do Y now" section.

### 4.3 Release / distribution

- **Not on real PyPI yet** — only `test.pypi.org`. Need to:
  1. Reserve the name `drg-kg` on PyPI (free, takes 30 s).
  2. Add `.github/workflows/release.yml` that triggers on tag `v*.*.*`,
     builds with `uv build`, uploads via `twine` with a stored secret.
  3. Decide on a versioning source of truth — currently `__version__` in
     `drg/__init__.py` and `version` in `pyproject.toml` are duplicated.
     Use [`setuptools-scm`](https://setuptools-scm.readthedocs.io/) or
     `hatch-vcs` so the tag *is* the version.
- **No `py.typed` marker file.** The package is type-annotated, but
  without `drg/py.typed`, downstream mypy users get `[import-untyped]`
  errors. Trivial to add (PEP 561).
- **No PyPI / CI / coverage badges** in the README.
- **No GitHub release notes automation.** A `release-drafter.yml` would
  build the changelog from PR labels automatically.

### 4.4 CI / quality gates

- **No coverage upload to Codecov / Coveralls.** `coverage.xml` is
  uploaded as an artefact but isn't reported anywhere visible. PRs can
  silently regress coverage.
- **No integration-test job.** Currently `integration` marker tests are
  skipped in CI. At least a scheduled workflow with mocked-LLM
  integration runs would catch silent breakage.
- **Pre-commit is local-only.** No CI step runs `pre-commit run --all`,
  so contributors can land code that violates the same hooks pre-commit
  would have caught.
- **`mypy` runs in gradual mode** with eight error codes disabled
  (`assignment`, `arg-type`, `var-annotated`, `index`, `misc`,
  `override`, `truthy-function`, `attr-defined`). Each one is a future
  cleanup pass — there's no tracking issue / milestone for them.
- **No security scan.** `pip-audit` / `safety` / `dependabot` aren't
  wired in. `SECURITY.md` exists but enforcement is manual.

### 4.5 Examples / usability

- **All three quickstarts require an LLM key.** A no-key, **mocked**
  quickstart (using `tests/test_extract_mock.py`-style fakes) would let
  someone evaluate the library shape in 30 seconds.
- **No Jupyter notebook quickstart.** Notebooks are the universal "I
  want to try this on a flight" format.
- **API server example shows endpoints but no end-to-end UI walkthrough.**
  Screenshots / a GIF in the README would help conversion.
- **No `examples/benchmarks/`.** There's no story for "how does DRG
  extraction quality compare to `<X>` on `<dataset>`?".

### 4.6 Benchmarks / evaluation

- **No formal extraction benchmark.** There's `tests/multi_dataset/`
  but it's not invoked anywhere obvious, and there's no
  `benchmarks/` directory with NER F1 / relation precision-recall
  / cluster-purity metrics on a public corpus (DocRED, CoNLL, etc.).
- **No performance budgets.** No measurement of "X tokens / sec per
  provider", "memory for 1M-node graph", "build time for clustering on
  100k-node graph". Without these, scaling claims are anecdotal.
- **`drg/optimizer/`** is meant to *evaluate* extraction quality but is
  barely covered (15 %) and has no doc'd end-to-end recipe.

### 4.7 API / public-surface stability

- **Lazy-import in `drg/__init__.py`** is clever but undocumented. Users
  see `from drg import KGExtractor` working magically; type-checkers
  often don't.
- **No `__all__` discipline below the top level.** Submodules export
  whatever they happen to define; some are private (`_xxx`) but the
  rule isn't formalised.
- **No deprecation policy.** "Pre-`1.0` minor bumps may include breaking
  changes" is in the CHANGELOG, but there's no `DeprecationWarning`
  helper, no migration window standard.

### 4.8 Ecosystem / integrations

- **No `langchain` adapter.** The audience for "give me a KG from text"
  overlaps heavily with LangChain users; even a one-file adapter would
  buy a lot of discoverability.
- **No `dspy` example for *training* the extractor.** The optimiser
  module exists but the README doesn't show how to actually run a
  bootstrap / MIPRO loop against DRG's schema.
- **MCP integration is shipped but undocumented for non-MCP users.**
  `drg/mcp_api.py` is 600 lines; `docs/mcp_integration.md` is Turkish.
- **No Docker image.** A `Dockerfile` + `docker run drg-kg/server` would
  remove "install spaCy models, sklearn, leidenalg" friction.

### 4.9 Governance

- `CONTRIBUTING.md` exists but is short — needs a "Architecture
  guardrails" section pointing at the strategy patterns and the
  "no RAG" non-goal.
- **No `CODE_OF_CONDUCT.md`.** Standard expectation for open-source.
- **No issue triage labels** beyond default. `good-first-issue`,
  `help-wanted`, `area:extract`, `area:graph` would help inbound
  contributors self-select.

---

## 5. Suggested order of attack

If the goal is "ship a real `0.1.0` (not alpha) on PyPI within ~2 weeks,"
this is the order I'd take:

1. **(½ day)** Reserve `drg-kg` on real PyPI; add `release.yml` workflow
   tied to tags; collapse the dual `__version__` / `pyproject.toml`
   versioning to one source.
2. **(½ day)** Add `py.typed` marker; add PyPI / CI / coverage badges to
   README.
3. **(1 day)** Translate `docs/project_overview.md` and
   `docs/pipeline_overview.md` to English (the rest can stay Turkish
   for now with a banner). Unblocks non-Turkish contributors.
4. **(1–2 days)** Plug the worst three coverage gaps —
   `graph/kg_core.py`, `graph/community_report.py`, `graph/hub_mitigation.py`.
   These are the public-surface classes everyone hits.
5. **(½ day)** Add a no-key **mocked** quickstart and a Jupyter notebook
   version. Lowers the "try it" barrier from "get an API key" to "click".
6. **(½ day)** Wire Codecov upload + a pre-commit CI job + dependabot.
   Quality regressions become visible.
7. **(1 day)** First public release: tag `v0.1.0`, push to real PyPI,
   open a GitHub release with auto-generated notes, drop a tweet / Show
   HN, see what feedback comes back.
8. **(ongoing)** Track each disabled mypy code in its own issue; tackle
   one per week; once all are re-enabled, bump to `0.2.0`.

Anything past step 7 is steady-state library maintenance, not
launch-blocking.

---

## 6. What's already solid

Lest the gap list feel discouraging — these things are in **good shape**
right now and don't need attention:

- **CI matrix** — 3.10 / 3.11 / 3.12, lint + mypy + tests, all green.
- **Pre-commit config** — Ruff + MyPy + `detect-private-key` + standard
  hygiene hooks.
- **Modular packaging** — 9 optional-dependency groups; core install
  pulls only DSPy + Pydantic.
- **`pyproject.toml` discipline** — single source of truth, no
  `setup.py` / `setup.cfg` / `requirements.txt` drift.
- **Typed error hierarchy** — `DRGError` + 11 subclasses with
  `ValueError` / `RuntimeError` back-compat. Rare to see this done well
  in research codebases.
- **Strategy-pattern modularisation** — five major monoliths (extract,
  coref, entity-res, relationship-model, vis-adapter) already
  broken down. Future contributors can add strategies without
  touching the core.
- **Quickstarts** — three runnable, domain-different scripts in
  `examples/quickstarts/`. Beats a single canned example.
- **CHANGELOG hygiene** — Keep-a-Changelog format, semver discipline,
  per-release "Breaking" / "Known limitations" sections.
- **Test discipline** — 295 passing tests, deterministic via
  `FakeTokenizer` / `pytest.importorskip`, integration tests properly
  marker-gated.

---

## 7. TL;DR for the impatient

DRG-KG is a **schema-driven, DSPy-based, declarative KG-extraction
library** with strong modular foundations and a clean alpha release. The
biggest things blocking it from being broadly adopted right now are
(a) translating `docs/` to English, (b) closing the worst coverage
holes (`kg_core`, `community_report`, `hub_mitigation`, the `extract`
helpers), (c) shipping on real PyPI with proper release automation,
and (d) lowering the "try it in 30 s" barrier with a mocked, no-key
quickstart.

Everything else is steady-state polish.
