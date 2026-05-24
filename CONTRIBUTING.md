# Contributing to DRG

Thanks for considering a contribution! DRG is an alpha-stage research
codebase, so the bar for PRs is "clear, tested, and consistent with the
existing patterns" — not "perfect."

## TL;DR

```bash
# Fork → clone
git clone https://github.com/<your-username>/drg-kg.git
cd drg-kg

# Dev install
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run the test suite (no API keys needed for default markers)
pytest -m "not integration"
```

If everything is green, you're ready to start.

## Project layout (quick map)

```
drg/
├── extract/                     # DSPy-based KG extractor (package)
├── coreference_resolution/      # Pronoun resolution (strategy pattern)
├── entity_resolution/           # Entity merging (strategy pattern)
├── chunking/                    # Token / sentence / semantic chunkers
├── embedding/                   # Provider abstraction
├── graph/
│   ├── relationship_model/      # Relation taxonomy + classifier (package)
│   └── visualization_adapter/   # KG → Cytoscape/vis-network/D3 (package)
├── clustering/                  # Louvain / Leiden / Spectral
├── optimizer/                   # DSPy optimizer + metrics
├── errors.py                    # Typed exception hierarchy
├── protocols.py                 # Structural interfaces
├── config.py                    # Env-driven DSPy LM config
└── utils/                       # env_loader, logging, cache, strict, llm_throttle

tests/                           # Unit + integration + fixtures
docs/                            # Architecture / design docs (no code)
examples/                        # Runnable end-to-end examples
```

Architecture detail lives in `docs/project_overview.md`.

## Development workflow

### Branching

- `main` is the only long-lived branch.
- Open PRs from feature branches: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`,
  `refactor/<topic>`, `test/<topic>`.

### Commit messages

We use **Conventional Commits**:

```
feat: add hybrid embedding similarity strategy
fix: prevent KGEdge construction when source equals target
refactor: split relationship_model.py into package
test: cover hub-proxy flatten with rebuilt triples
docs: document EmbeddingCache normalization behavior
chore: bump ruff to v0.6
```

Scope is optional but encouraged for cross-cutting changes:
`feat(extract): ...`, `refactor(graph): ...`.

### Code style

- **Ruff** for lint + format (config in `pyproject.toml`).
- **MyPy** in gradual mode for `drg/`.
- **Type hints** on every public function. Internal helpers can skip them
  when they hurt readability.
- **Google-style docstrings** on public APIs.

Run everything at once:

```bash
ruff check drg tests examples
ruff format drg tests examples
mypy drg
pytest -m "not integration"
```

Or rely on the pre-commit hook (`pre-commit install`) to enforce style
before each commit.

### Tests

- Default suite (`pytest -m "not integration"`) must stay green on every
  PR and runs without any API keys.
- Integration tests are opt-in (`pytest -m integration`) and require an
  LLM key configured via `.env`.
- New behavior **needs a test**. Refactors must keep existing tests
  green; if you delete a test, explain why in the PR description.
- Prefer **small, hand-written fixtures** (see `tests/fixtures/`) over
  generated/recorded payloads.

### Documentation

- Architecture, design rationale, trade-offs → `docs/*.md`. **No code in
  docs files** (code lives in `examples/` if it's runnable).
- Public-API docstrings are part of the contract — update them when you
  change a signature.
- Update `README.md` for user-facing changes (new envs, CLI flags, etc.).

## What we accept

- ✅ Bug fixes with a regression test
- ✅ New embedding providers / clustering algorithms following the
  existing protocols (`drg.protocols`)
- ✅ Coreference / entity-resolution strategies under the strategy pattern
- ✅ Documentation improvements (examples especially welcome)
- ✅ Performance fixes with a benchmark

## What we usually push back on

- ❌ Large refactors without a stated motivation
- ❌ New runtime dependencies in `[project] dependencies` (move them to an
  optional extra)
- ❌ Behavior changes without test coverage
- ❌ Commits that mix unrelated changes (please split)

## PR checklist

Before opening a PR:

- [ ] `pytest -m "not integration"` is green
- [ ] `ruff check drg tests examples` is clean
- [ ] `mypy drg` has no new errors
- [ ] Public API changes are reflected in `README.md` and the relevant
      `docs/*.md`
- [ ] Commit messages follow Conventional Commits
- [ ] No secrets, `.env` files, or large generated artifacts are staged

## Questions?

Open a **Discussion** for design questions, or an **Issue** for bugs and
feature requests. Security reports → `SECURITY.md`.
