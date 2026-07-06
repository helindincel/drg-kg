# Migration Policy

DRG is **alpha** (`0.x`). This document describes how public APIs evolve before
and after `v1.0`.

## Alpha series (`0.x`)

### Stable for alpha

Documented in [`public_api.md`](public_api.md):

- Top-level extraction functions (`extract_typed`, `extract_from_chunks`, async
  variants, `extract_triples`)
- CLI commands: `extract`, `validate`, `diff`, `versions`, `eval`
- Core graph builders and evaluation entry points listed in `public_api.md`

Changes to stable-alpha APIs require:

1. A `CHANGELOG.md` entry (Breaking section when behavior changes).
2. A migration note in `public_api.md` or README.
3. A compatibility shim for at least one alpha minor release when practical.

### Experimental

Optimizer internals, confidence calibration formats, MCP/API response details,
clustering strategy classes, and UI template internals may change without a
deprecation window until `v1.0`.

## Pre-1.0 upgrades

- Pin `drg-kg==0.1.x` in production experiments.
- Read `CHANGELOG.md` before every upgrade.
- Prefer documented imports over private modules (`drg.extract._*`).

## v1.0 target

At `1.0.0` DRG will publish:

- Frozen public import surface (semver-major for breaking changes).
- Documented deprecation policy (minimum one minor release with warnings).
- Generated API reference documentation.

See [`adr/0001-record-architecture-decisions.md`](adr/0001-record-architecture-decisions.md).
