# ADR 0001: Record architecture decisions

## Status

Accepted

## Context

DRG-KG is an alpha-stage open-source library with multiple integration surfaces
(API, MCP, Neo4j, evaluation). Design choices need a lightweight, searchable
record as the project approaches `v1.0`.

## Decision

We will use Architecture Decision Records (ADRs) stored in `docs/adr/`:

- One markdown file per decision: `NNNN-short-title.md`
- Sections: Status, Context, Decision, Consequences
- Superseded ADRs keep their file and link to the replacement

## Consequences

- Contributors can propose ADRs in pull requests alongside code changes.
- Breaking API decisions should reference an ADR and a `CHANGELOG.md` entry.
- Hosted docs (`mkdocs.yml`) can link the ADR index in `v0.2`.
