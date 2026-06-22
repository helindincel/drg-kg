"""Deterministic KG query engine.

This module powers the UI "Query" box with graph lookup only.
It performs lightweight, explainable graph lookup:
- entity name matching (case-insensitive substring + token overlap)
- neighborhood expansion (incident edges)
- optional relation-type filtering from the query string

The output is intentionally simple and UI-friendly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .kg_core import EnhancedKG, KGEdge

_STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "for",
    "to",
    "and",
    "or",
    "with",
    "about",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "which",
    "that",
}


@dataclass(frozen=True)
class QueryResult:
    """Result of a deterministic graph query."""

    seed_entities: list[str]
    matched_entities: list[str]
    matched_edges: list[dict[str, Any]]
    answer: str


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _tokenize(query: str) -> list[str]:
    q = _normalize(query)
    # keep letters/numbers and a few separators; split on whitespace
    tokens = [t for t in q.replace("(", " ").replace(")", " ").replace(",", " ").split() if t]
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _parse_relation_filter(query: str) -> str | None:
    """Try to extract an explicit relation filter from the query.

    Supported informal patterns:
    - "(works_at)"  -> "works_at"
    - "relation:works_at" -> "works_at"
    - "rel=works_at" -> "works_at"
    """
    q = (query or "").strip()
    if not q:
        return None

    # (relation_name)
    if "(" in q and ")" in q:
        inside = q.split("(", 1)[1].split(")", 1)[0].strip()
        if inside and " " not in inside and len(inside) <= 64:
            return inside

    lowered = q.lower()
    for key in ("relation:", "rel:", "relation=", "rel="):
        if key in lowered:
            after = lowered.split(key, 1)[1].strip()
            rel = after.split()[0].strip().strip(",.;")
            if rel:
                return rel

    return None


def _score_entity(
    entity_id: str, query_norm: str, query_tokens: Sequence[str]
) -> tuple[float, list[str]]:
    """Return (score, reasons). Higher is better."""
    eid_norm = _normalize(entity_id)
    reasons: list[str] = []
    score = 0.0

    if not eid_norm:
        return 0.0, reasons

    # Exact match
    if eid_norm == query_norm:
        score += 5.0
        reasons.append("exact_match")
        return score, reasons

    # Substring match
    if query_norm and query_norm in eid_norm:
        score += 3.0
        reasons.append("substring_match")

    # Token overlap
    if query_tokens:
        eid_tokens = set(eid_norm.split())
        overlap = len(eid_tokens.intersection(query_tokens))
        if overlap > 0:
            score += 1.0 + 0.4 * overlap
            reasons.append(f"token_overlap:{overlap}")

    return score, reasons


def _edge_to_dict(edge: KGEdge) -> dict[str, Any]:
    return edge.to_dict()


def execute_query(
    kg: EnhancedKG,
    query: str,
    k_entities: int = 10,
    k_edges: int = 40,
) -> QueryResult:
    """Execute a deterministic KG query (no LLM).

    Args:
        kg: Loaded EnhancedKG.
        query: Free-text query from UI.
        k_entities: Max seed entities to return.
        k_edges: Max incident edges to return for context.

    Returns:
        QueryResult including seed entities and matched edges.
    """
    query_norm = _normalize(query)
    query_tokens = _tokenize(query)
    rel_filter = _parse_relation_filter(query)
    rel_filter_norm = _normalize(rel_filter) if rel_filter else None

    # 1) Find best-matching entities by name.
    scored: list[tuple[float, str, list[str]]] = []
    for node_id in kg.nodes.keys():
        s, reasons = _score_entity(node_id, query_norm, query_tokens)
        if s > 0:
            scored.append((s, node_id, reasons))

    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    seed_entities = [node_id for _, node_id, _ in scored[: max(1, k_entities)]]

    # If nothing matched, fall back to top-degree entities (still deterministic).
    if not seed_entities:
        degree: dict[str, int] = dict.fromkeys(kg.nodes.keys(), 0)
        for e in kg.edges:
            degree[e.source] = degree.get(e.source, 0) + 1
            degree[e.target] = degree.get(e.target, 0) + 1
        seed_entities = [
            nid
            for nid, _ in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0].lower()))[
                : max(1, k_entities)
            ]
        ]

    seed_set = set(seed_entities)

    # 2) Collect incident edges around the seed entities.
    matched_edges: list[KGEdge] = []
    for e in kg.edges:
        if e.source in seed_set or e.target in seed_set:
            if rel_filter_norm and _normalize(e.relationship_type) != rel_filter_norm:
                continue
            matched_edges.append(e)
            if len(matched_edges) >= k_edges:
                break

    matched_entities: set[str] = set(seed_entities)
    for e in matched_edges:
        matched_entities.add(e.source)
        matched_entities.add(e.target)

    # 3) Produce a compact, deterministic "answer".
    if rel_filter_norm:
        answer = (
            f"Matched relation filter '{rel_filter}'. "
            f"Found {len(seed_entities)} seed entities and {len(matched_edges)} related edges."
        )
    else:
        answer = (
            f"Found {len(seed_entities)} relevant entities and {len(matched_edges)} related edges."
        )

    return QueryResult(
        seed_entities=seed_entities,
        matched_entities=sorted(matched_entities, key=lambda s: s.lower()),
        matched_edges=[_edge_to_dict(e) for e in matched_edges],
        answer=answer,
    )
