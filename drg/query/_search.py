"""Entity and relationship search helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ._evidence import edge_to_view, node_to_view
from ._types import EntityMatch, QueryAnswer
from .query_engine_compat import execute_query_compat

if TYPE_CHECKING:
    from ._backend import QueryBackend

__all__ = ["find_entities", "search_graph"]

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
    "all",
    "show",
    "find",
    "related",
    "involving",
    "connected",
    "within",
    "hops",
    "hop",
    "entities",
    "entity",
    "events",
    "event",
    "companies",
    "company",
}


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _tokenize(query: str) -> list[str]:
    q = _normalize(query)
    tokens = [t for t in q.replace("(", " ").replace(")", " ").replace(",", " ").split() if t]
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _score_entity(
    entity_id: str,
    query_norm: str,
    query_tokens: Sequence[str],
) -> tuple[float, tuple[str, ...]]:
    eid_norm = _normalize(entity_id)
    reasons: list[str] = []
    score = 0.0

    if not eid_norm:
        return 0.0, ()

    if eid_norm == query_norm:
        return 5.0, ("exact_match",)

    if query_norm and query_norm in eid_norm:
        score += 3.0
        reasons.append("substring_match")

    if query_tokens:
        eid_tokens = set(eid_norm.split())
        overlap = len(eid_tokens.intersection(query_tokens))
        if overlap > 0:
            score += 1.0 + 0.4 * overlap
            reasons.append(f"token_overlap:{overlap}")

    return score, tuple(reasons)


def _score_aliases(
    aliases: Sequence[str],
    query_norm: str,
    query_tokens: Sequence[str],
) -> tuple[float, tuple[str, ...]]:
    best_score = 0.0
    best_reasons: tuple[str, ...] = ()
    for alias in aliases:
        score, reasons = _score_entity(alias, query_norm, query_tokens)
        if score > best_score:
            best_score = score
            best_reasons = tuple(f"alias_{reason}" for reason in reasons)
    return best_score, best_reasons


def find_entities(
    backend: QueryBackend,
    query: str,
    *,
    entity_type: str | None = None,
    limit: int = 10,
) -> list[EntityMatch]:
    """Rank entities by name match against ``query``."""
    query_norm = _normalize(query)
    query_tokens = _tokenize(query)
    type_norm = _normalize(entity_type) if entity_type else None

    scored: list[tuple[float, str, tuple[str, ...]]] = []
    for node_id in backend.all_node_ids():
        node = backend.get_node(node_id)
        if node is None:
            continue
        if type_norm and _normalize(node.type or "") != type_norm:
            continue
        s, reasons = _score_entity(node_id, query_norm, query_tokens)
        alias_score, alias_reasons = _score_aliases(
            [str(a) for a in (node.metadata.get("aliases", []) or [])],
            query_norm,
            query_tokens,
        )
        if alias_score > s:
            s = alias_score
            reasons = alias_reasons
        if s > 0:
            scored.append((s, node_id, reasons))

    scored.sort(key=lambda x: (-x[0], x[1].lower()))

    if not scored:
        degree_pairs = [
            (backend.node_degree(nid), nid)
            for nid in backend.all_node_ids()
            if backend.get_node(nid) is not None
            and (
                not type_norm or _normalize(backend.get_node(nid).type or "") == type_norm  # type: ignore[union-attr]
            )
        ]
        degree_pairs.sort(key=lambda x: (-x[0], x[1].lower()))
        scored = [(float(deg), nid, ("top_degree_fallback",)) for deg, nid in degree_pairs[:limit]]

    matches: list[EntityMatch] = []
    for score, node_id, reasons in scored[: max(1, limit)]:
        node = backend.get_node(node_id)
        if node is None:
            continue
        matches.append(
            EntityMatch(
                entity=node_to_view(node, backend),
                score=score,
                match_reasons=reasons,
            )
        )
    return matches


def search_graph(
    backend: QueryBackend,
    query: str,
    *,
    k_entities: int = 10,
    k_edges: int = 40,
    include_inferred: bool = True,
) -> QueryAnswer:
    """Free-text graph search with provenance-enriched edges.

      Reuses the deterministic matching logic from ``drg.graph.query_engine``
      for seed selection, then enriches results with full :class:`EdgeView`
    objects.
    """
    from ._types import Provenance

    result = execute_query_compat(backend.kg, query, k_entities=k_entities, k_edges=k_edges)

    edge_views = []
    matched_entities: set[str] = set(result.seed_entities)
    provenances = []

    for edge_dict in result.matched_edges:
        src = edge_dict["source"]
        tgt = edge_dict["target"]
        rel = edge_dict["relationship_type"]
        candidates = backend.edges_matching(
            source=src, target=tgt, include_inferred=include_inferred
        )
        rel_norm = rel.strip().lower()
        picked = None
        for e in candidates:
            if e.relationship_type.lower() == rel_norm:
                if not include_inferred and e.metadata.get("inferred"):
                    continue
                picked = e
                break
        if picked is None and candidates:
            picked = candidates[0]
        if picked is not None:
            view = edge_to_view(picked)
            edge_views.append(view)
            provenances.append(view.provenance)
            matched_entities.add(src)
            matched_entities.add(tgt)

    prov = merge_provenance_safe(provenances) if provenances else Provenance()

    return QueryAnswer(
        query=query,
        seed_entities=tuple(result.seed_entities),
        matched_entities=tuple(sorted(matched_entities, key=str.lower)),
        edges=tuple(edge_views),
        answer=result.answer,
        provenance=prov,
    )


def merge_provenance_safe(provenances):
    from ._evidence import merge_provenance

    return merge_provenance(*provenances)
