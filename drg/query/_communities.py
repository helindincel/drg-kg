"""Community-aware query helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._evidence import node_to_view
from ._traversal import find_paths
from ._types import CommunityView, QueryError, RelatedEntityMatch

if TYPE_CHECKING:
    from ._backend import QueryBackend

__all__ = [
    "community_neighbors",
    "community_of",
    "related_entities",
]

_EVENT_TYPES = frozenset({"event", "acquisition", "merger", "launch", "announcement"})


def community_of(backend: QueryBackend, entity_id: str) -> CommunityView | None:
    """Return the cluster containing ``entity_id``, if any."""
    if backend.get_node(entity_id) is None:
        raise QueryError(f"Entity not found: {entity_id!r}")
    cluster = backend.cluster_for(entity_id)
    if cluster is None:
        return None
    cluster_id, members = cluster
    meta: dict = {}
    raw = backend.kg.clusters.get(cluster_id)
    if raw and raw.metadata:
        meta = dict(raw.metadata)
    return CommunityView(
        cluster_id=cluster_id,
        node_ids=tuple(sorted(members, key=str.lower)),
        metadata=meta,
    )


def community_neighbors(
    backend: QueryBackend,
    entity_id: str,
    *,
    limit: int = 20,
) -> list[str]:
    """Other entities in the same cluster as ``entity_id``."""
    view = community_of(backend, entity_id)
    if view is None:
        return []
    others = [nid for nid in view.node_ids if nid != entity_id]
    others.sort(key=str.lower)
    return others[:limit]


def related_entities(
    backend: QueryBackend,
    entity_id: str,
    *,
    mode: str = "shared_neighbors",
    hops: int = 2,
    entity_type: str | None = None,
    limit: int = 10,
    include_inferred: bool = True,
) -> list[RelatedEntityMatch]:
    """Rank entities related to ``entity_id``.

    Modes:
    - ``shared_neighbors``: rank by count of shared 1-hop neighbors.
    - ``shortest_path``: rank by shortest path length (then confidence).
    - ``same_community``: rank by same-cluster membership.
    - ``degree``: rank by node degree among candidates.
    """
    if backend.get_node(entity_id) is None:
        raise QueryError(f"Entity not found: {entity_id!r}")

    mode_norm = mode.strip().lower()
    type_norm = entity_type.strip().lower() if entity_type else None

    if mode_norm == "same_community":
        return _related_by_community(backend, entity_id, type_norm, limit)

    if mode_norm == "degree":
        return _related_by_degree(backend, entity_id, type_norm, limit)

    if mode_norm == "shortest_path":
        return _related_by_shortest_path(
            backend,
            entity_id,
            hops=hops,
            entity_type=type_norm,
            limit=limit,
            include_inferred=include_inferred,
        )

    return _related_by_shared_neighbors(
        backend,
        entity_id,
        entity_type=type_norm,
        limit=limit,
        include_inferred=include_inferred,
    )


def _filter_type(backend: QueryBackend, node_id: str, type_norm: str | None) -> bool:
    if not type_norm:
        return True
    node = backend.get_node(node_id)
    if node is None:
        return False
    return (node.type or "").strip().lower() == type_norm


def _related_by_shared_neighbors(
    backend: QueryBackend,
    entity_id: str,
    *,
    entity_type: str | None,
    limit: int,
    include_inferred: bool,
) -> list[RelatedEntityMatch]:
    seed_neighbors = set(
        backend.neighbors(entity_id, direction="both", include_inferred=include_inferred)
    )
    scores: dict[str, tuple[int, list[str]]] = {}

    for candidate in backend.all_node_ids():
        if candidate == entity_id:
            continue
        if not _filter_type(backend, candidate, entity_type):
            continue
        cand_neighbors = set(
            backend.neighbors(candidate, direction="both", include_inferred=include_inferred)
        )
        shared = sorted(seed_neighbors & cand_neighbors, key=str.lower)
        if shared:
            scores[candidate] = (len(shared), shared)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[0].lower()))[:limit]
    out: list[RelatedEntityMatch] = []
    for cand, (count, shared) in ranked:
        node = backend.get_node(cand)
        if node is None:
            continue
        out.append(
            RelatedEntityMatch(
                entity=node_to_view(node, backend),
                score=float(count),
                relation_mode="shared_neighbors",
                shared_neighbors=tuple(shared),
            )
        )
    return out


def _related_by_shortest_path(
    backend: QueryBackend,
    entity_id: str,
    *,
    hops: int,
    entity_type: str | None,
    limit: int,
    include_inferred: bool,
) -> list[RelatedEntityMatch]:
    out: list[RelatedEntityMatch] = []
    for candidate in backend.all_node_ids():
        if candidate == entity_id:
            continue
        if not _filter_type(backend, candidate, entity_type):
            continue
        paths = find_paths(
            backend,
            entity_id,
            candidate,
            max_hops=hops,
            include_inferred=include_inferred,
            max_paths=1,
        )
        if not paths:
            continue
        path = paths[0]
        node = backend.get_node(candidate)
        if node is None:
            continue
        score = 1.0 / max(1, path.hop_count)
        if path.confidence is not None:
            score *= path.confidence
        out.append(
            RelatedEntityMatch(
                entity=node_to_view(node, backend),
                score=score,
                relation_mode="shortest_path",
                connecting_paths=(path,),
            )
        )

    out.sort(key=lambda m: (-m.score, m.entity.id.lower()))
    return out[:limit]


def _related_by_community(
    backend: QueryBackend,
    entity_id: str,
    entity_type: str | None,
    limit: int,
) -> list[RelatedEntityMatch]:
    view = community_of(backend, entity_id)
    if view is None:
        return []
    out: list[RelatedEntityMatch] = []
    for cand in view.node_ids:
        if cand == entity_id:
            continue
        if not _filter_type(backend, cand, entity_type):
            continue
        node = backend.get_node(cand)
        if node is None:
            continue
        out.append(
            RelatedEntityMatch(
                entity=node_to_view(node, backend),
                score=1.0,
                relation_mode="same_community",
            )
        )
    out.sort(key=lambda m: m.entity.id.lower())
    return out[:limit]


def _related_by_degree(
    backend: QueryBackend,
    entity_id: str,
    entity_type: str | None,
    limit: int,
) -> list[RelatedEntityMatch]:
    scored: list[tuple[int, str]] = []
    for cand in backend.all_node_ids():
        if cand == entity_id:
            continue
        if not _filter_type(backend, cand, entity_type):
            continue
        scored.append((backend.node_degree(cand), cand))
    scored.sort(key=lambda x: (-x[0], x[1].lower()))

    out: list[RelatedEntityMatch] = []
    for deg, cand in scored[:limit]:
        node = backend.get_node(cand)
        if node is None:
            continue
        out.append(
            RelatedEntityMatch(
                entity=node_to_view(node, backend),
                score=float(deg),
                relation_mode="degree",
            )
        )
    return out


def is_event_type(type_name: str | None) -> bool:
    if not type_name:
        return False
    return type_name.strip().lower() in _EVENT_TYPES
