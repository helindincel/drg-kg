"""Temporal query helpers over the in-memory graph backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..temporal import (
    ChangesReport,
    Timeline,
    build_timeline,
    changes_between,
    detect_conflicts,
    detect_overlaps,
    entity_state_transitions,
    filter_edges_active_at,
)
from ._evidence import edge_to_view
from ._types import EdgeView

if TYPE_CHECKING:
    from ._backend import QueryBackend

__all__ = [
    "relations_active_at",
    "role_holders_at",
    "temporal_changes_between",
    "temporal_conflicts",
    "temporal_overlaps",
    "temporal_timeline",
]


def relations_active_at(
    backend: QueryBackend,
    as_of: str,
    *,
    source: str | None = None,
    target: str | None = None,
    relationship_type: str | None = None,
    include_inferred: bool = True,
) -> list[EdgeView]:
    """Return relationships that were active on ``as_of``."""
    candidates = backend.edges_matching(
        source=source,
        target=target,
        relationship_type=relationship_type,
        include_inferred=include_inferred,
    )
    active = filter_edges_active_at(candidates, as_of)
    return [edge_to_view(e) for e in active]


def role_holders_at(
    backend: QueryBackend,
    target: str,
    relationship_type: str,
    as_of: str,
    *,
    include_inferred: bool = True,
) -> list[EdgeView]:
    """Who held ``relationship_type`` toward ``target`` at ``as_of``?

    Example: ``role_holders_at(kg, "Apple", "CEO_OF", "2008")``
    """
    return relations_active_at(
        backend,
        as_of,
        target=target,
        relationship_type=relationship_type,
        include_inferred=include_inferred,
    )


def temporal_timeline(
    backend: QueryBackend,
    *,
    source: str | None = None,
    target: str | None = None,
    relationship_type: str | None = None,
    include_inferred: bool = True,
) -> Timeline:
    """Build a chronological timeline for matching edges."""
    edges = backend.edges_matching(
        source=source,
        target=target,
        relationship_type=relationship_type,
        include_inferred=include_inferred,
    )
    return build_timeline(
        edges,
        source=source,
        target=target,
        relationship_type=relationship_type,
    )


def temporal_changes_between(
    backend: QueryBackend,
    date_from: str,
    date_to: str,
    *,
    relationship_type: str | None = None,
    include_inferred: bool = True,
) -> ChangesReport:
    """Facts that started or ended between two dates."""
    edges = backend.edges_matching(
        relationship_type=relationship_type,
        include_inferred=include_inferred,
    )
    return changes_between(edges, date_from, date_to, relationship_type=relationship_type)


def temporal_overlaps(
    backend: QueryBackend,
    *,
    include_inferred: bool = True,
):
    return detect_overlaps(backend.all_edges(include_inferred=include_inferred))


def temporal_conflicts(
    backend: QueryBackend,
    *,
    relationship_type: str | None = None,
    target: str | None = None,
    include_inferred: bool = True,
):
    return detect_conflicts(
        backend.all_edges(include_inferred=include_inferred),
        relationship_type=relationship_type,
        target=target,
    )


def entity_transitions(
    backend: QueryBackend,
    entity_id: str,
    relationship_type: str,
    *,
    direction: str = "out",
    include_inferred: bool = True,
) -> Timeline:
    edges = backend.all_edges(include_inferred=include_inferred)
    return entity_state_transitions(
        edges,
        entity_id,
        relationship_type,
        direction=direction,
    )
