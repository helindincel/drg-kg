"""Partial-date comparison and interval logic for temporal queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import PartialDate, TemporalScope

if TYPE_CHECKING:
    from ..graph.kg_core import KGEdge

__all__ = [
    "is_active_at",
    "intervals_overlap",
    "compare_partial_dates",
    "scope_bounds",
]


def _as_partial(value: str | None, precision: str | None = None) -> PartialDate | None:
    if value is None:
        return None
    if precision:
        return PartialDate(value=value, precision=precision)  # type: ignore[arg-type]
    return PartialDate.parse(value)


def scope_bounds(scope: TemporalScope | None) -> tuple[str | None, str | None]:
    """Return inclusive (start, end) bounds for a scope.

    Open-ended scopes use ``None`` for the missing bound.
    """
    if scope is None:
        return None, None
    scoped = scope.with_precision_defaults()
    start = None
    end = None
    if scoped.valid_from:
        pd = _as_partial(scoped.valid_from, scoped.precision_from)
        start = pd.interval_start() if pd else scoped.valid_from
    if scoped.valid_to:
        pd = _as_partial(scoped.valid_to, scoped.precision_to)
        end = pd.interval_end() if pd else scoped.valid_to
    return start, end


def compare_partial_dates(a: str, b: str) -> int:
    """Compare two partial dates. Returns -1, 0, or 1."""
    pa = PartialDate.parse(a)
    pb = PartialDate.parse(b)
    if pa is None or pb is None:
        return (a > b) - (a < b)
    sa, ea = pa.interval_start(), pa.interval_end()
    sb, eb = pb.interval_start(), pb.interval_end()
    if sa >= sb and ea <= eb:
        return 0
    if sb >= sa and eb <= ea:
        return 0
    if ea < sb:
        return -1
    if eb < sa:
        return 1
    if sa < sb:
        return -1
    if sa > sb:
        return 1
    return 0


def is_active_at(scope: TemporalScope | None, as_of: str) -> bool:
    """Return whether ``scope`` was active at ``as_of`` (partial date allowed)."""
    if scope is None:
        return True  # atemporal facts are always active
    start, end = scope_bounds(scope)
    point = PartialDate.parse(as_of)
    if point is None:
        return False
    p_start = point.interval_start()
    p_end = point.interval_end()
    if start and p_end < start:
        return False
    if end and p_start > end:
        return False
    return True


def intervals_overlap(a: TemporalScope | None, b: TemporalScope | None) -> bool:
    """Return True when two scopes overlap in time (both open = overlap)."""
    if a is None or b is None:
        return True
    a_start, a_end = scope_bounds(a)
    b_start, b_end = scope_bounds(b)
    if a_start and b_end and b_end < a_start:
        return False
    if b_start and a_end and a_end < b_start:
        return False
    return True


def edge_temporal_scope(edge: KGEdge) -> TemporalScope | None:
    """Extract temporal scope from a :class:`KGEdge`."""
    from ._types import temporal_from_edge_fields

    return temporal_from_edge_fields(
        start_time=edge.start_time,
        end_time=edge.end_time,
        created_at=getattr(edge, "created_at", None),
        updated_at=getattr(edge, "updated_at", None),
        metadata=edge.metadata,
    )


def edge_is_active_at(edge: KGEdge, as_of: str) -> bool:
    return is_active_at(edge_temporal_scope(edge), as_of)
