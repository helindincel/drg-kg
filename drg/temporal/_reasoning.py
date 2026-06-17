"""Temporal reasoning over knowledge graph edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._compare import edge_is_active_at, edge_temporal_scope, intervals_overlap, is_active_at
from ._types import TemporalScope

if False:  # pragma: no cover - typing only
    from ..graph.kg_core import KGEdge


__all__ = [
    "OverlapConflict",
    "TemporalConflict",
    "TimelineEntry",
    "Timeline",
    "ChangesReport",
    "detect_overlaps",
    "detect_conflicts",
    "entity_state_transitions",
    "build_timeline",
    "changes_between",
]


@dataclass(frozen=True)
class OverlapConflict:
    """Two edges of the same type between the same endpoints overlap in time."""

    source: str
    target: str
    relationship_type: str
    edge_indices: tuple[int, ...]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type,
            "edge_indices": list(self.edge_indices),
            "message": self.message,
        }


@dataclass(frozen=True)
class TemporalConflict:
    """Potentially contradictory facts (e.g. two CEOs without gap)."""

    conflict_type: str
    edges: tuple[dict[str, Any], ...]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_type": self.conflict_type,
            "edges": list(self.edges),
            "message": self.message,
        }


@dataclass(frozen=True)
class TimelineEntry:
    """One fact on a timeline."""

    source: str
    target: str
    relationship_type: str
    temporal: TemporalScope | None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type,
        }
        if self.temporal is not None:
            out["temporal"] = self.temporal.to_dict()
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


@dataclass(frozen=True)
class Timeline:
    """Ordered sequence of temporal facts."""

    subject: str | None
    entries: tuple[TimelineEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass(frozen=True)
class ChangesReport:
    """Facts that started or ended between two dates."""

    from_date: str
    to_date: str
    started: tuple[TimelineEntry, ...]
    ended: tuple[TimelineEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_date": self.from_date,
            "to_date": self.to_date,
            "started": [e.to_dict() for e in self.started],
            "ended": [e.to_dict() for e in self.ended],
        }


def _edge_key(edge: KGEdge) -> tuple[str, str, str]:
    return (edge.source, edge.relationship_type, edge.target)


def _is_handoff(a: TemporalScope | None, b: TemporalScope | None) -> bool:
    """True when one scope ends exactly when the other begins (role succession)."""
    if a is None or b is None:
        return False
    a_scoped = a.with_precision_defaults()
    b_scoped = b.with_precision_defaults()
    if a_scoped.valid_to and b_scoped.valid_from:
        if a_scoped.valid_to == b_scoped.valid_from:
            return True
    if b_scoped.valid_to and a_scoped.valid_from:
        if b_scoped.valid_to == a_scoped.valid_from:
            return True
    return False


def _edge_summary(edge: KGEdge, index: int) -> dict[str, Any]:
    scope = edge_temporal_scope(edge)
    return {
        "index": index,
        "source": edge.source,
        "target": edge.target,
        "relationship_type": edge.relationship_type,
        "temporal": scope.to_dict() if scope else None,
        "confidence": edge.confidence,
    }


def detect_overlaps(edges: list[KGEdge]) -> list[OverlapConflict]:
    """Find same-triple edges whose validity intervals overlap."""
    groups: dict[tuple[str, str, str], list[tuple[int, KGEdge]]] = {}
    for i, edge in enumerate(edges):
        if edge.is_negated:
            continue
        groups.setdefault(_edge_key(edge), []).append((i, edge))

    conflicts: list[OverlapConflict] = []
    for (src, rel, tgt), items in groups.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                idx_a, edge_a = items[i]
                idx_b, edge_b = items[j]
                if intervals_overlap(
                    edge_temporal_scope(edge_a),
                    edge_temporal_scope(edge_b),
                ):
                    conflicts.append(
                        OverlapConflict(
                            source=src,
                            target=tgt,
                            relationship_type=rel,
                            edge_indices=(idx_a, idx_b),
                            message=(
                                f"Overlapping '{rel}' edges between {src!r} and {tgt!r}"
                            ),
                        )
                    )
    return conflicts


def detect_conflicts(
    edges: list[KGEdge],
    *,
    relationship_type: str | None = None,
    target: str | None = None,
) -> list[TemporalConflict]:
    """Detect multiple simultaneous holders of a functional role.

  Example: two different ``CEO_OF`` edges to the same company active at
  overlapping times.
    """
    rel_norm = (relationship_type or "").strip().lower()
    tgt_norm = (target or "").strip().lower() if target else None

    by_target: dict[tuple[str, str], list[tuple[int, KGEdge]]] = {}
    for i, edge in enumerate(edges):
        if edge.is_negated:
            continue
        if rel_norm and edge.relationship_type.strip().lower() != rel_norm:
            continue
        if tgt_norm and edge.target.strip().lower() != tgt_norm:
            continue
        by_target.setdefault((edge.relationship_type, edge.target), []).append((i, edge))

    conflicts: list[TemporalConflict] = []
    for (rel, tgt), items in by_target.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                idx_a, edge_a = items[i]
                idx_b, edge_b = items[j]
                if edge_a.source == edge_b.source:
                    continue
                scope_a = edge_temporal_scope(edge_a)
                scope_b = edge_temporal_scope(edge_b)
                if _is_handoff(scope_a, scope_b):
                    continue
                if intervals_overlap(scope_a, scope_b):
                    conflicts.append(
                        TemporalConflict(
                            conflict_type="concurrent_role_holders",
                            edges=(
                                _edge_summary(edge_a, idx_a),
                                _edge_summary(edge_b, idx_b),
                            ),
                            message=(
                                f"Concurrent '{rel}' holders for {tgt!r}: "
                                f"{edge_a.source!r} and {edge_b.source!r}"
                            ),
                        )
                    )
    return conflicts


def _sort_key(entry: TimelineEntry) -> tuple[str, str]:
    scope = entry.temporal
    if scope and scope.valid_from:
        return (scope.valid_from, entry.source)
    return ("", entry.source)


def build_timeline(
    edges: list[KGEdge],
    *,
    source: str | None = None,
    target: str | None = None,
    relationship_type: str | None = None,
) -> Timeline:
    """Build a chronological timeline from matching edges."""
    entries: list[TimelineEntry] = []
    for edge in edges:
        if source is not None and edge.source != source:
            continue
        if target is not None and edge.target != target:
            continue
        if relationship_type is not None and edge.relationship_type != relationship_type:
            continue
        entries.append(
            TimelineEntry(
                source=edge.source,
                target=edge.target,
                relationship_type=edge.relationship_type,
                temporal=edge_temporal_scope(edge),
                confidence=edge.confidence,
                metadata=dict(edge.metadata) if edge.metadata else {},
            )
        )
    entries.sort(key=_sort_key)
    subject = source or target
    return Timeline(subject=subject, entries=tuple(entries))


def entity_state_transitions(
    edges: list[KGEdge],
    entity_id: str,
    relationship_type: str,
    *,
    direction: str = "out",
) -> Timeline:
    """Timeline of how an entity's relationships of one type evolved."""
    filtered: list[KGEdge] = []
    for edge in edges:
        if edge.relationship_type != relationship_type:
            continue
        if direction == "out" and edge.source == entity_id:
            filtered.append(edge)
        elif direction == "in" and edge.target == entity_id:
            filtered.append(edge)
        elif direction == "both" and (edge.source == entity_id or edge.target == entity_id):
            filtered.append(edge)
    return build_timeline(filtered, source=entity_id if direction != "in" else None)


def changes_between(
    edges: list[KGEdge],
    date_from: str,
    date_to: str,
    *,
    relationship_type: str | None = None,
) -> ChangesReport:
    """Facts that became active or inactive between two dates."""
    started: list[TimelineEntry] = []
    ended: list[TimelineEntry] = []

    for edge in edges:
        if relationship_type and edge.relationship_type != relationship_type:
            continue
        scope = edge_temporal_scope(edge)
        entry = TimelineEntry(
            source=edge.source,
            target=edge.target,
            relationship_type=edge.relationship_type,
            temporal=scope,
            confidence=edge.confidence,
        )
        vf = scope.valid_from if scope else None
        vt = scope.valid_to if scope else None
        if vf and is_active_at(TemporalScope(valid_from=vf), date_to):
            if not is_active_at(TemporalScope(valid_from=vf), date_from):
                started.append(entry)
        if vt and is_active_at(TemporalScope(valid_to=vt), date_from):
            if not is_active_at(TemporalScope(valid_to=vt), date_to):
                ended.append(entry)

    return ChangesReport(
        from_date=date_from,
        to_date=date_to,
        started=tuple(started),
        ended=tuple(ended),
    )


def filter_edges_active_at(edges: list[KGEdge], as_of: str) -> list[KGEdge]:
    """Return edges active at ``as_of``."""
    return [e for e in edges if edge_is_active_at(e, as_of)]
