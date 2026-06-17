"""Temporal knowledge graph support.

Provides partial-date metadata, interval comparison, temporal queries,
and reasoning (overlap detection, conflict detection, timelines).

Quick example::

    from drg.temporal import TemporalScope, is_active_at

    scope = TemporalScope(valid_from="2011", valid_to=None)
    assert is_active_at(scope, "2011-06")

See ``docs/temporal_graph.md`` for the full design and migration guide.
"""

from __future__ import annotations

from ._compare import compare_partial_dates, intervals_overlap, is_active_at, scope_bounds
from ._migrate import migrate_edge_dict, migrate_node_dict
from ._reasoning import (
    ChangesReport,
    OverlapConflict,
    TemporalConflict,
    Timeline,
    TimelineEntry,
    build_timeline,
    changes_between,
    detect_conflicts,
    detect_overlaps,
    entity_state_transitions,
    filter_edges_active_at,
)
from ._types import (
    PartialDate,
    TemporalPrecision,
    TemporalScope,
    temporal_from_edge_fields,
    temporal_to_edge_fields,
)

__all__ = [
    "ChangesReport",
    "OverlapConflict",
    "PartialDate",
    "TemporalConflict",
    "TemporalPrecision",
    "TemporalScope",
    "Timeline",
    "TimelineEntry",
    "build_timeline",
    "changes_between",
    "compare_partial_dates",
    "detect_conflicts",
    "detect_overlaps",
    "entity_state_transitions",
    "filter_edges_active_at",
    "intervals_overlap",
    "is_active_at",
    "migrate_edge_dict",
    "migrate_node_dict",
    "scope_bounds",
    "temporal_from_edge_fields",
    "temporal_to_edge_fields",
]
