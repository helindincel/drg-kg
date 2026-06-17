"""Event extraction — first-class events for the knowledge graph.

Public surface
--------------

Data model
~~~~~~~~~~
- :class:`Event` — canonical in-memory event
- :class:`EventRole` — typed role definition
- :class:`EventTypeDefinition` — event-type schema
- :class:`EventTimestamp` — ISO 8601 temporal scope
- :class:`EventProvenance` — origin + confidence
- :class:`TextSpan` — supporting evidence span

Registry
~~~~~~~~
- :class:`EventTypeRegistry` — registry of event-type definitions
- :func:`default_event_registry` — empty registry (the canonical default)
- :func:`example_event_registry` — curated 10-type example set

Extraction
~~~~~~~~~~
- :func:`extract_events` — DSPy-backed LLM extractor

Graph mapping
~~~~~~~~~~~~~
- :func:`event_to_kg_node` / :func:`event_to_role_edges`
- :func:`events_to_kg_nodes_and_edges`
- :func:`is_event_node` / :func:`is_event_role_edge`
- :func:`event_role_from_edge` / :func:`event_from_kg_node`
- :data:`EVENT_NODE_TYPE_PREFIX` / :data:`EVENT_ROLE_EDGE_PREFIX`
"""

from __future__ import annotations

from ._extraction import extract_events
from ._graph_mapping import (
    EVENT_LOCATION_EDGE_TYPE,
    EVENT_NODE_TYPE_PREFIX,
    EVENT_ROLE_EDGE_PREFIX,
    event_from_kg_node,
    event_node_type,
    event_role_from_edge,
    event_to_kg_node,
    event_to_role_edges,
    events_to_kg_nodes_and_edges,
    is_event_node,
    is_event_role_edge,
)
from ._postprocess import (
    build_event,
    find_evidence_span,
    has_all_required_roles,
    make_event_id,
    normalize_participants,
    parse_timestamp,
    required_role_coverage,
)
from ._registry import (
    EventTypeRegistry,
    default_event_registry,
    example_event_registry,
)
from ._types import (
    Event,
    EventProvenance,
    EventRole,
    EventTimestamp,
    EventTypeDefinition,
    TextSpan,
    TimestampPrecision,
)

__all__ = [
    "EVENT_LOCATION_EDGE_TYPE",
    "EVENT_NODE_TYPE_PREFIX",
    "EVENT_ROLE_EDGE_PREFIX",
    "Event",
    "EventProvenance",
    "EventRole",
    "EventTimestamp",
    "EventTypeDefinition",
    "EventTypeRegistry",
    "TextSpan",
    "TimestampPrecision",
    "build_event",
    "default_event_registry",
    "event_from_kg_node",
    "event_node_type",
    "event_role_from_edge",
    "event_to_kg_node",
    "event_to_role_edges",
    "events_to_kg_nodes_and_edges",
    "example_event_registry",
    "extract_events",
    "find_evidence_span",
    "has_all_required_roles",
    "is_event_node",
    "is_event_role_edge",
    "make_event_id",
    "normalize_participants",
    "parse_timestamp",
    "required_role_coverage",
]
