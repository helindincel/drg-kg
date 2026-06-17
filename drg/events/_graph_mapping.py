"""Bridge events to the underlying graph storage layer.

Events live as ordinary :class:`drg.graph.kg_core.KGNode` instances with
``type='Event:<event_type>'`` plus one :class:`KGEdge` per
``(role, participant)`` pair. The prefix conventions are exposed as
module-level constants so consumers (UI, exporters) can rely on stable
naming without re-implementing the mapping.

This module is intentionally one-way at the storage level: writing an
event to the graph is canonical, reading one back is best-effort.
"""

from __future__ import annotations

from typing import Any

from ..graph.kg_core import KGEdge, KGNode
from ._types import Event, EventProvenance, EventTimestamp

__all__ = [
    "EVENT_NODE_TYPE_PREFIX",
    "EVENT_ROLE_EDGE_PREFIX",
    "EVENT_LOCATION_EDGE_TYPE",
    "event_to_kg_node",
    "event_to_role_edges",
    "events_to_kg_nodes_and_edges",
    "is_event_node",
    "event_node_type",
    "is_event_role_edge",
    "event_role_from_edge",
    "event_from_kg_node",
]


EVENT_NODE_TYPE_PREFIX = "Event:"
"""Prefix attached to ``KGNode.type`` for event nodes (e.g. ``Event:Acquisition``)."""

EVENT_ROLE_EDGE_PREFIX = "role:"
"""Prefix attached to ``KGEdge.relationship_type`` for participant edges."""

EVENT_LOCATION_EDGE_TYPE = "occurred_at"
"""Relationship type used for the optional event-to-location edge."""


def event_node_type(event_type: str) -> str:
    """Return the canonical ``KGNode.type`` string for an event type."""
    return f"{EVENT_NODE_TYPE_PREFIX}{event_type}"


def is_event_node(node: KGNode) -> bool:
    """True when ``node`` was produced from an :class:`Event`."""
    if not isinstance(node.type, str):
        return False
    if node.type.startswith(EVENT_NODE_TYPE_PREFIX):
        return True
    return bool(node.metadata.get("is_event"))


def is_event_role_edge(edge: KGEdge) -> bool:
    """True when ``edge`` represents an event-participant role link."""
    if isinstance(edge.relationship_type, str) and edge.relationship_type.startswith(
        EVENT_ROLE_EDGE_PREFIX
    ):
        return True
    return bool(edge.metadata.get("is_event_role"))


def event_role_from_edge(edge: KGEdge) -> str | None:
    """Extract the role name from a role edge, or ``None`` if not a role edge."""
    if not isinstance(edge.relationship_type, str):
        return None
    if edge.relationship_type.startswith(EVENT_ROLE_EDGE_PREFIX):
        return edge.relationship_type[len(EVENT_ROLE_EDGE_PREFIX):]
    return None


def event_to_kg_node(event: Event) -> KGNode:
    """Project an :class:`Event` onto a single ``KGNode``.

    The full event payload (participants, timestamp, location, provenance)
    is preserved on ``metadata`` so consumers that want to round-trip
    through pure graph storage can reconstruct the event with
    :func:`event_from_kg_node`.
    """
    metadata: dict[str, Any] = {
        "is_event": True,
        "event_type": event.event_type,
        "participants": {k: list(v) for k, v in event.participants.items()},
        "properties": dict(event.properties),
        "provenance": event.provenance.to_dict(),
    }
    if event.timestamp is not None and not event.timestamp.is_empty():
        metadata["timestamp"] = event.timestamp.to_dict()
    if event.location is not None:
        metadata["location"] = event.location
    if event.metadata:
        for k, v in event.metadata.items():
            metadata.setdefault(k, v)

    return KGNode(
        id=event.id,
        type=event_node_type(event.event_type),
        properties=dict(event.properties),
        metadata=metadata,
        confidence=event.provenance.confidence,
    )


def event_to_role_edges(event: Event) -> list[KGEdge]:
    """Generate one role edge per ``(role, participant)`` pair.

    A separate edge per pair (rather than a JSON list crammed into one
    edge) is what lets the rest of the graph layer — query engine,
    visualization, Neo4j export — treat events naturally.
    """
    edges: list[KGEdge] = []
    for role_name, entities in event.participants.items():
        for entity_id in entities:
            if not entity_id or entity_id == event.id:
                continue
            edge_type = f"{EVENT_ROLE_EDGE_PREFIX}{role_name}"
            detail = (
                f"{event.event_type} event '{event.id}' has {role_name}: {entity_id}"
            )
            md: dict[str, Any] = {
                "is_event_role": True,
                "event_id": event.id,
                "event_type": event.event_type,
                "role_name": role_name,
            }
            start_time, end_time = _timestamp_bounds(event.timestamp)
            edges.append(
                KGEdge(
                    source=event.id,
                    target=entity_id,
                    relationship_type=edge_type,
                    relationship_detail=detail,
                    metadata=md,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=event.provenance.confidence,
                )
            )

    if event.location:
        start_time, end_time = _timestamp_bounds(event.timestamp)
        edges.append(
            KGEdge(
                source=event.id,
                target=event.location,
                relationship_type=EVENT_LOCATION_EDGE_TYPE,
                relationship_detail=(
                    f"{event.event_type} event '{event.id}' occurred at {event.location}"
                ),
                metadata={
                    "is_event_role": True,
                    "event_id": event.id,
                    "event_type": event.event_type,
                    "role_name": "location",
                },
                start_time=start_time,
                end_time=end_time,
                confidence=event.provenance.confidence,
            )
        )

    return edges


def _timestamp_bounds(ts: EventTimestamp | None) -> tuple[str | None, str | None]:
    if ts is None or ts.is_empty():
        return None, None
    return ts.start, ts.end


def events_to_kg_nodes_and_edges(
    events: list[Event],
) -> tuple[list[KGNode], list[KGEdge]]:
    """Convenience: project a list of events to nodes + edges."""
    nodes: list[KGNode] = []
    edges: list[KGEdge] = []
    for event in events:
        nodes.append(event_to_kg_node(event))
        edges.extend(event_to_role_edges(event))
    return nodes, edges


def event_from_kg_node(node: KGNode) -> Event | None:
    """Best-effort reconstruction of an :class:`Event` from a ``KGNode``.

    Returns ``None`` when the node was not produced by this mapping.
    Used by serialisers that want to surface events as their own list
    in JSON output without re-running extraction.
    """
    if not is_event_node(node):
        return None
    md = node.metadata or {}
    event_type = md.get("event_type")
    if not isinstance(event_type, str) or not event_type.strip():
        if isinstance(node.type, str) and node.type.startswith(EVENT_NODE_TYPE_PREFIX):
            event_type = node.type[len(EVENT_NODE_TYPE_PREFIX):]
        else:
            return None

    participants_raw = md.get("participants") or {}
    participants: dict[str, list[str]] = {}
    if isinstance(participants_raw, dict):
        for k, v in participants_raw.items():
            if isinstance(v, list):
                participants[str(k)] = [str(x) for x in v]
            elif isinstance(v, str):
                participants[str(k)] = [v]

    ts_raw = md.get("timestamp")
    timestamp = (
        EventTimestamp.from_dict(ts_raw) if isinstance(ts_raw, dict) else None
    )

    prov_raw = md.get("provenance") or {}
    provenance = (
        EventProvenance.from_dict(prov_raw)
        if isinstance(prov_raw, dict)
        else EventProvenance(confidence=node.confidence or 1.0)
    )

    return Event(
        id=node.id,
        event_type=event_type,
        participants=participants,
        timestamp=timestamp,
        location=md.get("location") if isinstance(md.get("location"), str) else None,
        properties=dict(md.get("properties") or node.properties or {}),
        provenance=provenance,
        metadata={
            k: v
            for k, v in md.items()
            if k
            not in {
                "is_event",
                "event_type",
                "participants",
                "timestamp",
                "location",
                "provenance",
                "properties",
            }
        },
    )
