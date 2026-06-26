"""Core data types for the DRG query & reasoning layer.

Every type here is **pure data** — no graph imports, no engine imports —
so backends, traversal, and the public facade can share a single lightweight
surface without cycles.

Design notes
------------
- All result types are frozen dataclasses with ``.to_dict()`` for JSON export.
- Every answer that cites graph facts carries :class:`EvidenceItem` entries so
  callers can trace results back to nodes, edges, events, or source documents.
- ``QueryError`` is raised (never silently swallowed) when a lookup cannot be
  grounded in the graph — the layer never fabricates unsupported answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CommunityView",
    "EdgeView",
    "EntityMatch",
    "EntityView",
    "EventView",
    "EvidenceBundle",
    "EvidenceItem",
    "Explanation",
    "GraphMetricScore",
    "GraphPath",
    "NeighborhoodView",
    "Provenance",
    "QueryAnswer",
    "QueryError",
    "RelatedEntityMatch",
]


class QueryError(ValueError):
    """Raised when a query cannot be answered from graph facts alone."""


# ---------------------------------------------------------------------------
# Evidence & provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceItem:
    """A single piece of graph-grounded evidence."""

    triple: tuple[str, str, str]
    source_ref: str | None = None
    snippet: str | None = None
    confidence: float | None = None
    is_inferred: bool = False
    inference: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"triple": list(self.triple)}
        if self.source_ref is not None:
            out["source_ref"] = self.source_ref
        if self.snippet:
            out["snippet"] = self.snippet
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.is_inferred:
            out["is_inferred"] = True
        if self.inference:
            out["inference"] = dict(self.inference)
        return out


@dataclass(frozen=True)
class Provenance:
    """Aggregated provenance for a query result."""

    source_documents: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    cluster_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_documents": list(self.source_documents),
            "evidence": [e.to_dict() for e in self.evidence],
            "cluster_id": self.cluster_id,
        }


# ---------------------------------------------------------------------------
# Entity & edge views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityView:
    """Resolved entity with full node metadata."""

    id: str
    type: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    cluster_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "type": self.type}
        if self.properties:
            out["properties"] = dict(self.properties)
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.cluster_id:
            out["cluster_id"] = self.cluster_id
        return out


@dataclass(frozen=True)
class EntityMatch:
    """Entity lookup hit with relevance score."""

    entity: EntityView
    score: float
    match_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity.to_dict(),
            "score": self.score,
            "match_reasons": list(self.match_reasons),
        }


@dataclass(frozen=True)
class EdgeView:
    """Relationship view with provenance."""

    source: str
    target: str
    relationship_type: str
    relationship_detail: str
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    is_inferred: bool = False
    start_time: str | None = None
    end_time: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    is_negated: bool = False
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type,
            "relationship_detail": self.relationship_detail,
            "metadata": dict(self.metadata),
            "is_inferred": self.is_inferred,
            "provenance": self.provenance.to_dict(),
        }
        if self.confidence is not None:
            out["confidence"] = self.confidence
        vf = self.valid_from or self.start_time
        vt = self.valid_to or self.end_time
        if vf:
            out["valid_from"] = vf
            out["start_time"] = vf
        if vt:
            out["valid_to"] = vt
            out["end_time"] = vt
        if self.created_at:
            out["created_at"] = self.created_at
        if self.updated_at:
            out["updated_at"] = self.updated_at
        if self.is_negated:
            out["is_negated"] = True
        return out

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.source, self.relationship_type, self.target)


# ---------------------------------------------------------------------------
# Paths & neighborhoods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphPath:
    """A multi-hop path through the graph."""

    nodes: tuple[str, ...]
    edges: tuple[EdgeView, ...]
    confidence: float | None = None
    hop_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hop_count", max(0, len(self.edges)))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "nodes": list(self.nodes),
            "edges": [e.to_dict() for e in self.edges],
            "hop_count": self.hop_count,
        }
        if self.confidence is not None:
            out["confidence"] = self.confidence
        return out


@dataclass(frozen=True)
class GraphMetricScore:
    """Ranked graph analytics score for one entity."""

    entity: EntityView
    metric: str
    score: float
    rank: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity.to_dict(),
            "metric": self.metric,
            "score": self.score,
            "rank": self.rank,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class NeighborhoodView:
    """Entities and edges reachable within N hops from a seed."""

    seed: str
    hops: int
    entities: tuple[EntityView, ...]
    edges: tuple[EdgeView, ...]
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "hops": self.hops,
            "entities": [e.to_dict() for e in self.entities],
            "edges": [e.to_dict() for e in self.edges],
            "provenance": self.provenance.to_dict(),
        }


# ---------------------------------------------------------------------------
# Explanation & events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Explanation:
    """Explainable answer for why two entities are connected."""

    source: str
    target: str
    connected: bool
    paths: tuple[GraphPath, ...]
    evidence: tuple[EvidenceItem, ...]
    shared_community: CommunityView | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "connected": self.connected,
            "paths": [p.to_dict() for p in self.paths],
            "evidence": [e.to_dict() for e in self.evidence],
            "shared_community": (
                self.shared_community.to_dict() if self.shared_community else None
            ),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class EventView:
    """An event node and its incident relationships involving an entity."""

    event: EntityView
    incident_edges: tuple[EdgeView, ...]
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "incident_edges": [e.to_dict() for e in self.incident_edges],
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class CommunityView:
    """Community/cluster membership view."""

    cluster_id: str
    node_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "node_ids": list(self.node_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """All evidence supporting a specific relationship."""

    triple: tuple[str, str, str]
    edges: tuple[EdgeView, ...]
    evidence: tuple[EvidenceItem, ...]
    source_documents: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "triple": list(self.triple),
            "edges": [e.to_dict() for e in self.edges],
            "evidence": [e.to_dict() for e in self.evidence],
            "source_documents": list(self.source_documents),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class RelatedEntityMatch:
    """Entity related to a seed via a specific ranking mode."""

    entity: EntityView
    score: float
    relation_mode: str
    connecting_paths: tuple[GraphPath, ...] = ()
    shared_neighbors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "entity": self.entity.to_dict(),
            "score": self.score,
            "relation_mode": self.relation_mode,
        }
        if self.connecting_paths:
            out["connecting_paths"] = [p.to_dict() for p in self.connecting_paths]
        if self.shared_neighbors:
            out["shared_neighbors"] = list(self.shared_neighbors)
        return out


@dataclass(frozen=True)
class QueryAnswer:
    """Result of a free-text graph search."""

    query: str
    seed_entities: tuple[str, ...]
    matched_entities: tuple[str, ...]
    edges: tuple[EdgeView, ...]
    answer: str
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "seed_entities": list(self.seed_entities),
            "matched_entities": list(self.matched_entities),
            "edges": [e.to_dict() for e in self.edges],
            "answer": self.answer,
            "provenance": self.provenance.to_dict(),
        }
