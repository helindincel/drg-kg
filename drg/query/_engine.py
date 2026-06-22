"""Public :class:`GraphQuery` facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._analytics import degree_centrality, influence_scores, pagerank
from ._backend import QueryBackend
from ._communities import (
    community_neighbors,
    community_of,
    is_event_type,
    normalize_event_type,
    related_entities,
)
from ._evidence import edge_to_view, evidence_bundle_for_triple, node_to_view
from ._explain import build_explanation
from ._memory import InMemoryBackend
from ._search import find_entities, search_graph
from ._temporal import (
    entity_transitions,
    relations_active_at,
    role_holders_at,
    temporal_changes_between,
    temporal_conflicts,
    temporal_overlaps,
    temporal_query_text,
    temporal_timeline,
)
from ._traversal import bfs_neighborhood, find_paths, shortest_path
from ._types import (
    CommunityView,
    EdgeView,
    EntityMatch,
    EntityView,
    EventView,
    EvidenceBundle,
    Explanation,
    GraphMetricScore,
    GraphPath,
    NeighborhoodView,
    Provenance,
    QueryAnswer,
    QueryError,
    RelatedEntityMatch,
)

if TYPE_CHECKING:
    from ..graph.kg_core import EnhancedKG

__all__ = ["GraphQuery"]


class GraphQuery:
    """Evidence-first query interface over a knowledge graph.

    Usage::

        gq = GraphQuery.from_json("outputs/global_kg.json")
        gq.neighbors("Apple", hops=2)
        gq.explain("Apple", "Jimmy Iovine")
        gq.related_entities("Apple", entity_type="Company")
    """

    def __init__(
        self,
        kg: EnhancedKG,
        *,
        backend: QueryBackend | None = None,
    ) -> None:
        from ..graph.kg_core import EnhancedKG as _EnhancedKG

        if not isinstance(kg, _EnhancedKG):
            raise TypeError("GraphQuery requires an EnhancedKG instance")
        self._backend: QueryBackend = backend or InMemoryBackend(kg)

    @classmethod
    def from_json(cls, filepath: str | Path) -> GraphQuery:
        """Load a graph from JSON and return a query facade."""
        from ..graph.kg_core import EnhancedKG

        kg = EnhancedKG.load_json(str(filepath))
        return cls(kg)

    @property
    def kg(self) -> EnhancedKG:
        return self._backend.kg

    # ------------------------------------------------------------------
    # Entity lookup
    # ------------------------------------------------------------------

    def entity(self, entity_id: str) -> EntityView:
        """Exact entity lookup by id."""
        node = self._backend.get_node(entity_id)
        if node is None:
            raise QueryError(f"Entity not found: {entity_id!r}")
        return node_to_view(node, self._backend)

    def find_entities(
        self,
        query: str,
        *,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[EntityMatch]:
        """Fuzzy entity search ranked by name match."""
        return find_entities(
            self._backend,
            query,
            entity_type=entity_type,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def relations(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        relationship_type: str | None = None,
        include_inferred: bool = True,
    ) -> list[EdgeView]:
        """Filter relationships by endpoints and/or type."""
        edges = self._backend.edges_matching(
            source=source,
            target=target,
            relationship_type=relationship_type,
            include_inferred=include_inferred,
        )
        return [edge_to_view(e) for e in edges]

    def evidence_for(
        self,
        source: str,
        relationship_type: str,
        target: str,
        *,
        include_inferred: bool = True,
    ) -> EvidenceBundle:
        """All evidence supporting a specific relationship."""
        return evidence_bundle_for_triple(
            self._backend,
            (source, relationship_type, target),
            include_inferred=include_inferred,
        )

    # ------------------------------------------------------------------
    # Neighborhood & traversal
    # ------------------------------------------------------------------

    def neighbors(
        self,
        entity_id: str,
        *,
        hops: int = 1,
        direction: str = "both",
        relationship_type: str | None = None,
        include_inferred: bool = True,
        max_edges: int = 200,
    ) -> NeighborhoodView:
        """Multi-hop neighborhood expansion."""
        return bfs_neighborhood(
            self._backend,
            entity_id,
            hops=hops,
            direction=direction,
            relationship_type=relationship_type,
            include_inferred=include_inferred,
            max_edges=max_edges,
        )

    def find_paths(
        self,
        source: str,
        target: str,
        *,
        max_hops: int = 3,
        direction: str = "both",
        relationship_type: str | None = None,
        include_inferred: bool = True,
        max_paths: int = 10,
    ) -> list[GraphPath]:
        """Find paths between two entities."""
        return find_paths(
            self._backend,
            source,
            target,
            max_hops=max_hops,
            direction=direction,
            relationship_type=relationship_type,
            include_inferred=include_inferred,
            max_paths=max_paths,
        )

    def shortest_path(
        self,
        source: str,
        target: str,
        **kwargs,
    ) -> GraphPath | None:
        """Shortest path between two entities."""
        return shortest_path(self._backend, source, target, **kwargs)

    # ------------------------------------------------------------------
    # Graph analytics
    # ------------------------------------------------------------------

    def centrality(
        self,
        *,
        metric: str = "degree",
        limit: int = 10,
        include_inferred: bool = True,
    ) -> list[GraphMetricScore]:
        """Rank entities by a centrality metric."""
        if metric not in {"degree", "degree_centrality"}:
            raise QueryError(f"Unsupported centrality metric: {metric!r}")
        return degree_centrality(
            self._backend,
            limit=limit,
            include_inferred=include_inferred,
        )

    def pagerank(
        self,
        *,
        limit: int = 10,
        iterations: int = 30,
        damping: float = 0.85,
        include_inferred: bool = True,
    ) -> list[GraphMetricScore]:
        """Rank entities by PageRank."""
        return pagerank(
            self._backend,
            limit=limit,
            iterations=iterations,
            damping=damping,
            include_inferred=include_inferred,
        )

    def influence_scores(
        self,
        *,
        limit: int = 10,
        include_inferred: bool = True,
    ) -> list[GraphMetricScore]:
        """Rank entities by blended PageRank + degree influence."""
        return influence_scores(
            self._backend,
            limit=limit,
            include_inferred=include_inferred,
        )

    # ------------------------------------------------------------------
    # Explanation & events
    # ------------------------------------------------------------------

    def explain(
        self,
        source: str,
        target: str,
        *,
        max_hops: int = 3,
        max_paths: int = 5,
        include_inferred: bool = True,
    ) -> Explanation:
        """Explain why two entities are connected."""
        return build_explanation(
            self._backend,
            source,
            target,
            max_hops=max_hops,
            max_paths=max_paths,
            include_inferred=include_inferred,
        )

    def events_for(
        self,
        entity_id: str,
        *,
        event_types: tuple[str, ...] | None = None,
        include_inferred: bool = True,
    ) -> list[EventView]:
        """Return event nodes connected to ``entity_id``.

        An event is any neighbor whose ``type`` is in ``event_types``, or
        matches the built-in event type whitelist when ``event_types`` is
        ``None``.
        """
        if self._backend.get_node(entity_id) is None:
            raise QueryError(f"Entity not found: {entity_id!r}")

        allowed = {normalize_event_type(t) for t in event_types} if event_types else None

        events: list[EventView] = []
        for edge in self._backend.edges_incident(
            entity_id,
            direction="both",
            include_inferred=include_inferred,
        ):
            other = edge.target if edge.source == entity_id else edge.source
            node = self._backend.get_node(other)
            if node is None:
                continue
            type_lower = normalize_event_type(node.type)
            if allowed is not None:
                if type_lower not in allowed:
                    continue
            elif not is_event_type(node.type):
                continue

            incident = self._backend.edges_incident(
                other,
                direction="both",
                include_inferred=include_inferred,
            )
            edge_views = tuple(edge_to_view(e) for e in incident)
            prov = Provenance(
                source_documents=sorted(
                    {doc for ev in edge_views for doc in ev.provenance.source_documents}
                ),
                evidence=tuple(item for ev in edge_views for item in ev.provenance.evidence),
            )
            events.append(
                EventView(
                    event=node_to_view(node, self._backend),
                    incident_edges=edge_views,
                    provenance=prov,
                )
            )

        events.sort(key=lambda e: e.event.id.lower())
        return events

    # ------------------------------------------------------------------
    # Community & related entities
    # ------------------------------------------------------------------

    def community_of(self, entity_id: str) -> CommunityView | None:
        """Cluster membership for an entity."""
        return community_of(self._backend, entity_id)

    def community_neighbors(self, entity_id: str, *, limit: int = 20) -> list[str]:
        """Other entities in the same cluster."""
        return community_neighbors(self._backend, entity_id, limit=limit)

    def related_entities(
        self,
        entity_id: str,
        *,
        mode: str = "shared_neighbors",
        hops: int = 2,
        entity_type: str | None = None,
        limit: int = 10,
        include_inferred: bool = True,
    ) -> list[RelatedEntityMatch]:
        """Rank entities related to ``entity_id``."""
        return related_entities(
            self._backend,
            entity_id,
            mode=mode,
            hops=hops,
            entity_type=entity_type,
            limit=limit,
            include_inferred=include_inferred,
        )

    # ------------------------------------------------------------------
    # Free-text search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k_entities: int = 10,
        k_edges: int = 40,
        include_inferred: bool = True,
    ) -> QueryAnswer:
        """Deterministic free-text graph search with provenance."""
        return search_graph(
            self._backend,
            query,
            k_entities=k_entities,
            k_edges=k_edges,
            include_inferred=include_inferred,
        )

    def query(self, text: str, **kwargs) -> QueryAnswer:
        """Alias for :meth:`search` (``graph.query(...)`` ergonomics)."""
        return self.search(text, **kwargs)

    # ------------------------------------------------------------------
    # Temporal queries
    # ------------------------------------------------------------------

    def relations_active_at(
        self,
        as_of: str,
        *,
        source: str | None = None,
        target: str | None = None,
        relationship_type: str | None = None,
        include_inferred: bool = True,
    ) -> list[EdgeView]:
        """Relationships active on a given date (partial dates supported)."""
        return relations_active_at(
            self._backend,
            as_of,
            source=source,
            target=target,
            relationship_type=relationship_type,
            include_inferred=include_inferred,
        )

    def role_holders_at(
        self,
        target: str,
        relationship_type: str,
        as_of: str,
        *,
        include_inferred: bool = True,
    ) -> list[EdgeView]:
        """Entities holding a role toward ``target`` at ``as_of``.

        Example: ``gq.role_holders_at("Apple", "CEO_OF", "2008")``
        """
        return role_holders_at(
            self._backend,
            target,
            relationship_type,
            as_of,
            include_inferred=include_inferred,
        )

    def temporal_query(self, text: str, *, include_inferred: bool = True) -> list[EdgeView]:
        """Parse a compact natural temporal query.

        Example: ``gq.temporal_query("Apple CEO in 2008")``.
        """
        return temporal_query_text(
            self._backend,
            text,
            include_inferred=include_inferred,
        )

    def temporal_timeline(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        relationship_type: str | None = None,
        include_inferred: bool = True,
    ):
        """Chronological timeline of matching relationships."""
        return temporal_timeline(
            self._backend,
            source=source,
            target=target,
            relationship_type=relationship_type,
            include_inferred=include_inferred,
        )

    def changes_between(
        self,
        date_from: str,
        date_to: str,
        *,
        relationship_type: str | None = None,
        include_inferred: bool = True,
    ):
        """Relationships that started or ended between two dates."""
        return temporal_changes_between(
            self._backend,
            date_from,
            date_to,
            relationship_type=relationship_type,
            include_inferred=include_inferred,
        )

    def temporal_overlaps(self, *, include_inferred: bool = True):
        """Detect overlapping validity intervals on duplicate edges."""
        return temporal_overlaps(self._backend, include_inferred=include_inferred)

    def temporal_conflicts(
        self,
        *,
        relationship_type: str | None = None,
        target: str | None = None,
        include_inferred: bool = True,
    ):
        """Detect concurrent role holders (e.g. two CEOs at once)."""
        return temporal_conflicts(
            self._backend,
            relationship_type=relationship_type,
            target=target,
            include_inferred=include_inferred,
        )

    def entity_transitions(
        self,
        entity_id: str,
        relationship_type: str,
        *,
        direction: str = "out",
        include_inferred: bool = True,
    ):
        """How an entity's relationships of one type evolved over time."""
        return entity_transitions(
            self._backend,
            entity_id,
            relationship_type,
            direction=direction,
            include_inferred=include_inferred,
        )
