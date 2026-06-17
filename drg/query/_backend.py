"""Backend protocol for graph querying.

The query layer is deliberately storage-agnostic. The default
:class:`InMemoryBackend` wraps :class:`drg.graph.kg_core.EnhancedKG`; a future
:class:`Neo4jBackend` can implement the same protocol without changing the
public :class:`GraphQuery` facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..graph.kg_core import EnhancedKG, KGEdge, KGNode

__all__ = ["QueryBackend"]


@runtime_checkable
class QueryBackend(Protocol):
    """Structural interface every graph query backend must satisfy."""

    @property
    def kg(self) -> EnhancedKG:
        """Underlying knowledge graph (read-only from the caller's perspective)."""
        ...

    def get_node(self, node_id: str) -> KGNode | None:
        """Return a node by exact id, or ``None``."""
        ...

    def all_node_ids(self) -> list[str]:
        """All node ids in stable order."""
        ...

    def all_edges(
        self,
        *,
        include_inferred: bool = True,
    ) -> list[KGEdge]:
        """All edges, optionally filtering inferred ones."""
        ...

    def edges_incident(
        self,
        node_id: str,
        *,
        direction: str = "both",
        relationship_type: str | None = None,
        include_inferred: bool = True,
    ) -> list[KGEdge]:
        """Edges touching ``node_id`` with optional direction/type filters."""
        ...

    def edges_matching(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        relationship_type: str | None = None,
        include_inferred: bool = True,
    ) -> list[KGEdge]:
        """Filter edges by endpoint and/or relation type (case-insensitive type)."""
        ...

    def neighbors(
        self,
        node_id: str,
        *,
        direction: str = "both",
        relationship_type: str | None = None,
        include_inferred: bool = True,
    ) -> list[str]:
        """Direct neighbor node ids."""
        ...

    def cluster_for(self, node_id: str) -> tuple[str, set[str]] | None:
        """Return ``(cluster_id, member_ids)`` for ``node_id``, or ``None``."""
        ...

    def node_degree(self, node_id: str) -> int:
        """Undirected degree of ``node_id``."""
        ...
