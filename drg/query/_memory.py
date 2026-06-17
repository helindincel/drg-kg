"""In-memory query backend over :class:`drg.graph.kg_core.EnhancedKG`."""

from __future__ import annotations

from collections import defaultdict

from ..graph.kg_core import EnhancedKG, KGEdge
from ._backend import QueryBackend

__all__ = ["InMemoryBackend"]


def _normalize_rel(rel: str | None) -> str | None:
    if rel is None:
        return None
    s = rel.strip().lower()
    return s or None


def _is_inferred(edge: KGEdge) -> bool:
    return bool(edge.metadata) and bool(edge.metadata.get("inferred"))


class InMemoryBackend:
    """Indexed, read-only view of an :class:`EnhancedKG`.

    Builds adjacency indexes once at construction time so repeated
    neighborhood / path queries stay fast on medium-sized graphs.
    """

    def __init__(self, kg: EnhancedKG) -> None:
        self._kg = kg
        self._out: dict[str, list[KGEdge]] = defaultdict(list)
        self._in: dict[str, list[KGEdge]] = defaultdict(list)
        self._by_rel: dict[str, list[KGEdge]] = defaultdict(list)
        self._cluster_index: dict[str, tuple[str, set[str]]] = {}

        for edge in kg.edges:
            self._out[edge.source].append(edge)
            self._in[edge.target].append(edge)
            self._by_rel[_normalize_rel(edge.relationship_type) or ""].append(edge)

        for cluster_id, cluster in kg.clusters.items():
            for nid in cluster.node_ids:
                self._cluster_index[nid] = (cluster_id, set(cluster.node_ids))

    @property
    def kg(self) -> EnhancedKG:
        return self._kg

    def get_node(self, node_id: str):
        return self._kg.get_node(node_id)

    def all_node_ids(self) -> list[str]:
        return sorted(self._kg.nodes.keys(), key=str.lower)

    def all_edges(self, *, include_inferred: bool = True) -> list[KGEdge]:
        if include_inferred:
            return list(self._kg.edges)
        return [e for e in self._kg.edges if not _is_inferred(e)]

    def edges_incident(
        self,
        node_id: str,
        *,
        direction: str = "both",
        relationship_type: str | None = None,
        include_inferred: bool = True,
    ) -> list[KGEdge]:
        rel_norm = _normalize_rel(relationship_type)
        out: list[KGEdge] = []

        if direction in ("out", "both"):
            for e in self._out.get(node_id, []):
                if not include_inferred and _is_inferred(e):
                    continue
                if rel_norm and _normalize_rel(e.relationship_type) != rel_norm:
                    continue
                out.append(e)

        if direction in ("in", "both"):
            for e in self._in.get(node_id, []):
                if not include_inferred and _is_inferred(e):
                    continue
                if rel_norm and _normalize_rel(e.relationship_type) != rel_norm:
                    continue
                out.append(e)

        return out

    def edges_matching(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        relationship_type: str | None = None,
        include_inferred: bool = True,
    ) -> list[KGEdge]:
        rel_norm = _normalize_rel(relationship_type)
        candidates: list[KGEdge]

        if rel_norm:
            candidates = list(self._by_rel.get(rel_norm, []))
        else:
            candidates = list(self._kg.edges)

        out: list[KGEdge] = []
        for e in candidates:
            if not include_inferred and _is_inferred(e):
                continue
            if source is not None and e.source != source:
                continue
            if target is not None and e.target != target:
                continue
            out.append(e)
        return out

    def neighbors(
        self,
        node_id: str,
        *,
        direction: str = "both",
        relationship_type: str | None = None,
        include_inferred: bool = True,
    ) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for e in self.edges_incident(
            node_id,
            direction=direction,
            relationship_type=relationship_type,
            include_inferred=include_inferred,
        ):
            other = e.target if e.source == node_id else e.source
            if other not in seen:
                seen.add(other)
                ordered.append(other)
        ordered.sort(key=str.lower)
        return ordered

    def cluster_for(self, node_id: str) -> tuple[str, set[str]] | None:
        return self._cluster_index.get(node_id)

    def node_degree(self, node_id: str) -> int:
        return len(self.edges_incident(node_id, direction="both", include_inferred=True))
