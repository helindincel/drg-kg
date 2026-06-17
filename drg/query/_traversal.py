"""Graph traversal: multi-hop neighborhoods and path finding."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from ._evidence import edge_to_view, merge_provenance, node_to_view
from ._types import GraphPath, NeighborhoodView, Provenance, QueryError

if TYPE_CHECKING:
    from ..graph.kg_core import KGEdge
    from ._backend import QueryBackend

__all__ = ["bfs_neighborhood", "find_paths", "shortest_path"]


def _path_confidence(edges: list) -> float | None:
    confs = [e.confidence for e in edges if e.confidence is not None]
    if not confs:
        return None
    product = 1.0
    for c in confs:
        product *= c
    return max(0.0, min(1.0, product))


def _expand_frontier(
    backend: QueryBackend,
    current: str,
    *,
    direction: str,
    relationship_type: str | None,
    include_inferred: bool,
) -> list[tuple[str, KGEdge]]:
    """Return ``(neighbor, edge)`` pairs from ``current``."""
    pairs: list[tuple[str, KGEdge]] = []
    for edge in backend.edges_incident(
        current,
        direction=direction,
        relationship_type=relationship_type,
        include_inferred=include_inferred,
    ):
        neighbor = edge.target if edge.source == current else edge.source
        pairs.append((neighbor, edge))
    pairs.sort(key=lambda p: (p[0].lower(), p[1].relationship_type.lower()))
    return pairs


def bfs_neighborhood(
    backend: QueryBackend,
    seed: str,
    *,
    hops: int = 1,
    direction: str = "both",
    relationship_type: str | None = None,
    include_inferred: bool = True,
    max_edges: int = 200,
) -> NeighborhoodView:
    """Collect entities and edges within ``hops`` of ``seed``."""
    if backend.get_node(seed) is None:
        raise QueryError(f"Entity not found: {seed!r}")
    if hops < 1:
        raise QueryError("hops must be >= 1")

    visited_nodes: set[str] = {seed}
    collected_edges: list[KGEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()

    frontier: deque[tuple[str, int]] = deque([(seed, 0)])

    while frontier and len(collected_edges) < max_edges:
        node, depth = frontier.popleft()
        if depth >= hops:
            continue

        for neighbor, edge in _expand_frontier(
            backend,
            node,
            direction=direction,
            relationship_type=relationship_type,
            include_inferred=include_inferred,
        ):
            key = (edge.source, edge.relationship_type.lower(), edge.target)
            if key not in edge_keys:
                edge_keys.add(key)
                collected_edges.append(edge)
                if len(collected_edges) >= max_edges:
                    break

            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                if depth + 1 < hops:
                    frontier.append((neighbor, depth + 1))

    entity_views = []
    for nid in sorted(visited_nodes, key=str.lower):
        node = backend.get_node(nid)
        if node is not None:
            entity_views.append(node_to_view(node, backend))

    edge_views = tuple(edge_to_view(e) for e in collected_edges)
    prov = merge_provenance(*(ev.provenance for ev in edge_views)) if edge_views else Provenance()

    return NeighborhoodView(
        seed=seed,
        hops=hops,
        entities=tuple(entity_views),
        edges=edge_views,
        provenance=prov,
    )


def find_paths(
    backend: QueryBackend,
    source: str,
    target: str,
    *,
    max_hops: int = 3,
    direction: str = "both",
    relationship_type: str | None = None,
    include_inferred: bool = True,
    max_paths: int = 10,
) -> list[GraphPath]:
    """Find simple paths from ``source`` to ``target`` up to ``max_hops``."""
    if backend.get_node(source) is None:
        raise QueryError(f"Entity not found: {source!r}")
    if backend.get_node(target) is None:
        raise QueryError(f"Entity not found: {target!r}")
    if max_hops < 1:
        return []
    if source == target:
        return [
            GraphPath(
                nodes=(source,),
                edges=(),
                confidence=None,
            )
        ]

    results: list[GraphPath] = []

    # BFS over paths (node sequence, edge sequence)
    queue: deque[tuple[str, list[str], list[KGEdge]]] = deque()
    queue.append((source, [source], []))

    while queue and len(results) < max_paths:
        current, node_path, edge_path = queue.popleft()
        if len(edge_path) >= max_hops:
            continue

        for neighbor, edge in _expand_frontier(
            backend,
            current,
            direction=direction,
            relationship_type=relationship_type,
            include_inferred=include_inferred,
        ):
            if neighbor in node_path:
                continue

            new_nodes = node_path + [neighbor]
            new_edges = edge_path + [edge]

            if neighbor == target:
                edge_views = tuple(edge_to_view(e) for e in new_edges)
                results.append(
                    GraphPath(
                        nodes=tuple(new_nodes),
                        edges=edge_views,
                        confidence=_path_confidence(new_edges),
                    )
                )
                if len(results) >= max_paths:
                    break
            else:
                queue.append((neighbor, new_nodes, new_edges))

    results.sort(
        key=lambda p: (
            p.hop_count,
            -(p.confidence or 0.0),
            tuple(n.lower() for n in p.nodes),
        )
    )
    return results


def shortest_path(
    backend: QueryBackend,
    source: str,
    target: str,
    **kwargs,
) -> GraphPath | None:
    """Return the shortest path between two entities, or ``None``."""
    paths = find_paths(backend, source, target, max_paths=1, **kwargs)
    return paths[0] if paths else None
