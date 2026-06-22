"""Deterministic, dependency-light cluster generation for the UI.

This project treats clustering as an analysis step over extracted knowledge graphs.
However, the UI "Communities" view requires clusters to exist. Many extracted KGs do not
include clusters by default, so we provide a deterministic fallback that:

1) Tries connected components (undirected) using only Python data structures.
2) If there is only one meaningful component, falls back to type-based grouping
   (e.g., Person / Organization / Technology) to provide a useful community view.

No network calls, no LLM usage.
"""

from __future__ import annotations

from .kg_core import Cluster, EnhancedKG


def _build_undirected_adjacency(kg: EnhancedKG) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {node_id: set() for node_id in kg.nodes.keys()}
    for e in kg.edges:
        if e.source not in adj:
            adj[e.source] = set()
        if e.target not in adj:
            adj[e.target] = set()
        adj[e.source].add(e.target)
        adj[e.target].add(e.source)
    return adj


def _connected_components(adj: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for start in adj.keys():
        if start in seen:
            continue
        stack = [start]
        comp: set[str] = set()
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in adj.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    # Stable ordering: biggest first, then lexicographic representative
    components.sort(key=lambda c: (-len(c), sorted(c, key=str.lower)[0].lower() if c else ""))
    return components


def _type_based_groups(
    kg: EnhancedKG, node_ids: set[str] | None = None
) -> list[tuple[str, set[str]]]:
    groups: dict[str, set[str]] = {}
    for nid, node in kg.nodes.items():
        if node_ids is not None and nid not in node_ids:
            continue
        t = (node.type or "Unknown").strip() or "Unknown"
        groups.setdefault(t, set()).add(nid)
    # Stable ordering: biggest first, then type name
    items = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0].lower()))
    return items


def ensure_clusters(
    kg: EnhancedKG,
    *,
    min_cluster_size: int = 2,
    max_clusters: int = 12,
) -> bool:
    """Ensure `kg.clusters` is non-empty by generating deterministic clusters if needed.

    Args:
        kg: EnhancedKG instance (mutated in-place by adding clusters).
        min_cluster_size: Ignore very small clusters (size < min_cluster_size).
        max_clusters: Safety cap for UI.

    Returns:
        True if clusters exist (either pre-existing or generated), else False.
    """
    if kg.clusters:
        return True

    if not kg.nodes:
        return False

    adj = _build_undirected_adjacency(kg)
    comps = [c for c in _connected_components(adj) if len(c) >= min_cluster_size]

    created = 0

    # Case A: Multiple components -> treat each as a community
    if len(comps) >= 2:
        for i, comp in enumerate(comps[:max_clusters], start=1):
            kg.add_cluster(
                Cluster(
                    id=f"cc_{i}",
                    node_ids=set(comp),
                    metadata={
                        "algorithm": "connected_components",
                        "node_count": len(comp),
                    },
                )
            )
            created += 1
        return created > 0

    # Case B: Single big component (or none) -> use type-based communities within it
    universe = comps[0] if comps else set(kg.nodes.keys())
    type_groups = _type_based_groups(kg, node_ids=universe)

    # Keep only groups that are at least min_cluster_size, but ensure we create at least one.
    kept = [(t, ids) for t, ids in type_groups if len(ids) >= min_cluster_size]
    if not kept and type_groups:
        kept = [type_groups[0]]

    for i, (t, ids) in enumerate(kept[:max_clusters], start=1):
        kg.add_cluster(
            Cluster(
                id=f"type_{i}_{t.lower().replace(' ', '_')}",
                node_ids=set(ids),
                metadata={
                    "algorithm": "type_grouping",
                    "type": t,
                    "node_count": len(ids),
                },
            )
        )
        created += 1

    return created > 0
