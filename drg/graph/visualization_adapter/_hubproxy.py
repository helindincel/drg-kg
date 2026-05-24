"""Hub-proxy split utilities for Cytoscape visualization.

When a knowledge graph is dominated by a single hub node (e.g. a parent
company linked to dozens of products), Cytoscape layouts collapse into a
star with all nodes orbiting the hub. The hub-split visual mode mitigates
this by inserting a synthetic proxy node per ``(hub, relationship_type)``
pair, so the layout engine treats each relation type as its own cluster.

This module owns:

- :func:`is_hubproxy_id` — predicate that recognises proxy ids in stored KGs.
- :func:`flatten_hubproxy_view` — strip proxies from a KG that already has
  them baked in, so the UI can show a clean "original" view.
- :func:`build_hub_split` — compute hub set, proxy nodes, and the bookkeeping
  the Cytoscape exporter needs.

All functions are pure on their inputs (no global state).
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict

from ..kg_core import EnhancedKG, KGEdge, KGNode

__all__ = [
    "build_hub_split",
    "flatten_hubproxy_view",
    "is_hubproxy_id",
    "resolve_hub_split_flags",
]


def is_hubproxy_id(node_id: str) -> bool:
    """True if ``node_id`` looks like a hub-proxy node id (``hubproxy::...``)."""
    return isinstance(node_id, str) and node_id.startswith("hubproxy::")


def resolve_hub_split_flags(
    hub_split: bool | None,
    hub_split_threshold: int | None,
) -> tuple[bool, int]:
    """Resolve the ``(hub_split, threshold)`` pair from explicit args and env.

    Defaults read from ``DRG_UI_HUB_SPLIT`` and ``DRG_UI_HUB_SPLIT_THRESHOLD``,
    matching the previous module-level behavior so the UI works the same.
    """
    if hub_split is None:
        hub_split = os.getenv("DRG_UI_HUB_SPLIT", "0").strip().lower() in {"1", "true", "yes", "y"}
    if hub_split_threshold is None:
        try:
            hub_threshold = int(os.getenv("DRG_UI_HUB_SPLIT_THRESHOLD", "10"))
        except Exception:
            hub_threshold = 10
    else:
        hub_threshold = int(hub_split_threshold)
    return hub_split, hub_threshold


def flatten_hubproxy_view(kg: EnhancedKG) -> tuple[dict[str, KGNode], list[KGEdge]]:
    """Return ``(nodes, edges)`` with hub-proxy nodes/edges removed.

    Used when the UI's ``hub_split`` toggle is OFF but the stored KG already
    contains baked-in proxy nodes. We rebuild original edges from
    ``edge.metadata["triple"]`` where possible, and drop purely structural
    proxy connectors so no ``hubproxy::...`` ids leak into the UI.
    """
    edges_out: list[KGEdge] = []
    seen: set[tuple[str, str, str, str]] = set()

    for edge in kg.edges:
        md = edge.metadata or {}
        proxy_kind = md.get("proxy_kind")

        if proxy_kind == "hub_proxy_connector":
            continue  # structural connector — drop entirely

        touches_proxy = is_hubproxy_id(edge.source) or is_hubproxy_id(edge.target)
        looks_proxy_edge = touches_proxy or proxy_kind == "hub_split_edge"

        if looks_proxy_edge:
            triple = md.get("triple")
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                src, rel, dst = triple
                if isinstance(src, str) and isinstance(rel, str) and isinstance(dst, str):
                    new_md = dict(md)
                    new_md.pop("proxy_kind", None)
                    new_md.pop("hub", None)
                    new_md["flattened_from_proxy"] = True
                    rebuilt = KGEdge(
                        source=src,
                        target=dst,
                        relationship_type=rel,
                        relationship_detail=edge.relationship_detail,
                        metadata=new_md,
                        start_time=edge.start_time,
                        end_time=edge.end_time,
                        confidence=edge.confidence,
                        is_negated=edge.is_negated,
                    )
                    key = (
                        rebuilt.source,
                        rebuilt.target,
                        rebuilt.relationship_type,
                        rebuilt.relationship_detail,
                    )
                    if key not in seen:
                        edges_out.append(rebuilt)
                        seen.add(key)
                    continue

            # Proxy-related but unrecoverable — drop to avoid leaking proxy ids.
            if touches_proxy:
                continue

        key = (edge.source, edge.target, edge.relationship_type, edge.relationship_detail)
        if key not in seen:
            edges_out.append(edge)
            seen.add(key)

    connected: set[str] = set()
    for e in edges_out:
        connected.add(e.source)
        connected.add(e.target)

    nodes_out: dict[str, KGNode] = {}
    for nid in connected:
        node = kg.nodes.get(nid)
        if node is None:
            node = KGNode(id=nid, type=None)
        nodes_out[nid] = node

    return nodes_out, edges_out


def build_hub_split(
    edges_view: list[KGEdge],
    hub_threshold: int,
    enable: bool,
) -> tuple[set, dict[str, list[KGEdge]], Counter]:
    """Compute degree counts, incidence map, and the set of hubs to split.

    Returns:
        ``(hubs, incident, degrees)`` — the incident edge map is useful for
        building proxy nodes per relation type without re-iterating the edges.
    """
    deg: Counter = Counter()
    incident: dict[str, list[KGEdge]] = defaultdict(list)
    for e in edges_view:
        deg[e.source] += 1
        deg[e.target] += 1
        incident[e.source].append(e)
        incident[e.target].append(e)

    hubs = {n for n, d in deg.items() if d >= hub_threshold} if enable else set()
    return hubs, incident, deg
