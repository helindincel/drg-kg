"""Hub mitigation utilities for knowledge graph exports.

This module provides deterministic, explainable post-processing steps that can
reduce hub-dominant "star" layouts in visualization **without changing the
underlying extracted facts**.

Strategy implemented here:
- **Hub relation proxy split**: for any high-degree "hub" node, create a proxy
  node per relationship type (hub x rel). Re-route incident edges through that
  proxy node. This keeps the original relationship semantics on the proxy edge,
  and adds a lightweight structural connector hub -> proxy.

The result is a KG that is easier to visualize (less single-center), while still
preserving reachability and relation labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .kg_core import EnhancedKG, KGEdge, KGNode


def apply_hub_relation_proxy_split(
    kg: EnhancedKG,
    *,
    hub_degree_threshold: int = 10,
    enabled: bool = True,
    proxy_node_type: str = "HubProxy",
    proxy_id_prefix: str = "hubproxy::",
) -> dict[str, int]:
    """Split hub nodes into relation-specific proxy nodes (in-place).

    Args:
        kg: Target EnhancedKG to mutate.
        hub_degree_threshold: Degree threshold at/above which a node is treated
            as a hub. Degree is computed on the undirected edge incidence count.
        enabled: If False, this is a no-op.
        proxy_node_type: Node type string assigned to proxy nodes.
        proxy_id_prefix: Prefix used for proxy node ids.

    Returns:
        Stats dictionary with counts:
        - hubs: number of hub nodes detected
        - proxy_nodes: number of proxy nodes created
        - edges_replaced: number of original edges replaced by proxy edges
        - connector_edges: number of structural hub->proxy edges added
    """

    if not enabled:
        return {"hubs": 0, "proxy_nodes": 0, "edges_replaced": 0, "connector_edges": 0}

    if hub_degree_threshold < 3:
        raise ValueError("hub_degree_threshold must be >= 3")

    # Degree + incident edges
    deg: Counter[str] = Counter()
    incident: dict[str, list[KGEdge]] = defaultdict(list)
    for e in kg.edges:
        deg[e.source] += 1
        deg[e.target] += 1
        incident[e.source].append(e)
        incident[e.target].append(e)

    hubs: set[str] = {n for n, d in deg.items() if d >= hub_degree_threshold}
    if not hubs:
        return {"hubs": 0, "proxy_nodes": 0, "edges_replaced": 0, "connector_edges": 0}

    # Build proxy ids (hub, rel) -> proxy_node_id
    proxy_nodes_by_hub_rel: dict[tuple[str, str], str] = {}
    for hub in sorted(hubs, key=lambda s: s.lower()):
        rel_groups: dict[str, list[KGEdge]] = defaultdict(list)
        for e in incident.get(hub, []):
            rel_groups[e.relationship_type].append(e)
        for rel in sorted(rel_groups.keys(), key=lambda s: s.lower()):
            proxy_id = f"{proxy_id_prefix}{hub}::{rel}"
            proxy_nodes_by_hub_rel[(hub, rel)] = proxy_id

            if proxy_id not in kg.nodes:
                kg.add_node(
                    KGNode(
                        id=proxy_id,
                        type=proxy_node_type,
                        properties={},
                        metadata={
                            "hub": hub,
                            "relationship_type": rel,
                            "proxy_kind": "hub_relation_proxy",
                            "edge_count": len(rel_groups[rel]),
                        },
                    )
                )

    # Helper for stable edge identity (avoid duplicates while transforming).
    def _edge_key(edge: KGEdge) -> tuple[str, str, str, str]:
        return (edge.source, edge.target, edge.relationship_type, edge.relationship_detail)

    original_edges = list(kg.edges)
    new_edges: list[KGEdge] = []
    seen_new: set[tuple[str, str, str, str]] = set()

    # Add structural connector edges: hub -> proxy for each rel group.
    connector_edges = 0
    for (hub, rel), proxy_id in proxy_nodes_by_hub_rel.items():
        connector = KGEdge(
            source=hub,
            target=proxy_id,
            relationship_type=rel,
            relationship_detail="hub split connector",
            metadata={"proxy_kind": "hub_proxy_connector", "hub": hub, "relationship_type": rel},
        )
        k = _edge_key(connector)
        if k not in seen_new:
            new_edges.append(connector)
            seen_new.add(k)
            connector_edges += 1

    # Replace hub-incident edges with proxy edges.
    edges_replaced = 0
    for e in original_edges:
        # If edge touches a hub, re-route through the hub's rel proxy.
        if e.source in hubs or e.target in hubs:
            hub = e.source if e.source in hubs else e.target
            proxy_id = proxy_nodes_by_hub_rel.get((hub, e.relationship_type))
            if not proxy_id:
                # Should not happen, but keep original edge.
                k = _edge_key(e)
                if k not in seen_new:
                    new_edges.append(e)
                    seen_new.add(k)
                continue

            if e.source == hub:
                new_source, new_target = proxy_id, e.target
            else:
                new_source, new_target = e.source, proxy_id

            # Preserve the semantic edge (type/detail/metadata) on the proxy edge.
            proxy_edge = KGEdge(
                source=new_source,
                target=new_target,
                relationship_type=e.relationship_type,
                relationship_detail=e.relationship_detail,
                metadata={**(e.metadata or {}), "proxy_kind": "hub_split_edge", "hub": hub},
                start_time=e.start_time,
                end_time=e.end_time,
                confidence=e.confidence,
                is_negated=e.is_negated,
            )
            k = _edge_key(proxy_edge)
            if k not in seen_new:
                new_edges.append(proxy_edge)
                seen_new.add(k)
                edges_replaced += 1
            continue

        # Non-hub edge: keep as-is.
        k = _edge_key(e)
        if k not in seen_new:
            new_edges.append(e)
            seen_new.add(k)

    # Swap edges in-place.
    kg.edges = new_edges

    # Ensure all edge endpoints exist as nodes (proxy creation already handles proxy ids).
    for e in kg.edges:
        if e.source not in kg.nodes:
            kg.add_node(KGNode(id=e.source, type=None))
        if e.target not in kg.nodes:
            kg.add_node(KGNode(id=e.target, type=None))

    # Best-effort: clusters remain valid for original nodes; proxies are unassigned.
    # (UI communities view still works; proxies will appear as uncategorized.)

    # Count how many proxy nodes actually exist in the KG after insertion.
    proxy_nodes = sum(
        1
        for nid, n in kg.nodes.items()
        if nid.startswith(proxy_id_prefix) and n.type == proxy_node_type
    )

    return {
        "hubs": len(hubs),
        "proxy_nodes": proxy_nodes,
        "edges_replaced": edges_replaced,
        "connector_edges": connector_edges,
    }
