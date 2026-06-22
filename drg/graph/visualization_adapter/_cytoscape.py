"""Cytoscape.js export for EnhancedKG and community-overlaid views.

Two public functions:

- :func:`kg_to_cytoscape` — the primary exporter; emits a list of Cytoscape
  ``elements`` (nodes + edges + optional hub-proxy nodes).
- :func:`communities_to_cytoscape` — convenience wrapper that recolors nodes
  by their community/cluster id.

The exporter is intentionally a function rather than a method on the adapter
class so that callers without a stateful adapter can use it directly.
"""

from __future__ import annotations

from typing import Any

from ..community_report import CommunityReport
from ..kg_core import EnhancedKG, KGEdge
from . import _hubproxy
from ._palette import get_community_color, get_edge_color, get_node_color

__all__ = ["communities_to_cytoscape", "kg_to_cytoscape"]


def _edge_key(edge: KGEdge) -> tuple[str, str, str, str]:
    return (edge.source, edge.target, edge.relationship_type, edge.relationship_detail)


def _edge_detail_fields(edge: KGEdge) -> dict[str, Any]:
    """Optional edge fields consumed by the browser details panel."""

    out: dict[str, Any] = {}
    if edge.confidence is not None:
        out["confidence"] = edge.confidence
    if edge.start_time:
        out["start_time"] = edge.start_time
        out["valid_from"] = edge.start_time
    if edge.end_time:
        out["end_time"] = edge.end_time
        out["valid_to"] = edge.end_time
    if edge.created_at:
        out["created_at"] = edge.created_at
    if edge.updated_at:
        out["updated_at"] = edge.updated_at
    return out


def kg_to_cytoscape(
    kg: EnhancedKG,
    *,
    hub_split: bool | None = None,
    hub_split_threshold: int | None = None,
) -> list[dict[str, Any]]:
    """Convert ``EnhancedKG`` to a list of Cytoscape.js ``elements``.

    Args:
        kg: Source knowledge graph. **Must not be None.**
        hub_split: If True, split high-degree hubs into proxy nodes (one per
            relation type). Defaults to the ``DRG_UI_HUB_SPLIT`` env flag.
        hub_split_threshold: Minimum degree for a node to be considered a
            hub. Defaults to ``DRG_UI_HUB_SPLIT_THRESHOLD`` (10).

    Returns:
        Cytoscape elements (nodes first, then edges), ready to feed straight
        into ``cy.add(elements)`` on the front-end.
    """
    if kg is None:
        raise ValueError("No knowledge graph provided")

    hub_split, hub_threshold = _hubproxy.resolve_hub_split_flags(hub_split, hub_split_threshold)

    # When hub_split is OFF but the stored KG already contains baked-in proxy
    # nodes, flatten them so the UI never shows raw `hubproxy::...` ids.
    if not hub_split:
        nodes_view, edges_view = _hubproxy.flatten_hubproxy_view(kg)
    else:
        nodes_view, edges_view = kg.nodes, list(kg.edges)

    elements: list[dict[str, Any]] = []

    # Index of nodes that participate in at least one edge — isolated nodes
    # add clutter to Cytoscape layouts without conveying information.
    connected_node_ids: set[str] = set()
    for edge in edges_view:
        connected_node_ids.add(edge.source)
        connected_node_ids.add(edge.target)

    # Add connected nodes (with community lookup, degree, color, etc.)
    for node in nodes_view.values():
        if node.id not in connected_node_ids:
            continue
        color = get_node_color(node.type)

        community_id = None
        for cluster_id, cluster in kg.clusters.items():
            if node.id in cluster.node_ids:
                community_id = cluster_id
                break

        connection_count = sum(
            1 for edge in edges_view if edge.source == node.id or edge.target == node.id
        )
        node_weight = max(1, min(10, connection_count))

        node_data: dict[str, Any] = {
            "data": {
                "id": node.id,
                "label": node.id,
                "type": node.type or "Unknown",
                "properties": node.properties,
                "metadata": node.metadata,
                "weight": node_weight,
                "connection_count": connection_count,
            },
            "classes": [node.type or "entity"] if node.type else ["entity"],
            "style": {
                "background-color": color,
                "label": node.id,
            },
        }

        if community_id:
            node_data["data"]["community_id"] = community_id
            node_data["data"]["community"] = community_id

        elements.append(node_data)

    # Hub-split: compute hub set + incident edges. Used only when hub_split
    # is enabled; otherwise `hubs` is empty and the loop below behaves
    # identically to a vanilla edge export.
    hubs, incident, _ = _hubproxy.build_hub_split(edges_view, hub_threshold, hub_split)

    # Map node -> community for proxy node coloring/group inheritance.
    node_to_community: dict[str, str | None] = {}
    if kg.clusters:
        for node_id in connected_node_ids:
            cid: str | None = None
            for cluster_id, cluster in kg.clusters.items():
                if node_id in cluster.node_ids:
                    cid = cluster_id
                    break
            node_to_community[node_id] = cid

    # Emit proxy nodes (one per hub × relation_type) plus structural
    # connector edges (hub -> proxy). The Cytoscape layout will then place
    # each relation cluster around its own gravitational anchor.
    proxy_nodes_by_hub_rel: dict[tuple[str, str], str] = {}
    for hub in sorted(hubs, key=lambda s: s.lower()):
        from collections import defaultdict as _defaultdict

        rel_groups: dict[str, list[KGEdge]] = _defaultdict(list)
        for e in incident.get(hub, []):
            rel_groups[e.relationship_type].append(e)

        for rel in sorted(rel_groups.keys(), key=lambda s: s.lower()):
            proxy_id = f"hubproxy::{hub}::{rel}"
            proxy_nodes_by_hub_rel[(hub, rel)] = proxy_id

            hub_comm = node_to_community.get(hub)
            proxy_data: dict[str, Any] = {
                "data": {
                    "id": proxy_id,
                    "label": rel,
                    "type": "HubProxy",
                    "properties": {},
                    "metadata": {
                        "hub": hub,
                        "relationship_type": rel,
                        "proxy_kind": "hub_relation_proxy",
                        "edge_count": len(rel_groups[rel]),
                    },
                    "weight": len(rel_groups[rel]),
                    "connection_count": len(rel_groups[rel]),
                },
                "classes": ["HubProxy"],
                "style": {
                    "background-color": "#CBD5E1",
                    "label": rel,
                    "shape": "round-rectangle",
                    "text-wrap": "wrap",
                    "text-max-width": "120px",
                    "font-size": "9px",
                    "width": 18,
                    "height": 18,
                    "border-width": 1,
                    "border-color": "#94A3B8",
                },
            }
            if hub_comm:
                proxy_data["data"]["community_id"] = hub_comm
                proxy_data["data"]["community"] = hub_comm
            elements.append(proxy_data)

            elements.append(
                {
                    "data": {
                        "id": f"{hub}--{proxy_id}",
                        "source": hub,
                        "target": proxy_id,
                        "label": rel,
                        "relationship_type": rel,
                        "relationship_detail": "",
                        "relationship_description": "UI structural edge for hub splitting.",
                        "weight": 0.1,
                        "metadata": {"proxy_kind": "hub_proxy_connector"},
                    },
                    "style": {
                        "width": 1,
                        "line-color": "#CBD5E1",
                        "label": "",
                        "line-style": "dashed",
                    },
                }
            )

    # Emit edges. When hub-split is active, replace each hub-incident edge
    # with a proxy-routed edge so the layout sees relation-grouped clusters.
    replaced_edge_keys: set = set()
    for edge in edges_view:
        if hubs and (edge.source in hubs or edge.target in hubs):
            key = _edge_key(edge)
            if key in replaced_edge_keys:
                continue

            hub = edge.source if edge.source in hubs else edge.target
            proxy_id = proxy_nodes_by_hub_rel.get((hub, edge.relationship_type))
            if proxy_id:
                replaced_edge_keys.add(key)
                if edge.source == hub:
                    new_source, new_target = proxy_id, edge.target
                else:
                    new_source, new_target = edge.source, proxy_id

                weight = edge.metadata.get("weight", 1.0)
                if "confidence" in edge.metadata:
                    weight = edge.metadata["confidence"]
                color = get_edge_color(edge.relationship_type)

                elements.append(
                    {
                        "data": {
                            "id": f"{new_source}-{new_target}-{edge.relationship_type}",
                            "source": new_source,
                            "target": new_target,
                            "label": edge.relationship_type,
                            "relationship_type": edge.relationship_type,
                            "relationship_detail": edge.relationship_detail,
                            "relationship_description": (
                                edge.metadata.get("relationship_description")
                                or edge.metadata.get("description")
                                or ""
                            ),
                            "weight": float(weight),
                            "metadata": {
                                **edge.metadata,
                                "proxy_kind": "hub_split_edge",
                                "hub": hub,
                            },
                            **_edge_detail_fields(edge),
                        },
                        "style": {
                            "width": max(3, min(10, float(weight) * 5)),
                            "line-color": color,
                            "label": edge.relationship_type,
                        },
                    }
                )
                continue

        # Plain edge: no hub-splitting in effect for this edge.
        weight = edge.metadata.get("weight", 1.0)
        if "confidence" in edge.metadata:
            weight = edge.metadata["confidence"]
        color = get_edge_color(edge.relationship_type)

        elements.append(
            {
                "data": {
                    "id": f"{edge.source}-{edge.target}",
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.relationship_type,
                    "relationship_type": edge.relationship_type,
                    "relationship_detail": edge.relationship_detail,
                    "relationship_description": (
                        edge.metadata.get("relationship_description")
                        or edge.metadata.get("description")
                        or ""
                    ),
                    "weight": float(weight),
                    "metadata": edge.metadata,
                    **_edge_detail_fields(edge),
                },
                "style": {
                    "width": max(3, min(10, weight * 5)),
                    "line-color": color,
                    "label": edge.relationship_type,
                },
            }
        )

    return elements


def communities_to_cytoscape(
    kg: EnhancedKG,
    community_reports: list[CommunityReport] | None = None,
    *,
    hub_split: bool | None = None,
    hub_split_threshold: int | None = None,
) -> list[dict[str, Any]]:
    """Cytoscape elements with nodes re-colored by community.

    Equivalent to :func:`kg_to_cytoscape` followed by a community-color
    overlay; ``community_reports`` is accepted (and forwarded by the API
    layer) but not used directly here.
    """
    elements = kg_to_cytoscape(
        kg,
        hub_split=hub_split,
        hub_split_threshold=hub_split_threshold,
    )
    if kg is None:
        return elements

    community_colors: dict[str, str] = {}
    for idx, cluster_id in enumerate(kg.clusters.keys()):
        community_colors[cluster_id] = get_community_color(idx)

    for element in elements:
        data = element.get("data", {})
        community_id = data.get("community_id")
        if community_id and community_id in community_colors:
            element["style"]["background-color"] = community_colors[community_id]

    # community_reports is kept in the signature for forward-compat: the
    # web UI passes it through so future implementations can attach summary
    # tooltips, etc., without changing call sites.
    _ = community_reports
    return elements
