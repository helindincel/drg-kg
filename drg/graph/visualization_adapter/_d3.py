"""D3.js force-directed graph exporter.

D3 differs from Cytoscape/vis-network in two ways:

1. Edges reference nodes by **integer index** into the ``nodes`` array, not
   by id. We build a ``node_id -> index`` map up front and use it to wire
   links.
2. Isolated nodes are skipped (same as the other exporters), but the index
   is computed over the filtered set so edge indices stay consistent.
"""

from __future__ import annotations

from typing import Any

from ..kg_core import EnhancedKG
from ._palette import get_node_color

__all__ = ["kg_to_d3_json"]


def kg_to_d3_json(kg: EnhancedKG) -> dict[str, Any]:
    """Convert ``EnhancedKG`` to D3's ``{"nodes": [...], "links": [...]}`` shape."""
    if kg is None:
        raise ValueError("No knowledge graph provided")

    connected_node_ids: set[str] = set()
    for edge in kg.edges:
        connected_node_ids.add(edge.source)
        connected_node_ids.add(edge.target)

    connected_nodes_list = [
        (node_id, node) for node_id, node in kg.nodes.items() if node_id in connected_node_ids
    ]
    node_index = {node_id: idx for idx, (node_id, _) in enumerate(connected_nodes_list)}

    nodes = []
    for _, node in connected_nodes_list:
        color = get_node_color(node.type)

        community_id = None
        for cluster_id, cluster in kg.clusters.items():
            if node.id in cluster.node_ids:
                community_id = cluster_id
                break

        node_data: dict[str, Any] = {
            "id": node.id,
            "name": node.id,
            "type": node.type or "Unknown",
            "color": color,
            "group": community_id or 0,
            "properties": node.properties,
            "metadata": node.metadata,
        }
        if community_id:
            node_data["community_id"] = community_id
        nodes.append(node_data)

    links = []
    for edge in kg.edges:
        if edge.source not in node_index or edge.target not in node_index:
            continue

        weight = edge.metadata.get("weight", 1.0)
        if "confidence" in edge.metadata:
            weight = edge.metadata["confidence"]

        links.append(
            {
                "source": node_index[edge.source],
                "target": node_index[edge.target],
                "value": float(weight),
                "type": edge.relationship_type,
                "relationship_detail": edge.relationship_detail,
                "metadata": edge.metadata,
            }
        )

    return {"nodes": nodes, "links": links}
