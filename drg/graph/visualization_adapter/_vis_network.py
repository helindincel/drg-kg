"""vis-network.js exporter for ``EnhancedKG``.

vis-network uses ``{nodes, edges}`` with ``from``/``to`` instead of
``source``/``target``. The rest of the data flow mirrors the Cytoscape
exporter: skip isolated nodes, attach communities, attach edge weight as
``value``.
"""

from __future__ import annotations

from typing import Any

from ..kg_core import EnhancedKG
from ._palette import get_edge_color, get_node_color

__all__ = ["kg_to_vis_network"]


def kg_to_vis_network(kg: EnhancedKG) -> dict[str, Any]:
    """Convert ``EnhancedKG`` to ``{"nodes": [...], "edges": [...]}``."""
    if kg is None:
        raise ValueError("No knowledge graph provided")

    connected_node_ids: set[str] = set()
    for edge in kg.edges:
        connected_node_ids.add(edge.source)
        connected_node_ids.add(edge.target)

    nodes = []
    for node in kg.nodes.values():
        if node.id not in connected_node_ids:
            continue
        color = get_node_color(node.type)

        community_id = None
        for cluster_id, cluster in kg.clusters.items():
            if node.id in cluster.node_ids:
                community_id = cluster_id
                break

        node_data: dict[str, Any] = {
            "id": node.id,
            "label": node.id,
            "title": f"Type: {node.type or 'Unknown'}\nID: {node.id}",
            "color": color,
            "type": node.type or "Unknown",
            "properties": node.properties,
            "metadata": node.metadata,
        }
        if community_id:
            node_data["group"] = community_id
            node_data["community_id"] = community_id

        nodes.append(node_data)

    edges = []
    for edge in kg.edges:
        weight = edge.metadata.get("weight", 1.0)
        if "confidence" in edge.metadata:
            weight = edge.metadata["confidence"]
        color = get_edge_color(edge.relationship_type)

        edges.append(
            {
                "id": f"{edge.source}-{edge.target}",
                "from": edge.source,
                "to": edge.target,
                "label": edge.relationship_type,
                "title": f"{edge.relationship_type}\n{edge.relationship_detail}",
                "value": float(weight),
                "color": {"color": color},
                "relationship_type": edge.relationship_type,
                "relationship_detail": edge.relationship_detail,
                "metadata": edge.metadata,
            }
        )

    return {"nodes": nodes, "edges": edges}
