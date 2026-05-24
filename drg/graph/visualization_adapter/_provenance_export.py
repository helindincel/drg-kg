"""Provenance graph exporters: Cytoscape and generic JSON.

Keeps the (small) provenance-specific rendering out of the main Cytoscape
module so the KG exporter stays focused.
"""

from __future__ import annotations

from typing import Any

from ._palette import get_provenance_color
from ._provenance import ProvenanceGraph

__all__ = ["provenance_to_cytoscape", "provenance_to_json"]


def provenance_to_cytoscape(provenance: ProvenanceGraph) -> list[dict[str, Any]]:
    """Render a :class:`ProvenanceGraph` as a Cytoscape element list."""
    elements: list[dict[str, Any]] = []

    for node in provenance.nodes:
        color = get_provenance_color(node.type)
        elements.append(
            {
                "data": {
                    "id": node.id,
                    "label": node.label,
                    "type": node.type,
                    "data": node.data,
                    "metadata": node.metadata,
                },
                "classes": [node.type],
                "style": {
                    "background-color": color,
                    "label": node.label,
                },
            }
        )

    for edge in provenance.edges:
        elements.append(
            {
                "data": {
                    "id": f"{edge.source}-{edge.target}",
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "type": edge.type,
                    "weight": edge.weight,
                    "metadata": edge.metadata,
                },
                "style": {
                    "width": max(1, min(5, edge.weight * 3)),
                    "label": edge.label,
                },
            }
        )

    return elements


def provenance_to_json(provenance: ProvenanceGraph) -> dict[str, Any]:
    """Generic JSON dump of a :class:`ProvenanceGraph` (matches ``to_dict``)."""
    return {
        "query": provenance.query,
        "answer": provenance.answer,
        "nodes": [
            {
                "id": node.id,
                "type": node.type,
                "label": node.label,
                "data": node.data,
                "metadata": node.metadata,
            }
            for node in provenance.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "label": edge.label,
                "weight": edge.weight,
                "metadata": edge.metadata,
            }
            for edge in provenance.edges
        ],
        "metadata": provenance.metadata,
    }
