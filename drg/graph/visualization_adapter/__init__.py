"""Visualization adapter package.

Renders ``EnhancedKG`` instances and answer-provenance chains into the JSON
shapes expected by JavaScript graph libraries (Cytoscape.js, vis-network,
D3.js). Backwards-compatible: the same imports that worked on the legacy
single-file module continue to work here.

Public surface
==============

- :class:`VisualizationAdapter` — stateful facade kept for backward
  compatibility with the API server.
- :class:`ProvenanceNode`, :class:`ProvenanceEdge`, :class:`ProvenanceGraph` —
  the provenance data model.
- Module-level exporters callers can use without instantiating the adapter:

  - :func:`kg_to_cytoscape`
  - :func:`kg_to_vis_network`
  - :func:`kg_to_d3_json`
  - :func:`communities_to_cytoscape`
  - :func:`provenance_to_cytoscape`
  - :func:`provenance_to_json`

Architecture
============

::

    drg/graph/visualization_adapter/
        __init__.py             # public re-exports
        _provenance.py          # ProvenanceNode / Edge / Graph
        _palette.py             # color helpers + palettes
        _hubproxy.py            # hub-proxy split utilities
        _cytoscape.py           # Cytoscape exporter (+ communities overlay)
        _vis_network.py         # vis-network exporter
        _d3.py                  # D3 force-directed exporter
        _provenance_export.py   # provenance → Cytoscape / JSON
        _adapter.py             # VisualizationAdapter facade
"""

from __future__ import annotations

from ._adapter import VisualizationAdapter
from ._cytoscape import communities_to_cytoscape, kg_to_cytoscape
from ._d3 import kg_to_d3_json
from ._palette import (
    COMMUNITY_COLORS,
    EDGE_COLORS,
    NODE_COLORS,
    PROVENANCE_COLORS,
    get_community_color,
    get_edge_color,
    get_node_color,
    get_provenance_color,
)
from ._provenance import ProvenanceEdge, ProvenanceGraph, ProvenanceNode
from ._provenance_export import provenance_to_cytoscape, provenance_to_json
from ._vis_network import kg_to_vis_network

__all__ = [
    "COMMUNITY_COLORS",
    "EDGE_COLORS",
    "NODE_COLORS",
    "PROVENANCE_COLORS",
    "ProvenanceEdge",
    "ProvenanceGraph",
    "ProvenanceNode",
    "VisualizationAdapter",
    "communities_to_cytoscape",
    "get_community_color",
    "get_edge_color",
    "get_node_color",
    "get_provenance_color",
    "kg_to_cytoscape",
    "kg_to_d3_json",
    "kg_to_vis_network",
    "provenance_to_cytoscape",
    "provenance_to_json",
]
