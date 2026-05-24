"""``VisualizationAdapter`` — thin facade over per-format exporter functions.

The class exists for back-compat (the API server constructs an adapter and
calls instance methods); the heavy lifting lives in the per-format modules.
Most call sites can now use the module-level functions directly.
"""

from __future__ import annotations

from typing import Any

from ..community_report import CommunityReport
from ..kg_core import EnhancedKG, KGEdge, KGNode
from . import _hubproxy
from ._cytoscape import communities_to_cytoscape, kg_to_cytoscape
from ._d3 import kg_to_d3_json
from ._palette import get_community_color, get_edge_color, get_node_color
from ._provenance import ProvenanceGraph
from ._provenance_export import provenance_to_cytoscape, provenance_to_json
from ._vis_network import kg_to_vis_network

__all__ = ["VisualizationAdapter"]


class VisualizationAdapter:
    """Adapter for converting KGs and provenance chains into JS graph formats.

    Supports Cytoscape.js, vis-network, and D3. Holds an optional default KG
    so callers can construct once and emit multiple formats without
    threading the graph through every call.
    """

    def __init__(self, kg: EnhancedKG | None = None):
        self.kg = kg

    # --- KG exports -------------------------------------------------------

    def kg_to_cytoscape(
        self,
        kg: EnhancedKG | None = None,
        *,
        hub_split: bool | None = None,
        hub_split_threshold: int | None = None,
    ) -> list[dict[str, Any]]:
        return kg_to_cytoscape(
            kg or self.kg,
            hub_split=hub_split,
            hub_split_threshold=hub_split_threshold,
        )

    def kg_to_vis_network(self, kg: EnhancedKG | None = None) -> dict[str, Any]:
        return kg_to_vis_network(kg or self.kg)

    def kg_to_d3_json(self, kg: EnhancedKG | None = None) -> dict[str, Any]:
        return kg_to_d3_json(kg or self.kg)

    def communities_to_cytoscape(
        self,
        kg: EnhancedKG | None = None,
        community_reports: list[CommunityReport] | None = None,
        *,
        hub_split: bool | None = None,
        hub_split_threshold: int | None = None,
    ) -> list[dict[str, Any]]:
        return communities_to_cytoscape(
            kg or self.kg,
            community_reports=community_reports,
            hub_split=hub_split,
            hub_split_threshold=hub_split_threshold,
        )

    # --- Provenance exports ----------------------------------------------

    def provenance_to_cytoscape(self, provenance: ProvenanceGraph) -> list[dict[str, Any]]:
        return provenance_to_cytoscape(provenance)

    def provenance_to_json(self, provenance: ProvenanceGraph) -> dict[str, Any]:
        return provenance_to_json(provenance)

    # --- Back-compat: legacy private helpers some downstream code calls --

    @staticmethod
    def _is_hubproxy_id(node_id: str) -> bool:
        return _hubproxy.is_hubproxy_id(node_id)

    def _flatten_hubproxy_view(self, kg: EnhancedKG) -> tuple[dict[str, KGNode], list[KGEdge]]:
        return _hubproxy.flatten_hubproxy_view(kg)

    def _get_node_color(self, node_type: str | None) -> str:
        return get_node_color(node_type)

    def _get_edge_color(self, relationship_type: str) -> str:
        return get_edge_color(relationship_type)

    def _get_community_color(self, index: int) -> str:
        return get_community_color(index)
