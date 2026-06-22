"""Regression tests for the refactored drg.graph.visualization_adapter package.

Verifies:
- Public API (legacy import surface) is preserved.
- Each exporter (Cytoscape, vis-network, D3) emits the expected element shapes.
- Hub-proxy flatten/split behavior is correct.
- Palette helpers return sensible defaults.
- Provenance exports round-trip through both formats.
"""

from __future__ import annotations

import pytest

from drg.graph.kg_core import Cluster, EnhancedKG, KGEdge, KGNode
from drg.graph.visualization_adapter import (
    COMMUNITY_COLORS,
    EDGE_COLORS,
    NODE_COLORS,
    PROVENANCE_COLORS,
    ProvenanceEdge,
    ProvenanceGraph,
    ProvenanceNode,
    VisualizationAdapter,
    communities_to_cytoscape,
    get_community_color,
    get_edge_color,
    get_node_color,
    get_provenance_color,
    kg_to_cytoscape,
    kg_to_d3_json,
    kg_to_vis_network,
    provenance_to_cytoscape,
    provenance_to_json,
)
from drg.graph.visualization_adapter._hubproxy import (
    flatten_hubproxy_view,
    is_hubproxy_id,
    resolve_hub_split_flags,
)


def _build_simple_kg() -> EnhancedKG:
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Alice", type="Person"))
    kg.add_node(KGNode(id="Acme", type="Company"))
    kg.add_node(KGNode(id="Bob", type="Person"))
    kg.add_node(KGNode(id="Isolated", type="Person"))  # no edges
    kg.add_edge(
        KGEdge(
            source="Alice",
            target="Acme",
            relationship_type="works_with",
            relationship_detail="Alice works at Acme",
        )
    )
    kg.add_edge(
        KGEdge(
            source="Bob",
            target="Acme",
            relationship_type="works_with",
            relationship_detail="Bob works at Acme",
        )
    )
    return kg


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_public_api_preserved():
    assert callable(kg_to_cytoscape)
    assert callable(kg_to_vis_network)
    assert callable(kg_to_d3_json)
    assert callable(communities_to_cytoscape)
    assert callable(provenance_to_cytoscape)
    assert callable(provenance_to_json)
    assert ProvenanceNode and ProvenanceEdge and ProvenanceGraph
    assert VisualizationAdapter


# ---------------------------------------------------------------------------
# Cytoscape exporter
# ---------------------------------------------------------------------------


def test_cytoscape_skips_isolated_nodes():
    kg = _build_simple_kg()
    elements = kg_to_cytoscape(kg)
    node_ids = {e["data"]["id"] for e in elements if "source" not in e["data"]}
    assert "Isolated" not in node_ids
    assert {"Alice", "Bob", "Acme"} <= node_ids


def test_cytoscape_emits_edges():
    kg = _build_simple_kg()
    elements = kg_to_cytoscape(kg)
    edges = [e for e in elements if "source" in e["data"]]
    assert len(edges) == 2
    for edge in edges:
        assert "label" in edge["data"]
        assert "weight" in edge["data"]


def test_cytoscape_edge_data_includes_optional_details():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Alice", type="Person"))
    kg.add_node(KGNode(id="Acme", type="Company"))
    kg.add_edge(
        KGEdge(
            source="Alice",
            target="Acme",
            relationship_type="works_at",
            relationship_detail="Alice works at Acme",
            start_time="2024-01-01",
            end_time="2025-01-01",
            created_at="2024-01-02T00:00:00+00:00",
            updated_at="2024-02-02T00:00:00+00:00",
            confidence=0.92,
        )
    )

    edge = next(e for e in kg_to_cytoscape(kg) if "source" in e["data"])
    assert edge["data"]["confidence"] == 0.92
    assert edge["data"]["start_time"] == "2024-01-01"
    assert edge["data"]["valid_from"] == "2024-01-01"
    assert edge["data"]["end_time"] == "2025-01-01"
    assert edge["data"]["valid_to"] == "2025-01-01"
    assert edge["data"]["created_at"] == "2024-01-02T00:00:00+00:00"
    assert edge["data"]["updated_at"] == "2024-02-02T00:00:00+00:00"


def test_cytoscape_raises_on_none_kg():
    with pytest.raises(ValueError):
        kg_to_cytoscape(None)


def test_communities_to_cytoscape_overlays_color():
    kg = _build_simple_kg()
    cluster = Cluster(id="c0", node_ids={"Alice", "Acme"})
    kg.clusters[cluster.id] = cluster

    elements = communities_to_cytoscape(kg)
    alice = next(e for e in elements if e["data"].get("id") == "Alice")
    assert alice["style"]["background-color"] == get_community_color(0)


# ---------------------------------------------------------------------------
# vis-network exporter
# ---------------------------------------------------------------------------


def test_vis_network_shape():
    kg = _build_simple_kg()
    result = kg_to_vis_network(kg)
    assert set(result.keys()) == {"nodes", "edges"}
    node_ids = {n["id"] for n in result["nodes"]}
    assert "Isolated" not in node_ids
    assert all("from" in e and "to" in e for e in result["edges"])


# ---------------------------------------------------------------------------
# D3 exporter
# ---------------------------------------------------------------------------


def test_d3_links_reference_node_indices():
    kg = _build_simple_kg()
    result = kg_to_d3_json(kg)
    assert {"nodes", "links"} == set(result.keys())
    node_count = len(result["nodes"])
    for link in result["links"]:
        assert 0 <= link["source"] < node_count
        assert 0 <= link["target"] < node_count


# ---------------------------------------------------------------------------
# Hub-proxy utilities
# ---------------------------------------------------------------------------


def test_is_hubproxy_id_detects_prefix():
    assert is_hubproxy_id("hubproxy::Acme::works_with")
    assert not is_hubproxy_id("Alice")
    assert not is_hubproxy_id(42)  # non-string defensive path


def test_resolve_hub_split_flags_defaults(monkeypatch):
    monkeypatch.delenv("DRG_UI_HUB_SPLIT", raising=False)
    monkeypatch.delenv("DRG_UI_HUB_SPLIT_THRESHOLD", raising=False)
    enabled, threshold = resolve_hub_split_flags(None, None)
    assert enabled is False
    assert threshold == 10


def test_resolve_hub_split_flags_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("DRG_UI_HUB_SPLIT", "1")
    enabled, threshold = resolve_hub_split_flags(False, 5)
    assert enabled is False
    assert threshold == 5


def test_flatten_hubproxy_view_rebuilds_edges_from_triples():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Acme", type="Company"))
    kg.add_node(KGNode(id="Alice", type="Person"))
    kg.add_node(KGNode(id="hubproxy::Acme::works_with", type="HubProxy"))
    kg.add_edge(
        KGEdge(
            source="hubproxy::Acme::works_with",
            target="Alice",
            relationship_type="works_with",
            relationship_detail="Alice works at Acme",
            metadata={
                "proxy_kind": "hub_split_edge",
                "triple": ("Acme", "works_with", "Alice"),
            },
        )
    )

    nodes, edges = flatten_hubproxy_view(kg)
    assert all(not is_hubproxy_id(n) for n in nodes)
    assert len(edges) == 1
    assert edges[0].source == "Acme"
    assert edges[0].target == "Alice"
    assert edges[0].metadata["flattened_from_proxy"] is True


def test_cytoscape_hub_split_enabled_emits_proxy_nodes():
    """With hub_split enabled and a low threshold, central nodes spawn proxies."""
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Hub", type="Company"))
    for i in range(5):
        target = f"E{i}"
        kg.add_node(KGNode(id=target, type="Product"))
        kg.add_edge(
            KGEdge(
                source="Hub",
                target=target,
                relationship_type="produces",
                relationship_detail=f"Hub produces {target}",
            )
        )

    elements = kg_to_cytoscape(kg, hub_split=True, hub_split_threshold=3)
    proxy_ids = [e["data"]["id"] for e in elements if is_hubproxy_id(e["data"].get("id", ""))]
    assert proxy_ids, "Expected at least one hub proxy node"


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------


def test_palette_defaults():
    assert get_node_color(None) == NODE_COLORS["default"]
    assert get_edge_color("nonexistent") == EDGE_COLORS["default"]
    assert get_community_color(len(COMMUNITY_COLORS)) == COMMUNITY_COLORS[0]
    assert get_provenance_color("query") == PROVENANCE_COLORS["query"]
    assert get_provenance_color("???") == "#A8A8A8"


# ---------------------------------------------------------------------------
# Provenance exporters
# ---------------------------------------------------------------------------


def _build_provenance() -> ProvenanceGraph:
    return ProvenanceGraph(
        nodes=[
            ProvenanceNode(id="q", type="query", label="what?"),
            ProvenanceNode(id="a", type="answer", label="42"),
        ],
        edges=[ProvenanceEdge(source="q", target="a", type="generated_from", label="LLM")],
        query="what?",
        answer="42",
    )


def test_provenance_to_cytoscape_emits_nodes_and_edge():
    prov = _build_provenance()
    elements = provenance_to_cytoscape(prov)
    assert len(elements) == 3
    sources = [e["data"].get("source") for e in elements]
    assert "q" in sources


def test_provenance_to_json_round_trip():
    prov = _build_provenance()
    out = provenance_to_json(prov)
    assert out["query"] == "what?"
    assert out["answer"] == "42"
    assert len(out["nodes"]) == 2
    assert len(out["edges"]) == 1


def test_provenance_graph_to_dict_matches_export():
    prov = _build_provenance()
    assert provenance_to_json(prov) == prov.to_dict()


# ---------------------------------------------------------------------------
# Adapter facade
# ---------------------------------------------------------------------------


def test_adapter_methods_delegate_to_module_functions():
    kg = _build_simple_kg()
    adapter = VisualizationAdapter(kg)

    assert adapter.kg_to_cytoscape() == kg_to_cytoscape(kg)
    assert adapter.kg_to_vis_network() == kg_to_vis_network(kg)
    assert adapter.kg_to_d3_json() == kg_to_d3_json(kg)
    assert adapter._is_hubproxy_id("hubproxy::a::b") is True
    assert adapter._get_node_color("Person") == NODE_COLORS["Person"]
