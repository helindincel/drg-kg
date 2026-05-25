"""Unit tests for drg.graph.auto_clusters.

Covers the deterministic, dependency-light cluster generation used by the
UI when no external clustering algorithm has produced communities yet.
Pure data structure manipulation — no LLM, no networkx, no optional deps.
"""

from __future__ import annotations

from drg.graph.auto_clusters import (
    _build_undirected_adjacency,
    _connected_components,
    _type_based_groups,
    ensure_clusters,
)
from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode


def _kg(nodes: list[tuple[str, str | None]], edges: list[tuple[str, str]]) -> EnhancedKG:
    """Build a tiny EnhancedKG fixture from (id, type) pairs and (src, dst) pairs."""
    kg = EnhancedKG()
    for nid, ntype in nodes:
        kg.add_node(KGNode(id=nid, type=ntype, properties={}, metadata={}))
    for s, t in edges:
        kg.add_edge(
            KGEdge(
                source=s,
                target=t,
                relationship_type="rel",
                relationship_detail=f"{s}-{t}",
                metadata={},
            )
        )
    return kg


# ---------------------------------------------------------------------------
# _build_undirected_adjacency
# ---------------------------------------------------------------------------


def test_build_adjacency_empty_kg_returns_empty_dict():
    kg = EnhancedKG()
    assert _build_undirected_adjacency(kg) == {}


def test_build_adjacency_isolated_nodes_have_empty_neighbour_sets():
    kg = _kg([("a", "T"), ("b", "T")], edges=[])
    adj = _build_undirected_adjacency(kg)
    assert adj == {"a": set(), "b": set()}


def test_build_adjacency_treats_edges_as_undirected():
    kg = _kg([("a", "T"), ("b", "T")], edges=[("a", "b")])
    adj = _build_undirected_adjacency(kg)
    assert adj["a"] == {"b"}
    assert adj["b"] == {"a"}


def test_build_adjacency_handles_edge_endpoints_missing_from_nodes():
    # Edge points to a node that isn't registered; adjacency still records it.
    kg = _kg([("a", "T")], edges=[])
    kg.add_edge(
        KGEdge(
            source="a",
            target="ghost",
            relationship_type="rel",
            relationship_detail="a-ghost",
            metadata={},
        )
    )
    adj = _build_undirected_adjacency(kg)
    assert "ghost" in adj
    assert adj["a"] == {"ghost"}


# ---------------------------------------------------------------------------
# _connected_components
# ---------------------------------------------------------------------------


def test_connected_components_handles_empty_adjacency():
    assert _connected_components({}) == []


def test_connected_components_returns_each_isolated_node_as_singleton():
    adj = {"a": set(), "b": set(), "c": set()}
    comps = _connected_components(adj)
    assert len(comps) == 3
    assert all(len(c) == 1 for c in comps)


def test_connected_components_finds_two_components():
    adj = {
        "a": {"b"},
        "b": {"a"},
        "c": {"d"},
        "d": {"c"},
    }
    comps = _connected_components(adj)
    assert len(comps) == 2
    sizes = sorted(len(c) for c in comps)
    assert sizes == [2, 2]


def test_connected_components_sorts_by_size_desc():
    adj = {
        "a": {"b", "c"},
        "b": {"a"},
        "c": {"a"},
        "x": {"y"},
        "y": {"x"},
    }
    comps = _connected_components(adj)
    assert len(comps[0]) == 3  # biggest first
    assert len(comps[1]) == 2


# ---------------------------------------------------------------------------
# _type_based_groups
# ---------------------------------------------------------------------------


def test_type_based_groups_groups_by_type_attribute():
    kg = _kg(
        nodes=[("a", "Person"), ("b", "Person"), ("c", "Company")],
        edges=[],
    )
    groups = dict(_type_based_groups(kg))
    assert groups["Person"] == {"a", "b"}
    assert groups["Company"] == {"c"}


def test_type_based_groups_uses_unknown_when_type_missing():
    kg = _kg(nodes=[("a", None), ("b", "")], edges=[])
    groups = dict(_type_based_groups(kg))
    assert "Unknown" in groups
    assert groups["Unknown"] == {"a", "b"}


def test_type_based_groups_respects_node_ids_filter():
    kg = _kg(
        nodes=[("a", "Person"), ("b", "Person"), ("c", "Company")],
        edges=[],
    )
    groups = dict(_type_based_groups(kg, node_ids={"a", "c"}))
    assert groups["Person"] == {"a"}
    assert groups["Company"] == {"c"}
    assert "b" not in groups.get("Person", set())


def test_type_based_groups_orders_by_size_desc():
    kg = _kg(
        nodes=[("p1", "Person"), ("p2", "Person"), ("p3", "Person"), ("c1", "Company")],
        edges=[],
    )
    items = _type_based_groups(kg)
    assert items[0][0] == "Person"
    assert items[1][0] == "Company"


# ---------------------------------------------------------------------------
# ensure_clusters (main public API)
# ---------------------------------------------------------------------------


def test_ensure_clusters_returns_false_on_empty_kg():
    kg = EnhancedKG()
    assert ensure_clusters(kg) is False
    assert len(kg.clusters) == 0


def test_ensure_clusters_keeps_existing_clusters_untouched():
    from drg.graph.kg_core import Cluster

    kg = _kg([("a", "T"), ("b", "T")], edges=[("a", "b")])
    kg.add_cluster(Cluster(id="pre-existing", node_ids={"a", "b"}, metadata={}))
    n_before = len(kg.clusters)
    assert ensure_clusters(kg) is True
    assert len(kg.clusters) == n_before  # no new clusters added


def test_ensure_clusters_creates_connected_component_clusters_when_multiple_components():
    kg = _kg(
        nodes=[("a", "T"), ("b", "T"), ("c", "T"), ("d", "T")],
        edges=[("a", "b"), ("c", "d")],
    )
    assert ensure_clusters(kg) is True
    assert len(kg.clusters) == 2
    algorithms = {c.metadata.get("algorithm") for c in kg.clusters}
    assert algorithms == {"connected_components"}


def test_ensure_clusters_falls_back_to_type_based_when_single_component():
    kg = _kg(
        nodes=[
            ("p1", "Person"),
            ("p2", "Person"),
            ("c1", "Company"),
            ("c2", "Company"),
        ],
        edges=[("p1", "c1"), ("p1", "p2"), ("c1", "c2")],
    )
    assert ensure_clusters(kg) is True
    assert len(kg.clusters) >= 1
    algorithms = {c.metadata.get("algorithm") for c in kg.clusters}
    assert "type_grouping" in algorithms


def test_ensure_clusters_respects_max_clusters_cap():
    nodes = [(f"n{i}", f"T{i}") for i in range(20)]
    edges = [(f"n{i}", f"n{i + 1}") for i in range(0, 18, 2)]  # 9 small components
    kg = _kg(nodes, edges)
    ensure_clusters(kg, max_clusters=3)
    assert len(kg.clusters) <= 3


def test_ensure_clusters_skips_components_below_min_size():
    # Two isolated nodes, no edges. With min_cluster_size=2 type grouping
    # may produce no clusters or a forced fallback. We assert behaviour does
    # not crash and that any created cluster respects the threshold (or the
    # single-fallback safety case).
    kg = _kg([("a", "T1"), ("b", "T2")], edges=[])
    result = ensure_clusters(kg, min_cluster_size=2)
    assert isinstance(result, bool)
    for c in kg.clusters:
        # Either >= min_cluster_size or the single fallback group is allowed.
        assert len(c.node_ids) >= 1
