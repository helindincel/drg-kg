"""Unit tests for :mod:`drg.graph.hub_mitigation`.

The module exposes one public function (``apply_hub_relation_proxy_split``)
that mutates an :class:`EnhancedKG` in place to re-route high-degree
"hub" nodes through per-relationship proxy nodes for nicer layouts.

These tests cover every meaningful path:

    - ``enabled=False`` short-circuit
    - ``hub_degree_threshold < 3`` validation
    - No-hub case (returns zeroed stats)
    - Single hub with multiple relationship types: hub detection, proxy
      creation, edge re-routing both as source and as target, connector
      edges, semantic-edge preservation (relationship_type / detail /
      temporal / confidence / is_negated all flow through to the proxy
      edge).
    - Multiple hubs with overlapping relations: ids are unique and stable
      under the case-insensitive sort.
    - Non-hub edges are kept untouched.
    - Stats dictionary correctness (hubs / proxy_nodes / edges_replaced /
      connector_edges).
    - Idempotence / no-duplicate guarantees of ``seen_new`` tracking via
      a repeated-edge graph.

All tests are deterministic and run without LLM, network, or optional
deps.
"""

from __future__ import annotations

import pytest

from drg.graph.hub_mitigation import apply_hub_relation_proxy_split
from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kg(
    nodes: list[tuple[str, str | None]],
    edges: list[tuple[str, str, str]],
) -> EnhancedKG:
    """Build a tiny KG inline. edges: (src, tgt, rel_type)."""
    kg = EnhancedKG()
    for nid, ntype in nodes:
        kg.add_node(KGNode(id=nid, type=ntype))
    for src, tgt, rel in edges:
        kg.add_edge(
            KGEdge(
                source=src,
                target=tgt,
                relationship_type=rel,
                relationship_detail=f"{src} {rel} {tgt}",
            )
        )
    return kg


def _star_kg(hub: str, spokes: int, rel: str = "knows") -> EnhancedKG:
    """Build a star graph: one hub connected to N spoke nodes via the same relation."""
    nodes = [(hub, "Person")] + [(f"spoke{i}", "Person") for i in range(spokes)]
    edges = [(hub, f"spoke{i}", rel) for i in range(spokes)]
    return _kg(nodes, edges)


# ---------------------------------------------------------------------------
# Guard clauses
# ---------------------------------------------------------------------------


def test_disabled_returns_zero_stats_and_does_not_mutate():
    kg = _star_kg("alice", spokes=15)
    edges_before = list(kg.edges)
    stats = apply_hub_relation_proxy_split(kg, enabled=False)
    assert stats == {"hubs": 0, "proxy_nodes": 0, "edges_replaced": 0, "connector_edges": 0}
    assert kg.edges == edges_before  # no mutation


@pytest.mark.parametrize("bad", [0, 1, 2, -5])
def test_threshold_below_three_is_rejected(bad: int):
    kg = _star_kg("alice", spokes=15)
    with pytest.raises(ValueError, match="hub_degree_threshold must be >= 3"):
        apply_hub_relation_proxy_split(kg, hub_degree_threshold=bad)


def test_no_hub_returns_zeros_and_leaves_graph_intact():
    # Threshold = 10; nobody hits degree 10.
    kg = _kg(
        nodes=[("a", "P"), ("b", "P"), ("c", "P")],
        edges=[("a", "b", "knows"), ("b", "c", "knows")],
    )
    edges_before = list(kg.edges)
    nodes_before = dict(kg.nodes)
    stats = apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    assert stats == {"hubs": 0, "proxy_nodes": 0, "edges_replaced": 0, "connector_edges": 0}
    assert kg.edges == edges_before
    assert kg.nodes == nodes_before


# ---------------------------------------------------------------------------
# Single-hub happy path
# ---------------------------------------------------------------------------


def test_single_hub_single_relation_creates_one_proxy_and_one_connector():
    # 12 spokes ⇒ alice's degree = 12 ⇒ hub at threshold=10.
    kg = _star_kg("alice", spokes=12)
    stats = apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)

    assert stats["hubs"] == 1
    assert stats["proxy_nodes"] == 1
    assert stats["connector_edges"] == 1
    assert stats["edges_replaced"] == 12

    # A single proxy node should now exist for alice's 'knows' relation.
    proxy_ids = [nid for nid in kg.nodes if nid.startswith("hubproxy::")]
    assert len(proxy_ids) == 1
    assert proxy_ids[0] == "hubproxy::alice::knows"
    assert kg.nodes[proxy_ids[0]].type == "HubProxy"
    assert kg.nodes[proxy_ids[0]].metadata["hub"] == "alice"
    assert kg.nodes[proxy_ids[0]].metadata["relationship_type"] == "knows"
    assert kg.nodes[proxy_ids[0]].metadata["edge_count"] == 12


def test_single_hub_multi_relation_creates_one_proxy_per_relation():
    # alice has 6 'knows' edges + 6 'collaborates_with' edges = degree 12.
    nodes = [("alice", "Person")] + [(f"spoke{i}", "Person") for i in range(12)]
    edges = []
    for i in range(6):
        edges.append(("alice", f"spoke{i}", "knows"))
    for i in range(6, 12):
        edges.append(("alice", f"spoke{i}", "collaborates_with"))
    kg = _kg(nodes, edges)
    stats = apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)

    assert stats["hubs"] == 1
    assert stats["proxy_nodes"] == 2  # one per relation type
    assert stats["connector_edges"] == 2

    proxy_ids = sorted(nid for nid in kg.nodes if nid.startswith("hubproxy::"))
    assert proxy_ids == [
        "hubproxy::alice::collaborates_with",
        "hubproxy::alice::knows",
    ]


def test_hub_at_threshold_is_treated_as_hub():
    # Threshold = degree exactly → still a hub (>=, not >).
    kg = _star_kg("alice", spokes=10)
    stats = apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    assert stats["hubs"] == 1


def test_hub_below_threshold_is_not_a_hub():
    kg = _star_kg("alice", spokes=9)
    stats = apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    assert stats["hubs"] == 0


# ---------------------------------------------------------------------------
# Edge re-routing semantics
# ---------------------------------------------------------------------------


def test_hub_as_source_proxy_edge_uses_proxy_as_source():
    kg = _star_kg("alice", spokes=12)  # all edges: alice -> spoke
    apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    # After re-routing, the spoke-pointing edges should originate from
    # the proxy node, not alice.
    proxy_id = "hubproxy::alice::knows"
    proxy_outgoing = [e for e in kg.edges if e.source == proxy_id]
    # 12 re-routed edges (one per spoke), plus 0 connectors (connector
    # goes alice -> proxy, source is alice not proxy).
    assert len(proxy_outgoing) == 12
    assert all(e.target.startswith("spoke") for e in proxy_outgoing)


def test_hub_as_target_proxy_edge_uses_proxy_as_target():
    # 12 spokes pointing AT alice ⇒ alice is the target.
    nodes = [("alice", "Person")] + [(f"spoke{i}", "Person") for i in range(12)]
    edges = [(f"spoke{i}", "alice", "knows") for i in range(12)]
    kg = _kg(nodes, edges)
    apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    proxy_id = "hubproxy::alice::knows"
    proxy_incoming = [e for e in kg.edges if e.target == proxy_id and e.source.startswith("spoke")]
    assert len(proxy_incoming) == 12


def test_connector_edge_is_hub_to_proxy():
    kg = _star_kg("alice", spokes=12)
    apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    connectors = [e for e in kg.edges if e.metadata.get("proxy_kind") == "hub_proxy_connector"]
    assert len(connectors) == 1
    assert connectors[0].source == "alice"
    assert connectors[0].target == "hubproxy::alice::knows"


def test_proxy_edge_preserves_semantic_fields():
    nodes = [("alice", "Person"), ("bob", "Person")] + [(f"s{i}", "Person") for i in range(11)]
    edges = [
        # 11 padding edges to push alice over threshold...
        ("alice", f"s{i}", "knows")
        for i in range(11)
    ]
    edges.append(("alice", "bob", "knows"))  # ...plus one we'll inspect
    kg = _kg(nodes, edges)

    # Mutate the alice->bob edge with temporal + confidence + negation.
    target = next(e for e in kg.edges if e.source == "alice" and e.target == "bob")
    target.start_time = "2024-01-01"
    target.end_time = "2024-12-31"
    target.confidence = 0.77
    target.is_negated = True
    # Add a custom metadata key — should survive on the proxy edge.
    target.metadata["custom"] = "preserved"

    apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)

    # The proxy version of that edge: source=proxy, target=bob, same rel.
    proxy_edge = next(
        e for e in kg.edges if e.source.startswith("hubproxy::alice::") and e.target == "bob"
    )
    assert proxy_edge.relationship_type == "knows"
    assert proxy_edge.start_time == "2024-01-01"
    assert proxy_edge.end_time == "2024-12-31"
    assert proxy_edge.confidence == 0.77
    assert proxy_edge.is_negated is True
    assert proxy_edge.metadata["custom"] == "preserved"
    assert proxy_edge.metadata["proxy_kind"] == "hub_split_edge"
    assert proxy_edge.metadata["hub"] == "alice"


def test_non_hub_edges_are_preserved_unchanged():
    # alice (hub) has 12 spokes. Add an isolated pair bob<->carol; that
    # edge should be retained as-is.
    kg = _star_kg("alice", spokes=12)
    kg.add_node(KGNode(id="bob", type="Person"))
    kg.add_node(KGNode(id="carol", type="Person"))
    kg.add_edge(
        KGEdge(
            source="bob",
            target="carol",
            relationship_type="knows",
            relationship_detail="bob knows carol",
        )
    )
    apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    # The original bob->carol edge should still be there, untouched.
    bob_carol = [e for e in kg.edges if e.source == "bob" and e.target == "carol"]
    assert len(bob_carol) == 1
    assert bob_carol[0].relationship_detail == "bob knows carol"
    assert "proxy_kind" not in bob_carol[0].metadata


# ---------------------------------------------------------------------------
# Multi-hub
# ---------------------------------------------------------------------------


def test_multiple_hubs_each_get_their_own_proxies():
    # Two hubs (alice + bob), each star-shaped with their own spokes.
    nodes = (
        [("alice", "Person"), ("bob", "Person")]
        + [(f"a{i}", "Person") for i in range(12)]
        + [(f"b{i}", "Person") for i in range(12)]
    )
    edges = [("alice", f"a{i}", "knows") for i in range(12)] + [
        ("bob", f"b{i}", "knows") for i in range(12)
    ]
    kg = _kg(nodes, edges)
    stats = apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    assert stats["hubs"] == 2
    assert stats["proxy_nodes"] == 2
    proxy_ids = sorted(nid for nid in kg.nodes if nid.startswith("hubproxy::"))
    assert proxy_ids == [
        "hubproxy::alice::knows",
        "hubproxy::bob::knows",
    ]


def test_hub_to_hub_edge_routed_through_source_hub_proxy():
    # alice + bob both hubs; the edge between them should route through
    # the source hub's proxy (alice's), since `e.source in hubs` wins.
    nodes = (
        [("alice", "Person"), ("bob", "Person")]
        + [(f"a{i}", "Person") for i in range(11)]
        + [(f"b{i}", "Person") for i in range(11)]
    )
    edges = (
        [("alice", f"a{i}", "knows") for i in range(11)]
        + [("bob", f"b{i}", "knows") for i in range(11)]
        + [("alice", "bob", "knows")]
    )
    kg = _kg(nodes, edges)
    apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    # The re-routed alice->bob edge: source must be alice's proxy.
    rerouted = [e for e in kg.edges if e.source == "hubproxy::alice::knows" and e.target == "bob"]
    assert len(rerouted) == 1


# ---------------------------------------------------------------------------
# Custom configuration knobs
# ---------------------------------------------------------------------------


def test_custom_proxy_prefix_and_node_type_are_honoured():
    kg = _star_kg("alice", spokes=12)
    stats = apply_hub_relation_proxy_split(
        kg,
        hub_degree_threshold=10,
        proxy_node_type="MyProxy",
        proxy_id_prefix="__proxy_",
    )
    proxy_ids = [nid for nid in kg.nodes if nid.startswith("__proxy_")]
    assert len(proxy_ids) == 1
    assert kg.nodes[proxy_ids[0]].type == "MyProxy"
    assert stats["proxy_nodes"] == 1


# ---------------------------------------------------------------------------
# Idempotence / no-duplicate guarantees
# ---------------------------------------------------------------------------


def test_duplicate_input_edges_yield_unique_new_edges():
    # Add the same (source,target,rel,detail) twice. The deduplication
    # via `seen_new` should keep just one copy after re-routing.
    nodes = [("alice", "Person")] + [(f"s{i}", "Person") for i in range(11)]
    edges = [("alice", f"s{i}", "knows") for i in range(11)]
    kg = _kg(nodes, edges)
    # Inject a duplicate edge directly (the helper-built ones are unique).
    kg.edges.append(
        KGEdge(
            source="alice",
            target="s0",
            relationship_type="knows",
            relationship_detail="alice knows s0",  # same detail
        )
    )
    apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    # After re-routing, the proxy->s0 edge with that exact detail should
    # appear only once.
    matching = [
        e
        for e in kg.edges
        if e.target == "s0"
        and e.source == "hubproxy::alice::knows"
        and e.relationship_detail == "alice knows s0"
    ]
    assert len(matching) == 1


def test_stats_match_observed_graph_state_after_run():
    kg = _star_kg("alice", spokes=12)
    stats = apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)

    # Verify stats by counting the graph directly.
    proxy_nodes = [
        nid for nid in kg.nodes if nid.startswith("hubproxy::") and kg.nodes[nid].type == "HubProxy"
    ]
    connector_edges = [e for e in kg.edges if e.metadata.get("proxy_kind") == "hub_proxy_connector"]
    proxy_edges = [e for e in kg.edges if e.metadata.get("proxy_kind") == "hub_split_edge"]

    assert stats["proxy_nodes"] == len(proxy_nodes)
    assert stats["connector_edges"] == len(connector_edges)
    assert stats["edges_replaced"] == len(proxy_edges)


# ---------------------------------------------------------------------------
# Endpoint-existence guarantee
# ---------------------------------------------------------------------------


def test_orphan_endpoint_in_edges_is_backfilled_as_node():
    # Push alice over threshold using add_edge (clean), then directly
    # append a malformed edge into kg.edges that points to a node not
    # yet registered. The post-pass should backfill it.
    kg = _star_kg("alice", spokes=12)
    kg.edges.append(
        KGEdge(
            source="alice",
            target="hidden",
            relationship_type="knows",
            relationship_detail="alice knows hidden",
        )
    )
    apply_hub_relation_proxy_split(kg, hub_degree_threshold=10)
    assert "hidden" in kg.nodes
    assert kg.nodes["hidden"].type is None  # default type for backfilled nodes
