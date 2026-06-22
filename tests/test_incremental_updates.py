"""Unit tests for :mod:`drg.graph.incremental` and the new
``EnhancedKG.from_dict`` / ``EnhancedKG.load_json`` round-trip surface.

These tests deliberately import only from ``drg.graph.*`` and
``drg.entity_resolution`` so they can run even when the environment-level
DSPy install is broken. Extraction import issues should never block the
incremental layer's regression suite.

Coverage targets
----------------
- ``EnhancedKG.from_dict`` / ``load_json`` round-trip integrity, including
  the legacy "no metadata" shape and the new "with metadata" shape.
- ``MergeStrategy`` defaults and individual policy variants
  (``PREFER_EXISTING`` / ``PREFER_NEW`` / ``UNION`` for nodes;
  ``SKIP`` / ``APPEND_EVIDENCE`` / ``MAX_CONFIDENCE`` for edges).
- ``GraphMerger`` end-to-end behaviour: exact id match, normalized name
  match, type-mismatch refusal, edge dedup (case-insensitive), edge
  rewriting through the id remap, self-loop skipping, defensive node
  insertion for orphan edge endpoints.
- ``KGDiff`` reporting: counts, summary keys, JSON shape.
- Version + history bookkeeping: monotonically-increasing version,
  append-only history, opt-out via ``record_history=False``.
- Cluster merge: id-collision skip, partial node-id remap, empty-cluster
  drop.
- ``merge_graphs`` convenience matches ``GraphMerger.merge``.
- Backward compatibility: building a fresh ``EnhancedKG`` and writing it
  to disk produces a JSON shape byte-identical to the legacy three-key
  format (no ``metadata`` field).
"""

from __future__ import annotations

import json

import pytest

from drg.graph.incremental import (
    EdgeMergePolicy,
    GraphMerger,
    KGDiff,
    MergeStrategy,
    NodeMergePolicy,
    merge_graphs,
)
from drg.graph.kg_core import Cluster, EnhancedKG, KGEdge, KGNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base() -> EnhancedKG:
    """Two-node, one-edge base KG used by most merge tests."""
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Apple Inc", type="Company", properties={"sector": "tech"}))
    kg.add_node(KGNode(id="Tim Cook", type="Person"))
    kg.add_edge(
        KGEdge(
            source="Tim Cook",
            target="Apple Inc",
            relationship_type="WORKS_AT",
            relationship_detail="ceo",
            confidence=0.9,
            metadata={"source_ref": "doc_1"},
        )
    )
    return kg


def _make_incoming() -> EnhancedKG:
    """Incoming KG with one normalized-match, one new node, one duplicate edge."""
    kg = EnhancedKG()
    kg.add_node(KGNode(id="apple inc", type="Company", properties={"industry": "technology"}))
    kg.add_node(KGNode(id="Steve Jobs", type="Person"))
    kg.add_node(KGNode(id="Tim Cook", type="Person"))
    kg.add_edge(
        KGEdge(
            source="Steve Jobs",
            target="apple inc",
            relationship_type="FOUNDED",
            relationship_detail="founder",
            confidence=0.85,
        )
    )
    kg.add_edge(
        KGEdge(
            source="Tim Cook",
            target="apple inc",
            relationship_type="works_at",  # case differs from base ("WORKS_AT")
            relationship_detail="ceo",
            confidence=0.7,
            metadata={"source_ref": "doc_2"},
        )
    )
    return kg


# ---------------------------------------------------------------------------
# EnhancedKG round-trip
# ---------------------------------------------------------------------------


def test_to_json_for_legacy_kg_omits_metadata_key():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    kg.add_node(KGNode(id="bob"))
    kg.add_edge(
        KGEdge(
            source="alice",
            target="bob",
            relationship_type="KNOWS",
            relationship_detail="x",
        )
    )
    data = json.loads(kg.to_json())
    assert "metadata" not in data
    assert set(data.keys()) == {"nodes", "edges", "clusters"}


def test_to_json_includes_metadata_only_when_populated():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="alice"))
    kg.metadata = {"version": 3}
    data = json.loads(kg.to_json())
    assert data["metadata"] == {"version": 3}


def test_from_dict_round_trip_preserves_structure(tmp_path):
    base = _make_base()
    base.add_cluster(Cluster(id="c1", node_ids={"Apple Inc", "Tim Cook"}, metadata={"algo": "x"}))
    base.metadata = {"version": 7, "history": [{"op": "init"}]}

    text = base.to_json()
    restored = EnhancedKG.from_dict(json.loads(text))
    assert set(restored.nodes.keys()) == set(base.nodes.keys())
    assert len(restored.edges) == len(base.edges)
    assert restored.clusters["c1"].node_ids == {"Apple Inc", "Tim Cook"}
    assert restored.metadata == {"version": 7, "history": [{"op": "init"}]}


def test_load_json_round_trip(tmp_path):
    base = _make_base()
    p = tmp_path / "out.json"
    base.save_json(str(p))
    restored = EnhancedKG.load_json(str(p))
    assert set(restored.nodes.keys()) == {"Apple Inc", "Tim Cook"}
    assert len(restored.edges) == 1


def test_from_dict_drops_orphan_cluster_members():
    data = {
        "nodes": [{"id": "a", "type": None}],
        "edges": [],
        "clusters": [{"id": "c1", "node_ids": ["a", "ghost"], "metadata": {}}],
    }
    kg = EnhancedKG.from_dict(data)
    assert kg.clusters["c1"].node_ids == {"a"}


def test_from_dict_drops_clusters_with_no_valid_members():
    data = {
        "nodes": [{"id": "a", "type": None}],
        "edges": [],
        "clusters": [{"id": "ghosts", "node_ids": ["x", "y"], "metadata": {}}],
    }
    kg = EnhancedKG.from_dict(data)
    assert "ghosts" not in kg.clusters


def test_from_dict_handles_legacy_kg_without_metadata_key():
    data = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {
                "source": "a",
                "target": "b",
                "relationship_type": "R",
                "relationship_detail": "x",
            }
        ],
        "clusters": [],
    }
    kg = EnhancedKG.from_dict(data)
    assert kg.metadata == {}
    assert len(kg.edges) == 1


# ---------------------------------------------------------------------------
# MergeStrategy / KGDiff value-objects
# ---------------------------------------------------------------------------


def test_merge_strategy_default_is_conservative():
    s = MergeStrategy.default()
    assert s.node_policy is NodeMergePolicy.PREFER_EXISTING
    assert s.edge_policy is EdgeMergePolicy.SKIP
    assert s.require_type_match is True
    assert s.use_normalized_match is True
    assert s.case_insensitive_relation is True


def test_kgdiff_is_empty_starts_true_and_summary_keys_stable():
    diff = KGDiff()
    assert diff.is_empty() is True
    keys = set(diff.summary().keys())
    assert keys == {
        "added_nodes",
        "merged_nodes",
        "skipped_nodes",
        "added_edges",
        "skipped_edges",
        "rewritten_edges",
        "added_clusters",
        "skipped_clusters",
    }


def test_kgdiff_to_dict_is_json_serialisable():
    diff = KGDiff(
        added_nodes=["x"],
        merged_nodes=[("a", "b")],
        added_edges=[("a", "R", "b")],
    )
    d = diff.to_dict()
    json.dumps(d)  # must not raise
    assert d["merged_nodes"] == [["a", "b"]]
    assert d["added_edges"] == [["a", "R", "b"]]


# ---------------------------------------------------------------------------
# GraphMerger — node matching
# ---------------------------------------------------------------------------


def test_merge_into_empty_base_adds_all_nodes_and_edges():
    base = EnhancedKG()
    incoming = _make_base()
    diff = GraphMerger().merge(base, incoming, document_id="doc_1")
    assert set(base.nodes.keys()) == {"Apple Inc", "Tim Cook"}
    assert len(base.edges) == 1
    assert diff.summary()["added_nodes"] == 2
    assert diff.summary()["added_edges"] == 1
    assert diff.summary()["merged_nodes"] == 0


def test_merge_with_exact_id_match_records_merged_pair():
    base = _make_base()
    inc = EnhancedKG()
    inc.add_node(KGNode(id="Tim Cook", type="Person"))
    diff = GraphMerger().merge(base, inc)
    assert diff.merged_nodes == [("Tim Cook", "Tim Cook")]
    assert diff.added_nodes == []


def test_merge_with_normalized_match_remaps_to_existing_canonical_id():
    base = _make_base()
    inc = EnhancedKG()
    inc.add_node(KGNode(id="apple inc", type="Company"))
    inc.add_node(KGNode(id="Steve Jobs", type="Person"))
    diff = GraphMerger().merge(base, inc)
    # 'apple inc' should fold into 'Apple Inc' (existing canonical id is preserved).
    assert "apple inc" not in base.nodes
    assert "Apple Inc" in base.nodes
    # Steve Jobs is genuinely new.
    assert "Steve Jobs" in base.nodes
    pairs = dict(diff.merged_nodes)
    assert pairs["Apple Inc"] == "apple inc"


def test_merge_blocks_type_mismatch_when_require_type_match_true():
    base = EnhancedKG()
    base.add_node(KGNode(id="Mercury", type="Element"))
    inc = EnhancedKG()
    inc.add_node(KGNode(id="mercury", type="Planet"))  # same surface form, different type
    diff = GraphMerger().merge(base, inc)
    # Same normalized name but different type -> insert as new node, record skip reason.
    assert "mercury" in base.nodes
    assert "Mercury" in base.nodes
    assert ("mercury", "type_mismatch") in diff.skipped_nodes
    # The new node is also reported as added.
    assert "mercury" in diff.added_nodes


def test_merge_allows_type_mismatch_when_require_type_match_false():
    base = EnhancedKG()
    base.add_node(KGNode(id="Mercury", type="Element"))
    inc = EnhancedKG()
    inc.add_node(KGNode(id="mercury", type="Planet"))
    diff = GraphMerger(MergeStrategy(require_type_match=False)).merge(base, inc)
    # The two should now collapse — only the existing node remains.
    assert "mercury" not in base.nodes
    assert "Mercury" in base.nodes
    assert diff.merged_nodes == [("Mercury", "mercury")]


def test_merge_with_normalization_disabled_requires_byte_exact_id():
    base = _make_base()
    inc = EnhancedKG()
    inc.add_node(KGNode(id="apple inc", type="Company"))
    diff = GraphMerger(MergeStrategy(use_normalized_match=False)).merge(base, inc)
    # Strict match means 'apple inc' is treated as a brand-new node.
    assert "apple inc" in base.nodes
    assert "apple inc" in diff.added_nodes


# ---------------------------------------------------------------------------
# GraphMerger — node policies
# ---------------------------------------------------------------------------


def test_prefer_existing_records_provenance_in_merged_from():
    base = _make_base()
    inc = EnhancedKG()
    inc.add_node(
        KGNode(
            id="apple inc",
            type="Company",
            properties={"industry": "technology"},
            metadata={"source_ref": "doc_2"},
        )
    )
    GraphMerger().merge(base, inc)
    apple = base.nodes["Apple Inc"]
    # Existing properties unchanged.
    assert apple.properties == {"sector": "tech"}
    # Provenance recorded.
    merged_from = apple.metadata.get("merged_from")
    assert merged_from and merged_from[0]["id"] == "apple inc"
    assert merged_from[0]["properties"] == {"industry": "technology"}


def test_prefer_new_overwrites_existing_mutable_fields():
    base = _make_base()
    inc = EnhancedKG()
    inc.add_node(
        KGNode(
            id="apple inc",
            type="Company",
            properties={"industry": "technology"},
            confidence=0.95,
        )
    )
    GraphMerger(MergeStrategy(node_policy=NodeMergePolicy.PREFER_NEW)).merge(base, inc)
    apple = base.nodes["Apple Inc"]
    assert apple.properties == {"industry": "technology"}
    assert apple.confidence == 0.95


def test_union_merges_properties_and_keeps_higher_confidence():
    base = EnhancedKG()
    base.add_node(
        KGNode(
            id="x",
            type="T",
            properties={"a": 1, "b": 2},
            confidence=0.6,
        )
    )
    inc = EnhancedKG()
    inc.add_node(
        KGNode(
            id="x",
            type="T",
            properties={"b": 99, "c": 3},
            confidence=0.8,
        )
    )
    GraphMerger(MergeStrategy(node_policy=NodeMergePolicy.UNION)).merge(base, inc)
    n = base.nodes["x"]
    # Incoming wins on overlap, missing keys retained.
    assert n.properties == {"a": 1, "b": 99, "c": 3}
    assert n.confidence == 0.8


def test_union_averages_embeddings_when_dimensions_match():
    base = EnhancedKG()
    base.add_node(KGNode(id="x", type="T", embedding=[0.0, 0.0]))
    inc = EnhancedKG()
    inc.add_node(KGNode(id="x", type="T", embedding=[1.0, 1.0]))
    GraphMerger(MergeStrategy(node_policy=NodeMergePolicy.UNION)).merge(base, inc)
    assert base.nodes["x"].embedding == [0.5, 0.5]


def test_union_does_not_average_embeddings_with_mismatched_dimensions():
    base = EnhancedKG()
    base.add_node(KGNode(id="x", type="T", embedding=[0.1, 0.2]))
    inc = EnhancedKG()
    inc.add_node(KGNode(id="x", type="T", embedding=[1.0]))
    GraphMerger(MergeStrategy(node_policy=NodeMergePolicy.UNION)).merge(base, inc)
    # Existing embedding preserved when shapes disagree.
    assert base.nodes["x"].embedding == [0.1, 0.2]


# ---------------------------------------------------------------------------
# GraphMerger — edge handling
# ---------------------------------------------------------------------------


def test_duplicate_edge_is_skipped_with_default_policy():
    base = _make_base()
    inc = _make_incoming()
    diff = GraphMerger().merge(base, inc)
    types_present = {(e.source, e.relationship_type, e.target) for e in base.edges}
    # Original WORKS_AT edge stays; duplicate works_at is skipped.
    assert ("Tim Cook", "WORKS_AT", "Apple Inc") in types_present
    assert ("Tim Cook", "works_at", "Apple Inc") not in types_present
    # Reported in diff.
    assert ("Tim Cook", "works_at", "Apple Inc") in diff.skipped_edges


def test_append_evidence_collects_source_refs_on_duplicate():
    base = _make_base()
    inc = _make_incoming()
    diff = GraphMerger(MergeStrategy(edge_policy=EdgeMergePolicy.APPEND_EVIDENCE)).merge(base, inc)
    e0 = base.edges[0]
    refs = e0.metadata.get("evidence_refs", [])
    assert any(r.get("source_ref") == "doc_2" for r in refs)
    assert ("Tim Cook", "works_at", "Apple Inc") in diff.skipped_edges


def test_max_confidence_keeps_higher_score_and_remembers_alt():
    base = EnhancedKG()
    base.add_node(KGNode(id="a", type="T"))
    base.add_node(KGNode(id="b", type="T"))
    base.add_edge(
        KGEdge(
            source="a",
            target="b",
            relationship_type="R",
            relationship_detail="x",
            confidence=0.5,
        )
    )
    inc = EnhancedKG()
    inc.add_node(KGNode(id="a", type="T"))
    inc.add_node(KGNode(id="b", type="T"))
    inc.add_edge(
        KGEdge(
            source="a",
            target="b",
            relationship_type="R",
            relationship_detail="x",
            confidence=0.95,
        )
    )
    GraphMerger(MergeStrategy(edge_policy=EdgeMergePolicy.MAX_CONFIDENCE)).merge(base, inc)
    assert base.edges[0].confidence == 0.95
    assert 0.5 in base.edges[0].metadata.get("alt_confidences", [])


def test_edge_dedup_is_case_insensitive_for_relation_type_by_default():
    base = _make_base()
    inc = _make_incoming()
    diff = GraphMerger().merge(base, inc)
    # 'works_at' (incoming) should fold into 'WORKS_AT' (base).
    assert diff.summary()["skipped_edges"] >= 1


def test_edge_dedup_can_be_case_sensitive_when_strategy_disables_it():
    base = _make_base()
    inc = _make_incoming()
    diff = GraphMerger(MergeStrategy(case_insensitive_relation=False)).merge(base, inc)
    # Now 'works_at' is treated as a separate edge.
    types = {(e.source, e.relationship_type, e.target) for e in base.edges}
    assert ("Tim Cook", "works_at", "Apple Inc") in types
    assert ("Tim Cook", "WORKS_AT", "Apple Inc") in types
    assert diff.summary()["added_edges"] >= 2


def test_edges_are_rewritten_through_node_remap():
    base = _make_base()
    inc = EnhancedKG()
    inc.add_node(KGNode(id="apple inc", type="Company"))
    inc.add_node(KGNode(id="Steve Jobs", type="Person"))
    inc.add_edge(
        KGEdge(
            source="Steve Jobs",
            target="apple inc",
            relationship_type="FOUNDED",
            relationship_detail="x",
        )
    )
    diff = GraphMerger().merge(base, inc)
    edge = next(e for e in base.edges if e.relationship_type == "FOUNDED")
    assert edge.source == "Steve Jobs"
    assert edge.target == "Apple Inc"  # rewritten to existing canonical id
    assert ("Steve Jobs", "FOUNDED", "Apple Inc") in diff.rewritten_edges


def test_self_loop_after_remap_is_skipped_not_raised():
    base = EnhancedKG()
    base.add_node(KGNode(id="Apple Inc", type="Company"))
    inc = EnhancedKG()
    inc.add_node(KGNode(id="Apple Inc", type="Company"))
    inc.add_node(KGNode(id="apple inc", type="Company"))
    # After remap both endpoints would be 'Apple Inc' -> self-loop.
    inc.edges.append(
        KGEdge(
            source="Apple Inc",
            target="apple inc",
            relationship_type="ALIAS_OF",
            relationship_detail="x",
        )
    )
    diff = GraphMerger().merge(base, inc)
    # No edge added; reported in skipped_edges so callers can investigate.
    assert base.edges == []
    assert any(t[1] == "ALIAS_OF" for t in diff.skipped_edges)


def test_orphan_edge_endpoint_in_incoming_is_inserted_defensively():
    base = EnhancedKG()
    base.add_node(KGNode(id="a", type="T"))
    inc = EnhancedKG()
    # 'b' isn't in inc.nodes but the edge references it. The merger must
    # insert it rather than crash.
    inc.add_node(KGNode(id="a", type="T"))
    inc.edges.append(
        KGEdge(
            source="a",
            target="b",
            relationship_type="R",
            relationship_detail="x",
        )
    )
    diff = GraphMerger().merge(base, inc)
    assert "b" in base.nodes
    assert ("a", "R", "b") in diff.added_edges


# ---------------------------------------------------------------------------
# Cluster merge
# ---------------------------------------------------------------------------


def test_cluster_id_collision_is_skipped():
    base = _make_base()
    base.add_cluster(Cluster(id="c1", node_ids={"Tim Cook", "Apple Inc"}))
    inc = EnhancedKG()
    inc.add_node(KGNode(id="Tim Cook", type="Person"))
    inc.add_cluster(Cluster(id="c1", node_ids={"Tim Cook"}))
    diff = GraphMerger().merge(base, inc)
    assert "c1" in diff.skipped_clusters
    # Original cluster preserved untouched.
    assert base.clusters["c1"].node_ids == {"Tim Cook", "Apple Inc"}


def test_cluster_node_ids_are_remapped_through_node_match():
    base = _make_base()
    inc = EnhancedKG()
    inc.add_node(KGNode(id="apple inc", type="Company"))
    inc.add_cluster(Cluster(id="c2", node_ids={"apple inc"}, metadata={"algo": "x"}))
    diff = GraphMerger().merge(base, inc)
    # The cluster moves over but its members are remapped to canonical ids.
    assert "c2" in diff.added_clusters
    assert base.clusters["c2"].node_ids == {"Apple Inc"}


def test_cluster_with_no_resolvable_members_is_dropped():
    base = EnhancedKG()
    inc = EnhancedKG()
    # Inject the cluster directly to bypass ``add_cluster``'s referential
    # integrity check — we want to simulate the case where an incoming
    # graph has been hand-edited or comes from a different snapshot whose
    # nodes have all been pruned.
    inc.clusters["ghost"] = Cluster(id="ghost", node_ids={"x", "y"})
    diff = GraphMerger().merge(base, inc)
    assert "ghost" not in base.clusters
    assert "ghost" in diff.skipped_clusters


# ---------------------------------------------------------------------------
# Version + history bookkeeping
# ---------------------------------------------------------------------------


def test_first_merge_initialises_version_and_timestamps():
    base = EnhancedKG()
    inc = _make_base()
    GraphMerger().merge(base, inc, document_id="doc_42")
    assert base.metadata["version"] == 1
    assert "created_at" in base.metadata
    assert "updated_at" in base.metadata
    history = base.metadata["history"]
    assert len(history) == 1
    assert history[0]["operation"] == "merge"
    assert history[0]["document_id"] == "doc_42"
    assert history[0]["added_nodes"] == 2


def test_subsequent_merges_increment_version_and_append_history():
    base = EnhancedKG()
    GraphMerger().merge(base, _make_base(), document_id="doc_1")
    GraphMerger().merge(base, _make_incoming(), document_id="doc_2")
    GraphMerger().merge(base, EnhancedKG(), document_id="doc_3")  # no-op merge
    assert base.metadata["version"] == 3
    assert [h["document_id"] for h in base.metadata["history"]] == [
        "doc_1",
        "doc_2",
        "doc_3",
    ]


def test_history_can_be_disabled_per_call():
    base = EnhancedKG()
    GraphMerger().merge(base, _make_base(), record_history=False)
    assert "version" not in base.metadata
    assert "history" not in base.metadata


def test_version_metadata_survives_save_load_round_trip(tmp_path):
    base = EnhancedKG()
    GraphMerger().merge(base, _make_base(), document_id="doc_1")
    GraphMerger().merge(base, _make_incoming(), document_id="doc_2")
    p = tmp_path / "kg.json"
    base.save_json(str(p))
    restored = EnhancedKG.load_json(str(p))
    assert restored.metadata["version"] == 2
    assert len(restored.metadata["history"]) == 2


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def test_merge_graphs_function_matches_merger_class():
    base = _make_base()
    inc = _make_incoming()
    diff = merge_graphs(base, inc, document_id="doc_x")
    assert isinstance(diff, KGDiff)
    assert diff.summary()["merged_nodes"] >= 1


def test_merger_rejects_non_enhanced_kg_inputs():
    with pytest.raises(TypeError):
        GraphMerger().merge("not a kg", _make_base())
    with pytest.raises(TypeError):
        GraphMerger().merge(_make_base(), {"not": "a kg"})


# ---------------------------------------------------------------------------
# Backward compatibility — the legacy build path is untouched
# ---------------------------------------------------------------------------


def test_legacy_kg_save_json_shape_is_byte_compatible_with_pre_incremental():
    """A KG built without the incremental layer must produce a JSON whose
    top-level keys are exactly the legacy three. Otherwise downstream
    consumers that read ``outputs/<x>_kg.json`` files break."""
    kg = EnhancedKG()
    kg.add_node(KGNode(id="a"))
    kg.add_node(KGNode(id="b"))
    kg.add_edge(
        KGEdge(
            source="a",
            target="b",
            relationship_type="R",
            relationship_detail="x",
        )
    )
    data = json.loads(kg.to_json())
    assert list(data.keys()) == ["nodes", "edges", "clusters"]


def test_legacy_kg_files_load_via_from_dict_without_metadata_assumption():
    """Files written by the legacy pipeline (no metadata key) load cleanly
    and end up with an empty metadata dict — never None."""
    legacy_data = {
        "nodes": [{"id": "a", "type": "T"}],
        "edges": [],
        "clusters": [],
    }
    kg = EnhancedKG.from_dict(legacy_data)
    assert kg.metadata == {}
    assert kg.nodes["a"].type == "T"
