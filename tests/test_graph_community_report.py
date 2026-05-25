"""Unit tests for :mod:`drg.graph.community_report`.

The module is pure-Python, no external dependencies; coverage stood at
10% before this commit purely because no test file targeted it. These
tests exercise:

    - ``CommunityReport.to_dict`` serialisation shape
    - The five private helpers driven via the public ``generate_report``:
      summary string, top actors, top relationship types, themes
      (entity-type theme, relation-map theme, density themes), density.
    - The branch through the rel_theme_map (mapped + fallback ``f"{rel}
      networks"``).
    - Density boundaries (``< 2`` nodes => 0, possible_edges == 0
      fallback, the >0.5 / <0.2 theme thresholds).
    - ``generate_all_reports`` over multiple clusters.
    - ``export_reports_json`` file IO via ``tmp_path``.
    - ``generate_report_text`` text-report formatting in three branches
      (with metadata / without metadata / empty graph).

All tests are deterministic and run without LLM, network, or optional
deps.
"""

from __future__ import annotations

import json

import pytest

from drg.graph.community_report import CommunityReport, CommunityReportGenerator
from drg.graph.kg_core import Cluster, EnhancedKG, KGEdge, KGNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kg(
    nodes: list[tuple[str, str | None]],
    edges: list[tuple[str, str, str]],
    clusters: list[tuple[str, set[str], dict | None]] | None = None,
) -> EnhancedKG:
    """Build a small KG inline.

    nodes:    [(node_id, node_type), ...]
    edges:    [(source, target, relationship_type), ...]
    clusters: [(cluster_id, node_ids, metadata), ...] (optional)
    """
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
    if clusters:
        for cid, node_ids, md in clusters:
            kg.add_cluster(Cluster(id=cid, node_ids=node_ids, metadata=md or {}))
    return kg


# ---------------------------------------------------------------------------
# CommunityReport.to_dict
# ---------------------------------------------------------------------------


def test_community_report_to_dict_shape():
    report = CommunityReport(
        cluster_id="c1",
        summary="Brief summary.",
        top_actors=[("alice", 4), ("bob", 2)],
        top_relationships=[("knows", 3), ("works_with", 1)],
        themes=["Person-centric", "highly connected"],
        metadata={"node_count": 3, "edge_count": 4},
    )
    d = report.to_dict()
    assert d["cluster_id"] == "c1"
    assert d["summary"] == "Brief summary."
    assert d["top_actors"] == [
        {"id": "alice", "connections": 4},
        {"id": "bob", "connections": 2},
    ]
    assert d["top_relationships"] == [
        {"type": "knows", "count": 3},
        {"type": "works_with", "count": 1},
    ]
    assert d["themes"] == ["Person-centric", "highly connected"]
    assert d["metadata"] == {"node_count": 3, "edge_count": 4}


def test_community_report_default_metadata_is_empty_dict():
    # The `metadata` field defaults to an empty dict via
    # `field(default_factory=dict)`.
    report = CommunityReport(
        cluster_id="c0",
        summary="",
        top_actors=[],
        top_relationships=[],
        themes=[],
    )
    assert report.metadata == {}
    assert report.to_dict()["metadata"] == {}


# ---------------------------------------------------------------------------
# generate_report — main entry point
# ---------------------------------------------------------------------------


def test_generate_report_basic_fields_for_dense_person_cluster():
    kg = _kg(
        nodes=[("alice", "Person"), ("bob", "Person"), ("carol", "Person")],
        edges=[
            ("alice", "bob", "knows"),
            ("bob", "carol", "knows"),
            ("alice", "carol", "knows"),
        ],
        clusters=[("c1", {"alice", "bob", "carol"}, {"algo": "louvain"})],
    )
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(kg.clusters["c1"])

    assert report.cluster_id == "c1"
    assert "3 entities" in report.summary
    assert "'Person'" in report.summary
    assert "3 internal relationships" in report.summary
    assert "'knows'" in report.summary

    # All three nodes appear as actors; each is touched by 2 edges.
    actor_ids = {a for a, _ in report.top_actors}
    assert actor_ids == {"alice", "bob", "carol"}

    # Only one distinct relationship type in this cluster.
    assert report.top_relationships == [("knows", 3)]

    # Themes include the dominant entity type and density bucket.
    assert "Person-centric" in report.themes
    # Density = 3 / (3 * 2) = 0.5 ⇒ neither >0.5 nor <0.2 ⇒ no density theme.
    assert "highly connected" not in report.themes
    assert "loosely connected" not in report.themes

    # Metadata reflects raw counts and merges cluster.metadata.
    assert report.metadata["node_count"] == 3
    assert report.metadata["edge_count"] == 3
    assert report.metadata["density"] == pytest.approx(0.5)
    assert report.metadata["algo"] == "louvain"


def test_generate_report_caps_actors_and_relationships():
    kg = _kg(
        nodes=[(f"n{i}", "T") for i in range(6)],
        edges=[(f"n{i}", f"n{(i + 1) % 6}", "rel") for i in range(6)],
        clusters=[("c1", {f"n{i}" for i in range(6)}, None)],
    )
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(kg.clusters["c1"], max_actors=2, max_relationships=1)
    assert len(report.top_actors) == 2
    assert len(report.top_relationships) == 1


def test_generate_report_skips_missing_nodes_in_cluster():
    # cluster claims a node that doesn't exist; generate_report must
    # silently drop it instead of raising.
    kg = _kg(
        nodes=[("alice", "Person"), ("bob", "Person")],
        edges=[("alice", "bob", "knows")],
    )
    # Build cluster manually to bypass EnhancedKG.add_cluster guard.
    cluster = Cluster(id="c1", node_ids={"alice", "bob", "ghost"})
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(cluster)
    # Summary should still report 2 entities (ghost filtered out).
    assert "2 entities" in report.summary


# ---------------------------------------------------------------------------
# _generate_summary edge cases
# ---------------------------------------------------------------------------


def test_summary_no_relationships_falls_back_to_default_text():
    kg = _kg(
        nodes=[("alice", "Person"), ("bob", "Person")],
        edges=[],
        clusters=[("c1", {"alice", "bob"}, None)],
    )
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(kg.clusters["c1"])
    assert "0 internal relationships" in report.summary
    # When there are no relationships, the fallback word "relationships"
    # is wired in but the conditional "most common type" sentence is not.
    assert "The most common relationship type" not in report.summary


def test_summary_no_nodes_uses_entities_fallback():
    kg = EnhancedKG()
    cluster = Cluster(id="c1", node_ids={"ghost"})  # node doesn't exist
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(cluster)
    # No real cluster_nodes ⇒ most_common_type fallback = "entities".
    assert "primarily of type 'entities'" in report.summary


def test_summary_handles_nodes_without_type_as_unknown():
    kg = _kg(
        nodes=[("alice", None), ("bob", None)],
        edges=[],
        clusters=[("c1", {"alice", "bob"}, None)],
    )
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(kg.clusters["c1"])
    assert "'Unknown'" in report.summary


# ---------------------------------------------------------------------------
# _identify_themes — branch coverage
# ---------------------------------------------------------------------------


def test_themes_skip_entity_type_theme_when_dominant_is_unknown():
    kg = _kg(
        nodes=[("a", None), ("b", None)],
        edges=[("a", "b", "rel")],
        clusters=[("c1", {"a", "b"}, None)],
    )
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(kg.clusters["c1"])
    # Dominant type is "Unknown" ⇒ the "X-centric" theme is omitted.
    assert not any(t.endswith("-centric") for t in report.themes)


def test_themes_use_rel_theme_map_for_known_relations():
    kg = _kg(
        nodes=[("a", "Person"), ("b", "Person")],
        edges=[("a", "b", "collaborates_with")],
        clusters=[("c1", {"a", "b"}, None)],
    )
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(kg.clusters["c1"])
    assert "collaboration networks" in report.themes


def test_themes_fallback_for_unmapped_relation_uses_generic_label():
    kg = _kg(
        nodes=[("a", "Person"), ("b", "Person")],
        edges=[("a", "b", "exotic_relation")],
        clusters=[("c1", {"a", "b"}, None)],
    )
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(kg.clusters["c1"])
    assert "exotic_relation networks" in report.themes


def test_themes_emit_highly_connected_when_density_above_half():
    # 3 nodes fully connected (incl. both directions): 6 edges.
    # density = 6 / (3*2) = 1.0 > 0.5
    kg = _kg(
        nodes=[("a", "P"), ("b", "P"), ("c", "P")],
        edges=[
            ("a", "b", "r"),
            ("b", "a", "r"),
            ("a", "c", "r"),
            ("c", "a", "r"),
            ("b", "c", "r"),
            ("c", "b", "r"),
        ],
        clusters=[("c1", {"a", "b", "c"}, None)],
    )
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(kg.clusters["c1"])
    assert "highly connected" in report.themes


def test_themes_emit_loosely_connected_when_density_below_quintile():
    # 6 isolated-ish nodes, only 1 edge ⇒ density = 1/(6*5) ≈ 0.033 < 0.2
    kg = _kg(
        nodes=[(f"n{i}", "P") for i in range(6)],
        edges=[("n0", "n1", "r")],
        clusters=[("c1", {f"n{i}" for i in range(6)}, None)],
    )
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(kg.clusters["c1"])
    assert "loosely connected" in report.themes


def test_themes_respect_max_themes_cap():
    kg = _kg(
        nodes=[("a", "Person"), ("b", "Person"), ("c", "Person")],
        edges=[
            ("a", "b", "collaborates_with"),
            ("b", "a", "collaborates_with"),
            ("a", "c", "collaborates_with"),
            ("c", "a", "collaborates_with"),
            ("b", "c", "collaborates_with"),
            ("c", "b", "collaborates_with"),
        ],
        clusters=[("c1", {"a", "b", "c"}, None)],
    )
    gen = CommunityReportGenerator(kg)
    # Without a cap we'd get: Person-centric, collaboration networks,
    # highly connected. Cap to 1 ⇒ only the first survives.
    report = gen.generate_report(kg.clusters["c1"], max_themes=1)
    assert len(report.themes) == 1


# ---------------------------------------------------------------------------
# _calculate_density boundary cases
# ---------------------------------------------------------------------------


def test_density_zero_for_single_node_cluster():
    kg = _kg(nodes=[("a", "P")], edges=[], clusters=[("c1", {"a"}, None)])
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(kg.clusters["c1"])
    assert report.metadata["density"] == 0.0


def test_density_zero_for_empty_cluster_via_direct_construction():
    # Cluster.__post_init__ rejects empty node_ids, so we patch one in.
    kg = EnhancedKG()
    cluster = Cluster(id="c1", node_ids={"ghost"})  # ghost won't resolve
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(cluster)
    # cluster_nodes ⇒ {}, len == 0 ⇒ density helper returns 0.0.
    assert report.metadata["density"] == 0.0


# ---------------------------------------------------------------------------
# _identify_top_actors — external edges still bump cluster nodes
# ---------------------------------------------------------------------------


def test_top_actors_counts_external_edges_that_touch_cluster_nodes():
    # alice (in cluster) has edges to outside-cluster nodes; the count
    # of alice should reflect those too because _identify_top_actors
    # iterates over all kg.edges, not just cluster_edges.
    kg = _kg(
        nodes=[
            ("alice", "Person"),
            ("bob", "Person"),
            ("outside1", "Person"),
            ("outside2", "Person"),
        ],
        edges=[
            ("alice", "bob", "knows"),
            ("alice", "outside1", "knows"),
            ("alice", "outside2", "knows"),
        ],
    )
    cluster = Cluster(id="c1", node_ids={"alice", "bob"})
    gen = CommunityReportGenerator(kg)
    report = gen.generate_report(cluster)
    actors_dict = dict(report.top_actors)
    # alice touched by 3 edges total; bob by 1.
    assert actors_dict["alice"] == 3
    assert actors_dict["bob"] == 1


# ---------------------------------------------------------------------------
# generate_all_reports
# ---------------------------------------------------------------------------


def test_generate_all_reports_returns_one_report_per_cluster():
    kg = _kg(
        nodes=[("a", "P"), ("b", "P"), ("c", "P"), ("d", "P")],
        edges=[("a", "b", "knows"), ("c", "d", "knows")],
        clusters=[
            ("c1", {"a", "b"}, None),
            ("c2", {"c", "d"}, None),
        ],
    )
    gen = CommunityReportGenerator(kg)
    reports = gen.generate_all_reports()
    assert len(reports) == 2
    assert {r.cluster_id for r in reports} == {"c1", "c2"}


def test_generate_all_reports_empty_kg_yields_empty_list():
    gen = CommunityReportGenerator(EnhancedKG())
    assert gen.generate_all_reports() == []


# ---------------------------------------------------------------------------
# export_reports_json — file IO
# ---------------------------------------------------------------------------


def test_export_reports_json_writes_well_formed_file(tmp_path):
    kg = _kg(
        nodes=[("a", "P"), ("b", "P")],
        edges=[("a", "b", "knows")],
        clusters=[("c1", {"a", "b"}, None)],
    )
    gen = CommunityReportGenerator(kg)
    reports = gen.generate_all_reports()
    target = tmp_path / "nested" / "reports.json"
    gen.export_reports_json(reports, str(target))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["total_clusters"] == 1
    assert data["reports"][0]["cluster_id"] == "c1"


def test_export_reports_json_creates_parent_dirs(tmp_path):
    # The wrapper must mkdir parents=True; verify with a 3-level path.
    target = tmp_path / "a" / "b" / "c" / "out.json"
    gen = CommunityReportGenerator(EnhancedKG())
    gen.export_reports_json([], str(target))
    assert target.exists()


# ---------------------------------------------------------------------------
# generate_report_text
# ---------------------------------------------------------------------------


def test_generate_report_text_includes_all_sections_when_populated():
    report = CommunityReport(
        cluster_id="c1",
        summary="A small Person cluster.",
        top_actors=[("alice", 3), ("bob", 1)],
        top_relationships=[("knows", 2)],
        themes=["Person-centric"],
        metadata={"density": 0.5},
    )
    gen = CommunityReportGenerator(EnhancedKG())
    text = gen.generate_report_text(report)
    assert "Community Report: c1" in text
    assert "A small Person cluster." in text
    assert "alice: 3 connections" in text
    assert "bob: 1 connections" in text
    assert "knows: 2 occurrences" in text
    assert "Person-centric" in text
    assert "density: 0.5" in text


def test_generate_report_text_omits_metadata_section_when_empty():
    report = CommunityReport(
        cluster_id="c1",
        summary="...",
        top_actors=[],
        top_relationships=[],
        themes=[],
        metadata={},
    )
    gen = CommunityReportGenerator(EnhancedKG())
    text = gen.generate_report_text(report)
    assert "Metadata:" not in text


def test_generate_report_text_for_empty_report_still_well_formed():
    # Nothing populated except the cluster id; the text-report formatter
    # must not raise and must still emit the section headings.
    report = CommunityReport(
        cluster_id="empty",
        summary="",
        top_actors=[],
        top_relationships=[],
        themes=[],
    )
    gen = CommunityReportGenerator(EnhancedKG())
    text = gen.generate_report_text(report)
    assert "Top Actors:" in text
    assert "Top Relationships:" in text
    assert "Themes:" in text
