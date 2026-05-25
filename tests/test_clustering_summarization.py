"""Unit tests for drg.clustering.summarization.

`ClusterSummarizer` is intentionally LLM-free in its default path (template
summarization). These tests exercise that template path on small synthetic
`Cluster` fixtures — no DSPy, no network, fully deterministic.
"""

from __future__ import annotations

from drg.clustering.algorithms import Cluster
from drg.clustering.summarization import (
    ClusterSummarizer,
    ClusterSummary,
    create_summarizer,
)


def _cluster(
    cluster_id: int = 1,
    nodes: list[str] | None = None,
    edges: list[tuple[str, str, str]] | None = None,
    **md,
) -> Cluster:
    # Use explicit None checks so callers can pass empty lists/dicts on
    # purpose without falling back to defaults.
    return Cluster(
        cluster_id=cluster_id,
        nodes=nodes if nodes is not None else ["Alice", "Bob", "Carol"],
        edges=edges
        if edges is not None
        else [("Alice", "knows", "Bob"), ("Bob", "knows", "Carol")],
        metadata=md if md else {"node_count": 3, "edge_count": 2},
    )


# ---------------------------------------------------------------------------
# Construction and factory
# ---------------------------------------------------------------------------


def test_create_summarizer_default_is_template_based():
    s = create_summarizer()
    assert isinstance(s, ClusterSummarizer)
    assert s.use_llm is False


def test_summarizer_with_use_llm_falls_back_when_dspy_missing(monkeypatch):
    # Simulate DSPy not being importable by injecting an ImportError.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dspy":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    s = ClusterSummarizer(use_llm=True)
    # The constructor catches the ImportError and disables use_llm.
    assert s.use_llm is False


# ---------------------------------------------------------------------------
# summarize (template path)
# ---------------------------------------------------------------------------


def test_summarize_returns_ClusterSummary_with_expected_fields():
    s = ClusterSummarizer()
    summary = s.summarize(_cluster())
    assert isinstance(summary, ClusterSummary)
    assert summary.cluster_id == 1
    assert summary.name.endswith("_Community")
    assert "entities" in summary.description.lower()
    assert summary.key_entities  # not empty


def test_summarize_uses_default_name_when_no_nodes():
    s = ClusterSummarizer()
    cluster = _cluster(nodes=[], edges=[], cluster_id=42)
    summary = s.summarize(cluster)
    assert "42" in summary.name


def test_summarize_returns_top_relations_sorted_by_frequency():
    s = ClusterSummarizer()
    edges = [
        ("A", "knows", "B"),
        ("A", "knows", "B"),  # duplicate -> frequency 2
        ("B", "works_at", "Acme"),
    ]
    cluster = _cluster(nodes=["A", "B", "Acme"], edges=edges)
    summary = s.summarize(cluster)
    assert summary.key_relations  # at least one
    # First entry is the most frequent
    top = summary.key_relations[0]
    assert top["source"] == "A"
    assert top["relation"] == "knows"
    assert top["target"] == "B"
    assert top["frequency"] == 2


def test_summarize_caps_key_entities_at_ten():
    s = ClusterSummarizer()
    nodes = [f"n{i}" for i in range(25)]
    cluster = _cluster(nodes=nodes, edges=[])
    summary = s.summarize(cluster)
    assert len(summary.key_entities) <= 10


def test_summarize_caps_key_relations_at_five():
    s = ClusterSummarizer()
    edges = [(f"s{i}", "rel", f"t{i}") for i in range(20)]
    nodes = [n for tup in edges for n in (tup[0], tup[2])]
    cluster = _cluster(nodes=nodes, edges=edges)
    summary = s.summarize(cluster)
    assert len(summary.key_relations) <= 5


def test_summarize_statistics_carry_metadata():
    s = ClusterSummarizer()
    cluster = _cluster(
        nodes=["A", "B"],
        edges=[("A", "rel", "B")],
        algorithm="louvain",
        node_count=2,
        edge_count=1,
    )
    summary = s.summarize(cluster)
    assert summary.statistics["node_count"] == 2
    assert summary.statistics["edge_count"] == 1
    assert summary.statistics["algorithm"] == "louvain"
    assert "density" in summary.statistics


def test_calculate_density_for_singleton_is_zero():
    s = ClusterSummarizer()
    cluster = _cluster(nodes=["A"], edges=[])
    summary = s.summarize(cluster)
    assert summary.statistics["density"] == 0.0


def test_calculate_density_for_fully_connected_triangle_is_one():
    s = ClusterSummarizer()
    cluster = _cluster(
        nodes=["A", "B", "C"],
        edges=[
            ("A", "r", "B"),
            ("B", "r", "C"),
            ("A", "r", "C"),
        ],
    )
    summary = s.summarize(cluster)
    # max_edges = 3*(3-1)/2 = 3; actual = 3 -> density = 1.0
    assert summary.statistics["density"] == 1.0


# ---------------------------------------------------------------------------
# summarize_all
# ---------------------------------------------------------------------------


def test_summarize_all_returns_one_summary_per_cluster():
    s = ClusterSummarizer()
    clusters = [_cluster(cluster_id=i) for i in range(3)]
    summaries = s.summarize_all(clusters)
    assert len(summaries) == 3
    assert [x.cluster_id for x in summaries] == [0, 1, 2]


def test_summarize_all_empty_list_returns_empty_list():
    assert ClusterSummarizer().summarize_all([]) == []


# ---------------------------------------------------------------------------
# LLM path (currently falls back to template — verify it doesn't crash)
# ---------------------------------------------------------------------------


def test_summarizer_llm_path_falls_back_silently():
    # Force use_llm=True; even when DSPy IS importable, _llm_summarize is a
    # placeholder that delegates to template. Just verify no crash and a
    # ClusterSummary still comes back.
    s = ClusterSummarizer(use_llm=True)
    # `s.use_llm` may now be True or False depending on whether dspy actually
    # imported; either way `summarize` must work end-to-end.
    summary = s.summarize(_cluster())
    assert isinstance(summary, ClusterSummary)
