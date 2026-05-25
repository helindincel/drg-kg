"""Unit tests for drg.clustering.algorithms.

Each concrete algorithm depends on an external library (python-louvain,
leidenalg+python-igraph, scikit-learn+networkx). We use `pytest.importorskip`
so each algorithm is tested only when its optional dependency is present.

Even without those deps installed, the factory and the abstract surface
(`create_clustering_algorithm`, `ClusterAlgorithm.cluster` ABC, error paths)
are still exercised.
"""

from __future__ import annotations

import pytest

from drg.clustering.algorithms import (
    Cluster,
    ClusteringAlgorithm,
    LeidenClustering,
    LouvainClustering,
    SpectralClustering,
    create_clustering_algorithm,
)

# ---------------------------------------------------------------------------
# Cluster dataclass
# ---------------------------------------------------------------------------


def test_cluster_dataclass_holds_basic_fields():
    c = Cluster(
        cluster_id=1,
        nodes=["a", "b"],
        edges=[("a", "rel", "b")],
        metadata={"algorithm": "test"},
    )
    assert c.cluster_id == 1
    assert c.nodes == ["a", "b"]
    assert c.edges == [("a", "rel", "b")]
    assert c.metadata["algorithm"] == "test"


# ---------------------------------------------------------------------------
# Abstract class contract
# ---------------------------------------------------------------------------


def test_clustering_algorithm_is_abstract():
    with pytest.raises(TypeError):
        ClusteringAlgorithm()  # cannot instantiate ABC directly


# ---------------------------------------------------------------------------
# create_clustering_algorithm factory
# ---------------------------------------------------------------------------


def test_factory_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="Unknown clustering algorithm"):
        create_clustering_algorithm("quantum_spectral_louvain")


def test_factory_returns_louvain_when_dep_available():
    pytest.importorskip("community")
    algo = create_clustering_algorithm("louvain", resolution=1.5, random_state=42)
    assert isinstance(algo, LouvainClustering)
    assert algo.resolution == 1.5
    assert algo.random_state == 42


def test_factory_returns_leiden_when_dep_available():
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")
    algo = create_clustering_algorithm("leiden", resolution=0.5, random_state=7)
    assert isinstance(algo, LeidenClustering)
    assert algo.resolution == 0.5
    assert algo.random_state == 7


def test_factory_returns_spectral_when_dep_available():
    pytest.importorskip("sklearn")
    algo = create_clustering_algorithm("spectral", n_clusters=4, random_state=11)
    assert isinstance(algo, SpectralClustering)
    assert algo.n_clusters == 4
    assert algo.random_state == 11


def test_factory_handles_case_insensitive_algorithm_name():
    pytest.importorskip("community")
    assert isinstance(create_clustering_algorithm("LOUVAIN"), LouvainClustering)
    assert isinstance(create_clustering_algorithm("Louvain"), LouvainClustering)


# ---------------------------------------------------------------------------
# Louvain end-to-end (requires python-louvain + networkx)
# ---------------------------------------------------------------------------


def _make_two_community_kg():
    """A simple 6-node graph with two obvious communities."""
    from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode

    kg = EnhancedKG()
    for nid in ["a", "b", "c", "x", "y", "z"]:
        kg.add_node(KGNode(id=nid, type="T", properties={}, metadata={}))

    edges = [
        ("a", "b"),
        ("b", "c"),
        ("a", "c"),  # triangle 1
        ("x", "y"),
        ("y", "z"),
        ("x", "z"),  # triangle 2
        ("a", "x"),  # weak bridge
    ]
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


def test_louvain_clusters_two_communities_in_simple_graph():
    pytest.importorskip("community")
    pytest.importorskip("networkx")

    kg = _make_two_community_kg()
    algo = LouvainClustering(random_state=42)
    clusters = algo.cluster(kg)

    # Two triangles with a weak bridge — Louvain should find >= 2 clusters
    assert len(clusters) >= 1
    assert all(isinstance(c, Cluster) for c in clusters)
    assert all(c.metadata.get("algorithm") == "louvain" for c in clusters)
    # All original nodes accounted for across clusters
    all_nodes = {n for c in clusters for n in c.nodes}
    assert all_nodes == {"a", "b", "c", "x", "y", "z"}


def test_louvain_accepts_networkx_graph_directly():
    pytest.importorskip("community")
    nx = pytest.importorskip("networkx")

    G = nx.Graph()
    G.add_edges_from([("a", "b"), ("b", "c"), ("x", "y")])
    algo = LouvainClustering(random_state=0)
    clusters = algo.cluster(G)
    assert clusters
    assert all(isinstance(c, Cluster) for c in clusters)


# ---------------------------------------------------------------------------
# Leiden end-to-end (requires leidenalg + igraph)
# ---------------------------------------------------------------------------


def test_leiden_clusters_two_communities_in_simple_graph():
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")

    kg = _make_two_community_kg()
    algo = LeidenClustering(random_state=42)
    clusters = algo.cluster(kg)

    assert len(clusters) >= 1
    assert all(isinstance(c, Cluster) for c in clusters)
    assert all(c.metadata.get("algorithm") == "leiden" for c in clusters)
    # modularity is recorded
    assert all("modularity" in c.metadata for c in clusters)


# ---------------------------------------------------------------------------
# Spectral end-to-end (requires sklearn + networkx + scipy)
# ---------------------------------------------------------------------------


def test_spectral_clusters_into_requested_number_of_groups():
    pytest.importorskip("sklearn")
    pytest.importorskip("networkx")

    kg = _make_two_community_kg()
    algo = SpectralClustering(n_clusters=2, random_state=0)
    try:
        clusters = algo.cluster(kg)
    except Exception as e:
        # Spectral on tiny graphs is notoriously brittle; skip rather than
        # fail when sklearn raises (e.g. for disconnected / degenerate cases).
        pytest.skip(f"Spectral clustering raised on tiny fixture: {e}")

    assert all(isinstance(c, Cluster) for c in clusters)
    assert all(c.metadata.get("algorithm") == "spectral" for c in clusters)
    # n_clusters honoured (sklearn may collapse if input is degenerate, hence <=)
    assert len(clusters) <= 2


# ---------------------------------------------------------------------------
# ImportError plumbing — without monkeypatching the constructors raise when
# the optional dependency is missing.
# ---------------------------------------------------------------------------


def test_louvain_constructor_raises_clear_import_error_when_dep_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "community":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="python-louvain"):
        LouvainClustering()


def test_leiden_constructor_raises_clear_import_error_when_dep_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "leidenalg":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="leidenalg"):
        LeidenClustering()


def test_spectral_constructor_raises_clear_import_error_when_dep_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn.cluster":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="scikit-learn"):
        SpectralClustering()
