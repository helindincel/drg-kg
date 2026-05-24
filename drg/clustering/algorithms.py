"""Graph clustering algorithms implementation."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """Represents a graph cluster."""

    cluster_id: int
    nodes: list[str]
    edges: list[tuple]  # (source, relation, target)
    metadata: dict[str, Any]


class ClusteringAlgorithm(ABC):
    """Abstract clustering algorithm interface."""

    @abstractmethod
    def cluster(self, graph) -> list[Cluster]:
        """Cluster graph into communities.

        Args:
            graph: Knowledge graph (KG instance)

        Returns:
            List of Cluster objects
        """
        pass


class LouvainClustering(ClusteringAlgorithm):
    """Louvain algorithm for community detection."""

    def __init__(self, resolution: float = 1.0, random_state: int | None = None):
        """Initialize Louvain clustering.

        Args:
            resolution: Resolution parameter (higher = more communities)
            random_state: Random seed for reproducibility
        """
        try:
            import community as community_louvain

            self.community = community_louvain
        except ImportError as err:
            raise ImportError(
                "python-louvain is required. Install with: pip install python-louvain"
            ) from err

        self.resolution = resolution
        self.random_state = random_state

    def cluster(self, graph) -> list[Cluster]:
        """Cluster graph using Louvain algorithm.

        Supports both EnhancedKG and NetworkX graphs.
        """
        import networkx as nx

        # Check if graph is already a NetworkX graph
        if isinstance(graph, nx.Graph):
            G = graph
            # Extract edges from NetworkX graph for cluster edge assignment
            original_edges = []
            for source, target in G.edges():
                relation = G[source][target].get("relation", "related")
                original_edges.append((source, relation, target))
        else:
            # Convert KG to NetworkX graph
            G = nx.Graph()

            # Add nodes
            if hasattr(graph, "nodes") and isinstance(graph.nodes, dict):
                # EnhancedKG format
                for node_id, node_data in graph.nodes.items():
                    if isinstance(node_data, dict):
                        G.add_node(node_id, **node_data)
                    else:
                        G.add_node(node_id)
            else:
                # Fallback: assume graph has iterable nodes
                for node_id in graph.nodes:
                    G.add_node(node_id)

            # Extract edges for later use
            original_edges = []

            # Add edges with weights
            if hasattr(graph, "edges"):
                # EnhancedKG format: edges is List[KGEdge]
                for edge in graph.edges:
                    if hasattr(edge, "source") and hasattr(edge, "target"):
                        # KGEdge object
                        source = edge.source
                        target = edge.target
                        relation = getattr(edge, "relationship_type", "related")
                    else:
                        # Tuple format (source, relation, target) or (source, target)
                        if len(edge) == 3:
                            source, relation, target = edge
                        elif len(edge) == 2:
                            source, target = edge
                            relation = "related"
                        else:
                            continue

                    original_edges.append((source, relation, target))

                    if G.has_edge(source, target):
                        G[source][target]["weight"] = G[source][target].get("weight", 0) + 1
                    else:
                        G.add_edge(source, target, relation=relation, weight=1.0)

        # Run Louvain algorithm
        if self.random_state is not None:
            import random

            random.seed(self.random_state)

        partition = self.community.best_partition(
            G,
            resolution=self.resolution,
            random_state=self.random_state,
        )

        # Convert partition to clusters
        clusters = {}
        for node, cluster_id in partition.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "nodes": [],
                    "edges": [],
                }
            clusters[cluster_id]["nodes"].append(node)

        # Add edges to clusters
        for source, relation, target in original_edges:
            source_cluster = partition.get(source)
            target_cluster = partition.get(target)
            if source_cluster == target_cluster and source_cluster is not None:
                clusters[source_cluster]["edges"].append((source, relation, target))

        # Create Cluster objects
        result = []
        for cluster_id, cluster_data in clusters.items():
            cluster = Cluster(
                cluster_id=cluster_id,
                nodes=cluster_data["nodes"],
                edges=cluster_data["edges"],
                metadata={
                    "algorithm": "louvain",
                    "resolution": self.resolution,
                    "node_count": len(cluster_data["nodes"]),
                    "edge_count": len(cluster_data["edges"]),
                },
            )
            result.append(cluster)

        logger.info(f"Louvain clustering found {len(result)} communities")
        return result


class LeidenClustering(ClusteringAlgorithm):
    """Leiden algorithm for community detection."""

    def __init__(self, resolution: float = 1.0, random_state: int | None = None):
        """Initialize Leiden clustering.

        Args:
            resolution: Resolution parameter (higher = more communities)
            random_state: Random seed for reproducibility
        """
        try:
            import leidenalg

            self.leidenalg = leidenalg
        except ImportError as err:
            raise ImportError("leidenalg is required. Install with: pip install leidenalg") from err

        try:
            import igraph as ig

            self.ig = ig
        except ImportError as err:
            raise ImportError(
                "python-igraph is required. Install with: pip install python-igraph"
            ) from err

        self.resolution = resolution
        self.random_state = random_state

    def cluster(self, graph) -> list[Cluster]:
        """Cluster graph using Leiden algorithm."""
        # Convert KG to igraph
        G = self.ig.Graph()

        # Add nodes
        node_list = list(graph.nodes.keys())
        node_to_index = {node: i for i, node in enumerate(node_list)}
        G.add_vertices(len(node_list))
        G.vs["name"] = node_list

        # Add edges
        edge_list = []
        edge_weights = []
        for source, _relation, target in graph.edges:
            if source in node_to_index and target in node_to_index:
                edge_list.append((node_to_index[source], node_to_index[target]))
                edge_weights.append(1.0)

        G.add_edges(edge_list)
        G.es["weight"] = edge_weights

        # Run Leiden algorithm
        partition = self.leidenalg.find_partition(
            G,
            self.leidenalg.ModularityVertexPartition,
            resolution_parameter=self.resolution,
            seed=self.random_state,
        )

        # Convert partition to clusters
        clusters = {}
        for i, cluster_id in enumerate(partition.membership):
            node = node_list[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "nodes": [],
                    "edges": [],
                }
            clusters[cluster_id]["nodes"].append(node)

        # Add edges to clusters
        for source, relation, target in graph.edges:
            source_idx = node_to_index.get(source)
            target_idx = node_to_index.get(target)
            if source_idx is not None and target_idx is not None:
                source_cluster = partition.membership[source_idx]
                target_cluster = partition.membership[target_idx]
                if source_cluster == target_cluster:
                    clusters[source_cluster]["edges"].append((source, relation, target))

        # Create Cluster objects
        result = []
        for cluster_id, cluster_data in clusters.items():
            cluster = Cluster(
                cluster_id=cluster_id,
                nodes=cluster_data["nodes"],
                edges=cluster_data["edges"],
                metadata={
                    "algorithm": "leiden",
                    "resolution": self.resolution,
                    "node_count": len(cluster_data["nodes"]),
                    "edge_count": len(cluster_data["edges"]),
                    "modularity": partition.modularity,
                },
            )
            result.append(cluster)

        logger.info(f"Leiden clustering found {len(result)} communities")
        return result


class SpectralClustering(ClusteringAlgorithm):
    """Spectral clustering algorithm."""

    def __init__(self, n_clusters: int = 5, random_state: int | None = None):
        """Initialize spectral clustering.

        Args:
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility
        """
        try:
            from sklearn.cluster import SpectralClustering as SklearnSpectralClustering

            self.SklearnSpectralClustering = SklearnSpectralClustering
        except ImportError as err:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            ) from err

        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster(self, graph) -> list[Cluster]:
        """Cluster graph using spectral clustering."""
        import networkx as nx

        # Convert KG to NetworkX graph
        G = nx.Graph()

        # Add nodes
        node_list = list(graph.nodes.keys())
        node_to_index = {node: i for i, node in enumerate(node_list)}
        for node_id, node_data in graph.nodes.items():
            G.add_node(node_id, **node_data)

        # Add edges
        for source, relation, target in graph.edges:
            if source in node_to_index and target in node_to_index:
                if G.has_edge(source, target):
                    G[source][target]["weight"] = G[source][target].get("weight", 0) + 1
                else:
                    G.add_edge(source, target, relation=relation, weight=1.0)

        # Get adjacency matrix
        adj_matrix = nx.adjacency_matrix(G, nodelist=node_list, weight="weight")

        # Run spectral clustering
        clustering = self.SklearnSpectralClustering(
            n_clusters=self.n_clusters,
            affinity="precomputed",
            random_state=self.random_state,
        )

        labels = clustering.fit_predict(adj_matrix)

        # Convert labels to clusters
        clusters = {}
        for i, node in enumerate(node_list):
            cluster_id = int(labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "nodes": [],
                    "edges": [],
                }
            clusters[cluster_id]["nodes"].append(node)

        # Add edges to clusters
        for source, relation, target in graph.edges:
            source_idx = node_to_index.get(source)
            target_idx = node_to_index.get(target)
            if source_idx is not None and target_idx is not None:
                source_cluster = int(labels[source_idx])
                target_cluster = int(labels[target_idx])
                if source_cluster == target_cluster:
                    clusters[source_cluster]["edges"].append((source, relation, target))

        # Create Cluster objects
        result = []
        for cluster_id, cluster_data in clusters.items():
            cluster = Cluster(
                cluster_id=cluster_id,
                nodes=cluster_data["nodes"],
                edges=cluster_data["edges"],
                metadata={
                    "algorithm": "spectral",
                    "n_clusters": self.n_clusters,
                    "node_count": len(cluster_data["nodes"]),
                    "edge_count": len(cluster_data["edges"]),
                },
            )
            result.append(cluster)

        logger.info(f"Spectral clustering found {len(result)} communities")
        return result


def create_clustering_algorithm(algorithm: str = "louvain", **kwargs) -> ClusteringAlgorithm:
    """Factory function to create clustering algorithm.

    Args:
        algorithm: Algorithm name ("louvain", "leiden", "spectral")
        **kwargs: Algorithm-specific parameters

    Returns:
        ClusteringAlgorithm instance
    """
    algorithm_lower = algorithm.lower()

    if algorithm_lower == "louvain":
        resolution = kwargs.get("resolution", 1.0)
        random_state = kwargs.get("random_state", None)
        return LouvainClustering(resolution=resolution, random_state=random_state)

    elif algorithm_lower == "leiden":
        resolution = kwargs.get("resolution", 1.0)
        random_state = kwargs.get("random_state", None)
        return LeidenClustering(resolution=resolution, random_state=random_state)

    elif algorithm_lower == "spectral":
        n_clusters = kwargs.get("n_clusters", 5)
        random_state = kwargs.get("random_state", None)
        return SpectralClustering(n_clusters=n_clusters, random_state=random_state)

    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
