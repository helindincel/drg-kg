"""Clustering module for graph community detection."""

from .algorithms import (
    Cluster,  # backward-compat alias for ClusterResult
    ClusterResult,
    ClusteringAlgorithm,
    LeidenClustering,
    LouvainClustering,
    SpectralClustering,
    create_clustering_algorithm,
)
from .summarization import ClusterSummarizer, create_summarizer

__all__ = [
    "Cluster",  # backward-compat alias — prefer ClusterResult in new code
    "ClusterResult",
    "ClusterSummarizer",
    "ClusteringAlgorithm",
    "LeidenClustering",
    "LouvainClustering",
    "SpectralClustering",
    "create_clustering_algorithm",
    "create_summarizer",
]
