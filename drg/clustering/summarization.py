"""Cluster summarization for community reports."""

import logging
from dataclasses import dataclass
from typing import Any

from .algorithms import Cluster

logger = logging.getLogger(__name__)


@dataclass
class ClusterSummary:
    """Summary of a cluster."""

    cluster_id: int
    name: str
    description: str
    key_entities: list[str]
    key_relations: list[dict[str, Any]]
    statistics: dict[str, Any]


class ClusterSummarizer:
    """Summarizes clusters with deterministic template-based rules.

    `use_llm` and `llm_model` are retained as no-op compatibility arguments
    to avoid breaking existing call sites.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_model: str | None = None,
    ):
        """Initialize cluster summarizer.

        Args:
            use_llm: Deprecated compatibility flag. Ignored.
            llm_model: Deprecated compatibility argument. Ignored.
        """
        self.use_llm = False
        self.llm_model = llm_model
        if use_llm:
            logger.info("ClusterSummarizer ignores use_llm and uses template summarization")

    def summarize(
        self,
        cluster: Cluster,
        graph=None,  # Optional KG for context
    ) -> ClusterSummary:
        """Summarize a cluster using the deterministic template path.

        Args:
            cluster: Cluster to summarize
            graph: Optional knowledge graph for additional context

        Returns:
            ClusterSummary object
        """
        return self._template_summarize(cluster, graph)

    def _template_summarize(
        self,
        cluster: Cluster,
        graph=None,
    ) -> ClusterSummary:
        """Template-based summarization."""
        # Extract key entities (top entities by frequency or importance)
        key_entities = cluster.nodes[:10]  # Top 10 nodes

        # Extract key relations
        key_relations = []
        relation_counts: dict[str, int] = {}
        for source, relation, target in cluster.edges:
            rel_key = f"{source} → {relation} → {target}"
            relation_counts[rel_key] = relation_counts.get(rel_key, 0) + 1

        # Get top relations
        sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
        for rel_key, count in sorted_relations[:5]:
            parts = rel_key.split(" → ")
            key_relations.append(
                {
                    "source": parts[0],
                    "relation": parts[1],
                    "target": parts[2],
                    "frequency": count,
                }
            )

        # Generate name
        cluster_name = f"Cluster_{cluster.cluster_id}"
        if key_entities:
            # Use first entity as part of name
            cluster_name = f"{key_entities[0]}_Community"

        # Generate description
        description = (
            f"This cluster contains {cluster.metadata.get('node_count', len(cluster.nodes))} entities "
            f"and {cluster.metadata.get('edge_count', len(cluster.edges))} relationships. "
            f"Key entities include: {', '.join(key_entities[:5])}."
        )

        return ClusterSummary(
            cluster_id=cluster.cluster_id,
            name=cluster_name,
            description=description,
            key_entities=key_entities,
            key_relations=key_relations,
            statistics={
                "node_count": len(cluster.nodes),
                "edge_count": len(cluster.edges),
                "density": self._calculate_density(cluster),
                **cluster.metadata,
            },
        )

    def _calculate_density(self, cluster: Cluster) -> float:
        """Calculate cluster density.

        Args:
            cluster: Cluster to calculate density for

        Returns:
            Density value (0-1)
        """
        n = len(cluster.nodes)
        if n <= 1:
            return 0.0

        # Maximum possible edges in undirected graph
        max_edges = n * (n - 1) / 2
        actual_edges = len(cluster.edges)

        return actual_edges / max_edges if max_edges > 0 else 0.0

    def summarize_all(
        self,
        clusters: list[Cluster],
        graph=None,
    ) -> list[ClusterSummary]:
        """Summarize all clusters.

        Args:
            clusters: List of clusters to summarize
            graph: Optional knowledge graph

        Returns:
            List of ClusterSummary objects
        """
        summaries = []
        for cluster in clusters:
            summary = self.summarize(cluster, graph)
            summaries.append(summary)

        return summaries


def create_summarizer(
    use_llm: bool = False,
    llm_model: str | None = None,
) -> ClusterSummarizer:
    """Factory function to create cluster summarizer.

    Args:
        use_llm: Whether to use LLM
        llm_model: LLM model name

    Returns:
        ClusterSummarizer instance
    """
    return ClusterSummarizer(use_llm=use_llm, llm_model=llm_model)
