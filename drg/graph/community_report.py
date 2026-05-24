"""
Community Report Module

Provides cluster-based community reports with:
- Summary generation
- Top actors identification
- Top relationships extraction
- Theme identification

Algorithm-agnostic: accepts cluster assignments as input.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .kg_core import Cluster, EnhancedKG, KGEdge, KGNode


@dataclass
class CommunityReport:
    """
    Community/Cluster report with summary and themes.
    """

    cluster_id: str
    summary: str
    top_actors: list[tuple[str, int]]  # (actor_id, connection_count)
    top_relationships: list[tuple[str, int]]  # (relationship_type, count)
    themes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cluster_id": self.cluster_id,
            "summary": self.summary,
            "top_actors": [{"id": actor, "connections": count} for actor, count in self.top_actors],
            "top_relationships": [
                {"type": rel_type, "count": count} for rel_type, count in self.top_relationships
            ],
            "themes": self.themes,
            "metadata": self.metadata,
        }


class CommunityReportGenerator:
    """
    Generator for community/cluster reports.

    Algorithm-agnostic: accepts cluster assignments and generates reports.
    """

    def __init__(self, kg: EnhancedKG):
        self.kg = kg

    def generate_report(
        self,
        cluster: Cluster,
        max_actors: int = 10,
        max_relationships: int = 10,
        max_themes: int = 5,
    ) -> CommunityReport:
        """Generate a community report for a cluster."""
        cluster_nodes = {
            node_id: self.kg.get_node(node_id)
            for node_id in cluster.node_ids
            if self.kg.get_node(node_id)
        }

        cluster_edges = [
            edge
            for edge in self.kg.edges
            if edge.source in cluster.node_ids and edge.target in cluster.node_ids
        ]

        summary = self._generate_summary(cluster, cluster_nodes, cluster_edges)
        top_actors = self._identify_top_actors(cluster.node_ids, max_actors)
        top_relationships = self._extract_top_relationships(cluster_edges, max_relationships)
        themes = self._identify_themes(cluster_nodes, cluster_edges, max_themes)

        metadata = {
            "node_count": len(cluster.node_ids),
            "edge_count": len(cluster_edges),
            "density": self._calculate_density(len(cluster.node_ids), len(cluster_edges)),
        }
        metadata.update(cluster.metadata)

        return CommunityReport(
            cluster_id=cluster.id,
            summary=summary,
            top_actors=top_actors,
            top_relationships=top_relationships,
            themes=themes,
            metadata=metadata,
        )

    def _generate_summary(
        self,
        cluster: Cluster,
        cluster_nodes: dict[str, KGNode],
        cluster_edges: list[KGEdge],
    ) -> str:
        """Generate a summary for the cluster."""
        node_types = Counter(node.type or "Unknown" for node in cluster_nodes.values())
        most_common_type = node_types.most_common(1)[0][0] if node_types else "entities"

        relationship_types = Counter(edge.relationship_type for edge in cluster_edges)
        most_common_rel = (
            relationship_types.most_common(1)[0][0] if relationship_types else "relationships"
        )

        summary_parts = [
            f"Cluster {cluster.id} contains {len(cluster_nodes)} entities",
            f"primarily of type '{most_common_type}'",
            f"with {len(cluster_edges)} internal relationships.",
        ]

        if relationship_types:
            summary_parts.append(f"The most common relationship type is '{most_common_rel}'.")

        return " ".join(summary_parts)

    def _identify_top_actors(self, node_ids: set[str], max_actors: int) -> list[tuple[str, int]]:
        """Identify top actors (nodes with most connections)."""
        connection_counts = Counter()
        for edge in self.kg.edges:
            if edge.source in node_ids:
                connection_counts[edge.source] += 1
            if edge.target in node_ids:
                connection_counts[edge.target] += 1
        return connection_counts.most_common(max_actors)

    def _extract_top_relationships(
        self,
        cluster_edges: list[KGEdge],
        max_relationships: int,
    ) -> list[tuple[str, int]]:
        """Extract top relationship types in the cluster."""
        relationship_counts = Counter(edge.relationship_type for edge in cluster_edges)
        return relationship_counts.most_common(max_relationships)

    def _identify_themes(
        self,
        cluster_nodes: dict[str, KGNode],
        cluster_edges: list[KGEdge],
        max_themes: int,
    ) -> list[str]:
        """Identify themes in the cluster."""
        themes = []

        node_types = Counter(node.type or "Unknown" for node in cluster_nodes.values())
        if node_types:
            dominant_type = node_types.most_common(1)[0][0]
            if dominant_type != "Unknown":
                themes.append(f"{dominant_type}-centric")

        relationship_types = Counter(edge.relationship_type for edge in cluster_edges)
        if relationship_types:
            dominant_rel = relationship_types.most_common(1)[0][0]
            rel_theme_map = {
                "influences": "influence networks",
                "collaborates_with": "collaboration networks",
                "located_at": "spatial networks",
                "caused_by": "causal networks",
                "works_with": "work networks",
            }
            theme = rel_theme_map.get(dominant_rel, f"{dominant_rel} networks")
            themes.append(theme)

        density = self._calculate_density(len(cluster_nodes), len(cluster_edges))
        if density > 0.5:
            themes.append("highly connected")
        elif density < 0.2:
            themes.append("loosely connected")

        return themes[:max_themes]

    def _calculate_density(self, node_count: int, edge_count: int) -> float:
        """Calculate graph density."""
        if node_count < 2:
            return 0.0
        possible_edges = node_count * (node_count - 1)
        return edge_count / possible_edges if possible_edges > 0 else 0.0

    def generate_all_reports(
        self, max_actors: int = 10, max_relationships: int = 10, max_themes: int = 5
    ) -> list[CommunityReport]:
        """Generate reports for all clusters in the KG."""
        return [
            self.generate_report(cluster, max_actors, max_relationships, max_themes)
            for cluster in self.kg.clusters.values()
        ]

    def export_reports_json(self, reports: list[CommunityReport], filepath: str) -> None:
        """Export reports to JSON file."""
        import json
        from pathlib import Path

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "reports": [report.to_dict() for report in reports],
            "total_clusters": len(reports),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def generate_report_text(self, report: CommunityReport) -> str:
        """Generate human-readable text report."""
        lines = [
            "=" * 70,
            f"Community Report: {report.cluster_id}",
            "=" * 70,
            "",
            "Summary:",
            f"  {report.summary}",
            "",
            "Top Actors:",
        ]
        for actor, count in report.top_actors:
            lines.append(f"  - {actor}: {count} connections")
        lines.extend(
            [
                "",
                "Top Relationships:",
            ]
        )
        for rel_type, count in report.top_relationships:
            lines.append(f"  - {rel_type}: {count} occurrences")
        lines.extend(
            [
                "",
                "Themes:",
            ]
        )
        for theme in report.themes:
            lines.append(f"  - {theme}")
        if report.metadata:
            lines.extend(["", "Metadata:"])
            for key, value in report.metadata.items():
                lines.append(f"  - {key}: {value}")
        lines.append("=" * 70)
        return "\n".join(lines)
