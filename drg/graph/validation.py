"""Validation helpers for persisted :class:`EnhancedKG` JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    """A single graph validation finding."""

    severity: str
    code: str
    message: str
    path: str = "$"

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


@dataclass
class ValidationReport:
    """Machine-readable validation report for CLI and CI usage."""

    path: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "info")

    @property
    def valid(self) -> bool:
        return self.error_count == 0

    def add(self, severity: str, code: str, message: str, path: str = "$") -> None:
        self.issues.append(ValidationIssue(severity, code, message, path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "valid": self.valid,
            "summary": {
                "errors": self.error_count,
                "warnings": self.warning_count,
                "info": self.info_count,
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_graph_json(path: str | Path) -> dict[str, Any]:
    """Load a graph JSON document and require a top-level object."""

    graph_path = Path(path)
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Knowledge graph JSON must be an object.")
    return data


def validate_graph_data(data: dict[str, Any], path: str = "<memory>") -> ValidationReport:
    """Validate the persisted EnhancedKG JSON shape and references."""

    report = ValidationReport(path=path)

    nodes = data.get("nodes")
    edges = data.get("edges")
    clusters = data.get("clusters", [])

    if not isinstance(nodes, list):
        report.add("error", "nodes_not_list", "`nodes` must be a list.", "$.nodes")
        nodes = []
    if not isinstance(edges, list):
        report.add("error", "edges_not_list", "`edges` must be a list.", "$.edges")
        edges = []
    if not isinstance(clusters, list):
        report.add("error", "clusters_not_list", "`clusters` must be a list.", "$.clusters")
        clusters = []

    node_ids: set[str] = set()
    for idx, node in enumerate(nodes):
        node_path = f"$.nodes[{idx}]"
        if not isinstance(node, dict):
            report.add("error", "node_not_object", "Node entry must be an object.", node_path)
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            report.add(
                "error", "node_missing_id", "Node must have a non-empty string `id`.", node_path
            )
            continue
        if node_id in node_ids:
            report.add("error", "duplicate_node_id", f"Duplicate node id `{node_id}`.", node_path)
        node_ids.add(node_id)
        if (
            "type" in node
            and node.get("type") is not None
            and not isinstance(node.get("type"), str)
        ):
            report.add(
                "warning", "node_type_not_string", "`type` should be a string or null.", node_path
            )

    seen_edges: set[tuple[str, str, str]] = set()
    for idx, edge in enumerate(edges):
        edge_path = f"$.edges[{idx}]"
        if not isinstance(edge, dict):
            report.add("error", "edge_not_object", "Edge entry must be an object.", edge_path)
            continue

        source = edge.get("source")
        target = edge.get("target")
        rel_type = edge.get("relationship_type")

        if not isinstance(source, str) or not source.strip():
            report.add(
                "error", "edge_missing_source", "Edge must have a non-empty `source`.", edge_path
            )
        elif source not in node_ids:
            report.add(
                "error",
                "edge_source_missing_node",
                f"Edge source `{source}` is not present in nodes.",
                f"{edge_path}.source",
            )

        if not isinstance(target, str) or not target.strip():
            report.add(
                "error", "edge_missing_target", "Edge must have a non-empty `target`.", edge_path
            )
        elif target not in node_ids:
            report.add(
                "error",
                "edge_target_missing_node",
                f"Edge target `{target}` is not present in nodes.",
                f"{edge_path}.target",
            )

        if not isinstance(rel_type, str) or not rel_type.strip():
            report.add(
                "error",
                "edge_missing_relationship_type",
                "Edge must have a non-empty `relationship_type`.",
                f"{edge_path}.relationship_type",
            )

        if isinstance(source, str) and isinstance(target, str) and isinstance(rel_type, str):
            key = (source, rel_type, target)
            if key in seen_edges:
                report.add(
                    "warning",
                    "duplicate_edge",
                    f"Duplicate edge `{source}` -[{rel_type}]-> `{target}`.",
                    edge_path,
                )
            seen_edges.add(key)

    seen_clusters: set[str] = set()
    for idx, cluster in enumerate(clusters):
        cluster_path = f"$.clusters[{idx}]"
        if not isinstance(cluster, dict):
            report.add(
                "error", "cluster_not_object", "Cluster entry must be an object.", cluster_path
            )
            continue
        cluster_id = cluster.get("id")
        if not isinstance(cluster_id, str) or not cluster_id.strip():
            report.add(
                "error", "cluster_missing_id", "Cluster must have a non-empty `id`.", cluster_path
            )
        elif cluster_id in seen_clusters:
            report.add(
                "warning",
                "duplicate_cluster_id",
                f"Duplicate cluster id `{cluster_id}`.",
                cluster_path,
            )
        if isinstance(cluster_id, str):
            seen_clusters.add(cluster_id)

        members = cluster.get("node_ids")
        if not isinstance(members, list):
            report.add(
                "error",
                "cluster_members_not_list",
                "Cluster `node_ids` must be a list.",
                f"{cluster_path}.node_ids",
            )
            continue
        for member in members:
            if member not in node_ids:
                report.add(
                    "error",
                    "cluster_member_missing_node",
                    f"Cluster member `{member}` is not present in nodes.",
                    f"{cluster_path}.node_ids",
                )

    if report.valid:
        report.add("info", "graph_valid", "Knowledge graph passed validation.")
    return report


def validate_graph_file(path: str | Path) -> ValidationReport:
    """Load and validate a graph JSON file."""

    return validate_graph_data(load_graph_json(path), path=str(path))
