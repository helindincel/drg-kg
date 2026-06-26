"""Structural diff helpers for persisted EnhancedKG snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _node_key(node: dict[str, Any]) -> str:
    return str(node.get("id", ""))


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("source", "")),
        str(edge.get("relationship_type", "")),
        str(edge.get("target", "")),
    )


def _cluster_key(cluster: dict[str, Any]) -> str:
    return str(cluster.get("id", ""))


def _metadata_value(item: dict[str, Any], key: str) -> Any:
    metadata = item.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata.get(key)
    return item.get(key)


def _provenance_value(item: dict[str, Any]) -> Any:
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get("provenance") or {
            key: metadata.get(key)
            for key in ("source_ref", "source_documents")
            if metadata.get(key) is not None
        }
    return None


def _semantic_changes(
    old: dict[str, Any], new: dict[str, Any], fields: tuple[str, ...]
) -> list[str]:
    changed: list[str] = []
    for field_name in fields:
        if field_name == "provenance":
            old_value = _provenance_value(old)
            new_value = _provenance_value(new)
        elif field_name == "evidence":
            old_value = _metadata_value(old, "evidence")
            new_value = _metadata_value(new, "evidence")
        else:
            old_value = old.get(field_name)
            new_value = new.get(field_name)
        if old_value != new_value:
            changed.append(field_name)
    return changed


@dataclass
class SnapshotDiff:
    """Diff between two EnhancedKG JSON snapshots."""

    added_nodes: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    changed_nodes: list[str] = field(default_factory=list)
    added_edges: list[tuple[str, str, str]] = field(default_factory=list)
    removed_edges: list[tuple[str, str, str]] = field(default_factory=list)
    changed_edges: list[tuple[str, str, str]] = field(default_factory=list)
    added_clusters: list[str] = field(default_factory=list)
    removed_clusters: list[str] = field(default_factory=list)
    changed_clusters: list[str] = field(default_factory=list)
    node_semantic_changes: list[dict[str, Any]] = field(default_factory=list)
    edge_semantic_changes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(
            (
                self.added_nodes,
                self.removed_nodes,
                self.changed_nodes,
                self.added_edges,
                self.removed_edges,
                self.changed_edges,
                self.added_clusters,
                self.removed_clusters,
                self.changed_clusters,
                self.node_semantic_changes,
                self.edge_semantic_changes,
            )
        )

    def summary(self) -> dict[str, int]:
        return {
            "added_nodes": len(self.added_nodes),
            "removed_nodes": len(self.removed_nodes),
            "changed_nodes": len(self.changed_nodes),
            "added_edges": len(self.added_edges),
            "removed_edges": len(self.removed_edges),
            "changed_edges": len(self.changed_edges),
            "added_clusters": len(self.added_clusters),
            "removed_clusters": len(self.removed_clusters),
            "changed_clusters": len(self.changed_clusters),
            "node_semantic_changes": len(self.node_semantic_changes),
            "edge_semantic_changes": len(self.edge_semantic_changes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "summary": self.summary(),
            "added_nodes": self.added_nodes,
            "removed_nodes": self.removed_nodes,
            "changed_nodes": self.changed_nodes,
            "added_edges": [list(edge) for edge in self.added_edges],
            "removed_edges": [list(edge) for edge in self.removed_edges],
            "changed_edges": [list(edge) for edge in self.changed_edges],
            "added_clusters": self.added_clusters,
            "removed_clusters": self.removed_clusters,
            "changed_clusters": self.changed_clusters,
            "node_semantic_changes": self.node_semantic_changes,
            "edge_semantic_changes": [
                {
                    **change,
                    "edge": list(change["edge"])
                    if isinstance(change.get("edge"), tuple)
                    else change.get("edge"),
                }
                for change in self.edge_semantic_changes
            ],
        }


def diff_graph_data(old: dict[str, Any], new: dict[str, Any]) -> SnapshotDiff:
    """Return a deterministic structural diff between two graph JSON objects."""

    old_nodes = {_node_key(node): node for node in old.get("nodes", []) if isinstance(node, dict)}
    new_nodes = {_node_key(node): node for node in new.get("nodes", []) if isinstance(node, dict)}

    old_edges = {_edge_key(edge): edge for edge in old.get("edges", []) if isinstance(edge, dict)}
    new_edges = {_edge_key(edge): edge for edge in new.get("edges", []) if isinstance(edge, dict)}

    old_clusters = {
        _cluster_key(cluster): cluster
        for cluster in old.get("clusters", [])
        if isinstance(cluster, dict)
    }
    new_clusters = {
        _cluster_key(cluster): cluster
        for cluster in new.get("clusters", [])
        if isinstance(cluster, dict)
    }

    node_semantic_changes = [
        {"node": node_id, "fields": fields}
        for node_id in sorted(set(old_nodes) & set(new_nodes))
        if (
            fields := _semantic_changes(
                old_nodes[node_id],
                new_nodes[node_id],
                ("type", "confidence", "provenance", "evidence"),
            )
        )
    ]
    edge_semantic_changes = [
        {"edge": edge_key, "fields": fields}
        for edge_key in sorted(set(old_edges) & set(new_edges))
        if (
            fields := _semantic_changes(
                old_edges[edge_key],
                new_edges[edge_key],
                (
                    "relationship_type",
                    "relationship_detail",
                    "confidence",
                    "provenance",
                    "evidence",
                ),
            )
        )
    ]

    return SnapshotDiff(
        added_nodes=sorted(set(new_nodes) - set(old_nodes)),
        removed_nodes=sorted(set(old_nodes) - set(new_nodes)),
        changed_nodes=sorted(
            node_id
            for node_id in set(old_nodes) & set(new_nodes)
            if old_nodes[node_id] != new_nodes[node_id]
        ),
        added_edges=sorted(set(new_edges) - set(old_edges)),
        removed_edges=sorted(set(old_edges) - set(new_edges)),
        changed_edges=sorted(
            edge_key
            for edge_key in set(old_edges) & set(new_edges)
            if old_edges[edge_key] != new_edges[edge_key]
        ),
        added_clusters=sorted(set(new_clusters) - set(old_clusters)),
        removed_clusters=sorted(set(old_clusters) - set(new_clusters)),
        changed_clusters=sorted(
            cluster_id
            for cluster_id in set(old_clusters) & set(new_clusters)
            if old_clusters[cluster_id] != new_clusters[cluster_id]
        ),
        node_semantic_changes=node_semantic_changes,
        edge_semantic_changes=edge_semantic_changes,
    )
