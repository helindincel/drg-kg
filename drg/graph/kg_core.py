"""
Knowledge Graph Core - Modular Monolith Architecture

This module provides a comprehensive KG core with:
- Entities (nodes) with type, properties, metadata
- Relationships (edges) with enriched relationship model
- Clusters/Communities support (algorithm-agnostic)
- Multiple export formats (JSON, JSON-LD, enriched format)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .relationship_model import EnrichedRelationship


@dataclass
class KGNode:
    """
    Knowledge Graph Node (Entity).

    Attributes:
        id: Unique identifier for the node
        type: Entity type (e.g., "Person", "Location", "Event")
        properties: Optional dictionary of entity properties
        metadata: Optional metadata (confidence, source_ref, etc.)
        embedding: Optional embedding vector for semantic similarity
    """

    id: str
    type: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def __post_init__(self):
        """Validate node data."""
        if not self.id:
            raise ValueError("Node id cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "type": self.type,
        }
        if self.properties:
            result["properties"] = self.properties
        if self.metadata:
            result["metadata"] = self.metadata
        if self.embedding:
            result["embedding"] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KGNode":
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            type=data.get("type"),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


@dataclass
class KGEdge:
    """
    Knowledge Graph Edge (Relationship).

    Uses EnrichedRelationship structure.
    Supports temporal information and confidence scores.
    """

    source: str
    target: str
    relationship_type: str
    relationship_detail: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Temporal information
    start_time: str | None = None  # ISO format date/time when relationship started
    end_time: str | None = None  # ISO format date/time when relationship ended
    # Confidence score (0.0-1.0)
    confidence: float | None = None  # Confidence score from extraction (0.0-1.0)
    # Negation flag
    is_negated: bool = False  # Whether the relationship is negated (e.g., "no longer produces")

    def __post_init__(self):
        """Validate edge data."""
        if not self.source or not self.target:
            raise ValueError("Edge source and target cannot be empty")
        if not self.relationship_type or not self.relationship_detail:
            raise ValueError("Edge relationship_type and detail cannot be empty")
        if self.source == self.target:
            raise ValueError("Edge source and target cannot be the same")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type,
            "relationship_detail": self.relationship_detail,
            "metadata": self.metadata,
        }
        if self.start_time:
            result["start_time"] = self.start_time
        if self.end_time:
            result["end_time"] = self.end_time
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.is_negated:
            result["is_negated"] = True
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KGEdge":
        """Create from dictionary representation."""
        return cls(
            source=data["source"],
            target=data["target"],
            relationship_type=data["relationship_type"],
            relationship_detail=data["relationship_detail"],
            metadata=data.get("metadata", {}),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            confidence=data.get("confidence"),
            is_negated=data.get("is_negated", False),
        )

    @classmethod
    def from_enriched_relationship(cls, rel: EnrichedRelationship) -> "KGEdge":
        """Create KGEdge from EnrichedRelationship.

        Preserves temporal information, confidence, and negation from enriched relationship.
        Domain-agnostic: Works for any domain (technology, business, science, medicine, etc.).
        """
        metadata = {}
        if rel.confidence is not None:
            metadata["confidence"] = rel.confidence
        if rel.source_ref:
            metadata["source_ref"] = rel.source_ref

        # Extract temporal information if available in metadata
        start_time = None
        end_time = None
        if hasattr(rel, "metadata") and isinstance(rel.metadata, dict):
            temporal = rel.metadata.get("temporal")
            if isinstance(temporal, dict):
                start_time = temporal.get("start")
                end_time = temporal.get("end")
            # Also check direct fields (backward compatibility)
            if not start_time:
                start_time = rel.metadata.get("start_time")
            if not end_time:
                end_time = rel.metadata.get("end_time")

        return cls(
            source=rel.source,
            target=rel.target,
            relationship_type=rel.relationship_type.value,
            relationship_detail=rel.relationship_detail,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
            confidence=rel.confidence,
            is_negated=getattr(rel, "is_negated", False),
        )


@dataclass
class Cluster:
    """
    Cluster/Community representation (algorithm-agnostic).

    Clusters are identified externally and passed to the KG.
    """

    id: str
    node_ids: set[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate cluster data."""
        if not self.id or not self.node_ids:
            raise ValueError("Cluster id and node_ids cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "node_ids": list(self.node_ids),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Cluster":
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            node_ids=set(data["node_ids"]),
            metadata=data.get("metadata", {}),
        )


class EnhancedKG:
    """
    Enhanced Knowledge Graph with full support for:
    - Entities (nodes) with properties and metadata
    - Relationships (edges) with enriched details
    - Clusters/Communities
    - Multiple export formats
    """

    def __init__(self):
        self.nodes: dict[str, KGNode] = {}
        self.edges: list[KGEdge] = []
        self.clusters: dict[str, Cluster] = {}

    def add_node(self, node: KGNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> KGNode | None:
        """Get a node by id."""
        return self.nodes.get(node_id)

    def add_edge(self, edge: KGEdge) -> None:
        """Add an edge to the graph."""
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Source and target nodes must exist before adding edge")
        self.edges.append(edge)

    def add_cluster(self, cluster: Cluster) -> None:
        """Add a cluster to the graph."""
        missing_nodes = cluster.node_ids - set(self.nodes.keys())
        if missing_nodes:
            raise ValueError(f"Cluster contains non-existent nodes: {missing_nodes}")
        self.clusters[cluster.id] = cluster

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON format."""
        data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "clusters": [cluster.to_dict() for cluster in self.clusters.values()],
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def to_json_ld(self, indent: int = 2) -> str:
        """Export to JSON-LD format."""
        context = {
            "@context": {
                "@vocab": "https://schema.org/",
                "kg": "https://example.org/kg/",
            }
        }

        nodes = [
            {
                "@id": f"kg:node/{node.id}",
                "@type": node.type or "Thing",
                "identifier": node.id,
                **{f"kg:prop/{k}": v for k, v in node.properties.items()},
                **{f"kg:meta/{k}": v for k, v in node.metadata.items()},
            }
            for node in self.nodes.values()
        ]

        edges = [
            {
                "@id": f"kg:edge/{edge.source}-{edge.target}",
                "@type": "Relationship",
                "source": {"@id": f"kg:node/{edge.source}"},
                "target": {"@id": f"kg:node/{edge.target}"},
                "relationship_type": edge.relationship_type,
                "relationship_detail": edge.relationship_detail,
                **({"start_time": edge.start_time} if edge.start_time else {}),
                **({"end_time": edge.end_time} if edge.end_time else {}),
                **({"confidence": edge.confidence} if edge.confidence is not None else {}),
                **({"is_negated": True} if edge.is_negated else {}),
                **{f"kg:meta/{k}": v for k, v in edge.metadata.items()},
            }
            for edge in self.edges
        ]

        clusters = [
            {
                "@id": f"kg:cluster/{cluster.id}",
                "@type": "Cluster",
                "identifier": cluster.id,
                "members": [{"@id": f"kg:node/{node_id}"} for node_id in cluster.node_ids],
                **{f"kg:meta/{k}": v for k, v in cluster.metadata.items()},
            }
            for cluster in self.clusters.values()
        ]

        data = {**context, "nodes": nodes, "edges": edges, "clusters": clusters}
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def to_enriched_format(self, indent: int = 2) -> str:
        """Export to enriched format (entities, relationships, communities)."""
        nodes = [node.to_dict() for node in self.nodes.values()]

        edges = []
        for edge in self.edges:
            edge_dict = {
                "source": edge.source,
                "target": edge.target,
                "relationship_type": edge.relationship_type,
                "relationship_detail": edge.relationship_detail,
            }
            if edge.start_time:
                edge_dict["start_time"] = edge.start_time
            if edge.end_time:
                edge_dict["end_time"] = edge.end_time
            if edge.confidence is not None:
                edge_dict["confidence"] = edge.confidence
            elif "confidence" in edge.metadata:
                # Backward compatibility: confidence stored in metadata
                edge_dict["confidence"] = edge.metadata["confidence"]
            if edge.is_negated:
                edge_dict["is_negated"] = True
            if "source_ref" in edge.metadata:
                edge_dict["source_ref"] = edge.metadata["source_ref"]
            edges.append(edge_dict)

        clusters = [cluster.to_dict() for cluster in self.clusters.values()]

        data = {
            "entities": nodes,
            "relationships": edges,
            "communities": clusters if clusters else None,
        }

        return json.dumps(data, indent=indent, ensure_ascii=False)

    def save_json(self, filepath: str, indent: int = 2) -> None:
        """Save to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")

    def save_json_ld(self, filepath: str, indent: int = 2) -> None:
        """Save to JSON-LD file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json_ld(indent=indent), encoding="utf-8")

    def save_enriched_format(self, filepath: str, indent: int = 2) -> None:
        """Save to enriched format file (entities, relationships, communities)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_enriched_format(indent=indent), encoding="utf-8")

    @classmethod
    def from_enriched_relationships(
        cls,
        nodes: list[KGNode],
        relationships: list[EnrichedRelationship],
    ) -> "EnhancedKG":
        """Create EnhancedKG from KGNode list and EnrichedRelationship list."""
        kg = cls()
        for node in nodes:
            kg.add_node(node)
        for rel in relationships:
            edge = KGEdge.from_enriched_relationship(rel)
            kg.add_edge(edge)
        return kg

    def add_entity_embeddings(
        self,
        embedding_provider,
        entity_texts: dict[str, str] | None = None,
    ) -> None:
        """Add embeddings to all nodes in the KG.

        Args:
            embedding_provider: Embedding provider to use
            entity_texts: Optional dict mapping entity_id to text representation.
                         If None, uses entity_id as text.
        """
        if entity_texts is None:
            entity_texts = {node_id: node_id for node_id in self.nodes.keys()}

        # Get all entity texts
        entity_ids = list(self.nodes.keys())
        texts = [entity_texts.get(eid, eid) for eid in entity_ids]

        # Batch embed
        embeddings = embedding_provider.embed_batch(texts)

        # Add embeddings to nodes
        for node_id, embedding in zip(entity_ids, embeddings, strict=False):
            if node_id in self.nodes:
                self.nodes[node_id].embedding = embedding
