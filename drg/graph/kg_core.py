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
        metadata: Optional metadata (source_ref, etc.)
        embedding: Optional embedding vector for semantic similarity
        confidence: Optional extraction-time confidence score in [0.0, 1.0].
            ``None`` means "no confidence has been computed for this node"
            (the legacy default), preserving backward compatibility for
            callers/tests that don't yet wire a confidence strategy.
            See :mod:`drg.confidence` for the scoring framework.
    """

    id: str
    type: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    confidence: float | None = None

    def __post_init__(self):
        """Validate node data."""
        if not self.id:
            raise ValueError("Node id cannot be empty")
        # Mirror the validation already enforced on KGEdge.confidence so
        # node and edge confidence have identical semantics.
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")

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
        # Only emit confidence when explicitly set — keeps the legacy JSON
        # surface stable for KGs built without a confidence strategy.
        if self.confidence is not None:
            result["confidence"] = self.confidence
        temporal = self.get_temporal_scope()
        if temporal is not None:
            result["temporal"] = temporal.to_dict()
        return result

    def get_temporal_scope(self):
        """Return :class:`drg.temporal.TemporalScope` from node metadata."""
        from ..temporal import TemporalScope

        meta = self.metadata or {}
        nested = meta.get("temporal")
        if isinstance(nested, dict):
            return TemporalScope.from_dict(nested)
        return None

    def apply_temporal_scope(self, scope) -> None:
        """Store temporal validity on this node (in ``metadata['temporal']``)."""
        if scope is None or scope.is_empty():
            return
        meta = dict(self.metadata)
        meta["temporal"] = scope.to_dict()
        self.metadata = meta

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KGNode":
        """Create from dictionary representation."""
        metadata = dict(data.get("metadata", {}))
        temporal = data.get("temporal")
        if isinstance(temporal, dict) and "temporal" not in metadata:
            metadata["temporal"] = temporal
        return cls(
            id=data["id"],
            type=data.get("type"),
            properties=data.get("properties", {}),
            metadata=metadata,
            embedding=data.get("embedding"),
            confidence=data.get("confidence"),
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
    # Temporal information (``start_time``/``end_time`` are legacy names;
    # ``valid_from``/``valid_to`` are semantic aliases — see properties below)
    start_time: str | None = None
    end_time: str | None = None
    created_at: str | None = None  # extraction / ingestion timestamp (ISO instant)
    updated_at: str | None = None  # last merge or update timestamp (ISO instant)
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

    @property
    def valid_from(self) -> str | None:
        return self.start_time

    @valid_from.setter
    def valid_from(self, value: str | None) -> None:
        self.start_time = value

    @property
    def valid_to(self) -> str | None:
        return self.end_time

    @valid_to.setter
    def valid_to(self, value: str | None) -> None:
        self.end_time = value

    def get_temporal_scope(self):
        """Return :class:`drg.temporal.TemporalScope` for this edge."""
        from ..temporal import temporal_from_edge_fields

        return temporal_from_edge_fields(
            start_time=self.start_time,
            end_time=self.end_time,
            created_at=self.created_at,
            updated_at=self.updated_at,
            metadata=self.metadata,
        )

    def apply_temporal_scope(self, scope) -> None:
        """Apply a :class:`drg.temporal.TemporalScope` onto this edge."""
        if scope is None:
            return
        self.start_time = scope.valid_from
        self.end_time = scope.valid_to
        if scope.created_at is not None:
            self.created_at = scope.created_at
        if scope.updated_at is not None:
            self.updated_at = scope.updated_at
        if not scope.is_empty():
            meta = dict(self.metadata)
            meta["temporal"] = scope.to_dict()
            self.metadata = meta

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
            result["valid_from"] = self.start_time
        if self.end_time:
            result["end_time"] = self.end_time
            result["valid_to"] = self.end_time
        if self.created_at:
            result["created_at"] = self.created_at
        if self.updated_at:
            result["updated_at"] = self.updated_at
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
            start_time=data.get("start_time") or data.get("valid_from"),
            end_time=data.get("end_time") or data.get("valid_to"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
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
                start_time = temporal.get("valid_from") or temporal.get("start")
                end_time = temporal.get("valid_to") or temporal.get("end")
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
    - Optional graph-level metadata (version, history) for incremental updates

    The ``metadata`` field is intentionally a free-form dict so callers that
    don't opt into incremental ingestion (the legacy default) keep an empty
    ``{}`` and the JSON output stays byte-compatible with previous versions.
    The incremental update layer (``drg.graph.incremental``) populates
    ``version`` / ``history`` keys when it touches the graph.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, KGNode] = {}
        self.edges: list[KGEdge] = []
        self.clusters: dict[str, Cluster] = {}
        self.metadata: dict[str, Any] = {}

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

    def canonicalize_entities(
        self,
        name_mapping: dict[str, str],
        *,
        decisions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Merge alias nodes into canonical nodes and rewrite edge endpoints.

        ``name_mapping`` should map every observed mention to its canonical id,
        as returned by :class:`drg.entity_resolution.EntityResolver`. The method
        records aliases in node metadata so query/search surfaces can still find
        merged mentions.
        """
        if not name_mapping:
            return {"merged_nodes": 0, "removed_edges": 0, "aliases": {}}

        aliases_by_canonical: dict[str, set[str]] = {}
        for original, canonical in name_mapping.items():
            if original == canonical:
                continue
            aliases_by_canonical.setdefault(canonical, set()).add(original)

        merged_nodes = 0
        for canonical, aliases in aliases_by_canonical.items():
            canonical_node = self.nodes.get(canonical)
            if canonical_node is None:
                continue

            meta = dict(canonical_node.metadata)
            existing_aliases = set(meta.get("aliases", []) or [])
            existing_aliases.update(aliases)
            meta["aliases"] = sorted(existing_aliases, key=str.lower)
            er_meta = dict(meta.get("entity_resolution", {}) or {})
            er_meta["canonical"] = canonical
            er_meta["aliases"] = meta["aliases"]
            if decisions:
                er_meta["decisions"] = [
                    d
                    for d in decisions
                    if d.get("canonical") == canonical or d.get("original") == canonical
                ]
            meta["entity_resolution"] = er_meta

            for alias in sorted(aliases, key=str.lower):
                alias_node = self.nodes.get(alias)
                if alias_node is None:
                    continue
                for key, value in alias_node.properties.items():
                    canonical_node.properties.setdefault(key, value)
                alias_meta = dict(alias_node.metadata)
                alias_meta.pop("aliases", None)
                for key, value in alias_meta.items():
                    meta.setdefault(key, value)
                del self.nodes[alias]
                merged_nodes += 1

            canonical_node.metadata = meta

        rewritten_edges: list[KGEdge] = []
        seen_edges: set[tuple[Any, ...]] = set()
        removed_edges = 0
        for edge in self.edges:
            edge.source = name_mapping.get(edge.source, edge.source)
            edge.target = name_mapping.get(edge.target, edge.target)
            if edge.source == edge.target:
                removed_edges += 1
                continue
            key = (
                edge.source,
                edge.relationship_type,
                edge.target,
                edge.relationship_detail,
                edge.start_time,
                edge.end_time,
                edge.created_at,
                edge.updated_at,
            )
            if key in seen_edges:
                removed_edges += 1
                continue
            seen_edges.add(key)
            rewritten_edges.append(edge)
        self.edges = rewritten_edges

        for cluster in self.clusters.values():
            rewritten = {name_mapping.get(node_id, node_id) for node_id in cluster.node_ids}
            cluster.node_ids = {node_id for node_id in rewritten if node_id in self.nodes}

        er_summary = dict(self.metadata.get("entity_resolution", {}) or {})
        er_summary["aliases"] = {
            canonical: sorted(aliases, key=str.lower)
            for canonical, aliases in aliases_by_canonical.items()
            if canonical in self.nodes
        }
        er_summary["merged_nodes"] = int(er_summary.get("merged_nodes", 0)) + merged_nodes
        self.metadata["entity_resolution"] = er_summary

        return {
            "merged_nodes": merged_nodes,
            "removed_edges": removed_edges,
            "aliases": er_summary["aliases"],
        }

    def add_cluster(self, cluster: Cluster) -> None:
        """Add a cluster to the graph."""
        missing_nodes = cluster.node_ids - set(self.nodes.keys())
        if missing_nodes:
            raise ValueError(f"Cluster contains non-existent nodes: {missing_nodes}")
        self.clusters[cluster.id] = cluster

    def _events_payload(self) -> list[dict[str, Any]]:
        """Surface event nodes as a separate list (lazy import to avoid cycle).

        Returns an empty list when no event-typed nodes exist, which keeps
        the legacy serialization byte-compatible (callers gate on truthiness).
        """
        from ..events._graph_mapping import event_from_kg_node, is_event_node

        out: list[dict[str, Any]] = []
        for node in self.nodes.values():
            if not is_event_node(node):
                continue
            event = event_from_kg_node(node)
            if event is not None:
                out.append(event.to_dict())
        return out

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON format.

        ``metadata`` is included only when populated so the legacy three-key
        shape ({"nodes", "edges", "clusters"}) is preserved for graphs built
        without the incremental layer. The optional ``events`` key surfaces
        event-typed nodes as their canonical event payload; it is only
        emitted when the graph actually contains events, so legacy graphs
        round-trip byte-for-byte.
        """
        data: dict[str, Any] = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "clusters": [cluster.to_dict() for cluster in self.clusters.values()],
        }
        events_payload = self._events_payload()
        if events_payload:
            data["events"] = events_payload
        if self.metadata:
            data["metadata"] = self.metadata
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
                # Confidence is a first-class JSON-LD property (not a
                # ``kg:meta/`` field) so consumers can index/filter on it
                # without parsing the metadata bag.
                **({"confidence": node.confidence} if node.confidence is not None else {}),
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

        data: dict[str, Any] = {
            **context,
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
        }
        events_payload = self._events_payload()
        if events_payload:
            data["events"] = [
                {
                    "@id": f"kg:event/{ev['id']}",
                    "@type": "Event",
                    "identifier": ev["id"],
                    "kg:event_type": ev["event_type"],
                    **(
                        {"startDate": ev["timestamp"].get("start")}
                        if ev.get("timestamp") and ev["timestamp"].get("start")
                        else {}
                    ),
                    **(
                        {"endDate": ev["timestamp"].get("end")}
                        if ev.get("timestamp") and ev["timestamp"].get("end")
                        else {}
                    ),
                    **(
                        {"location": {"@id": f"kg:node/{ev['location']}"}}
                        if ev.get("location")
                        else {}
                    ),
                    "kg:participants": ev.get("participants", {}),
                    **{f"kg:prop/{k}": v for k, v in (ev.get("properties") or {}).items()},
                    "kg:provenance": ev.get("provenance", {}),
                }
                for ev in events_payload
            ]
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

        data: dict[str, Any] = {
            "entities": nodes,
            "relationships": edges,
            "communities": clusters if clusters else None,
        }
        events_payload = self._events_payload()
        if events_payload:
            data["events"] = events_payload

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnhancedKG":
        """Reconstruct an :class:`EnhancedKG` from its JSON-serialisable form.

        Accepts both legacy graphs (no ``metadata`` key) and graphs produced
        by the incremental update layer (``metadata.version`` / ``history``).
        Cluster references are validated against the loaded node set; orphan
        cluster members are silently dropped rather than raising so that
        partially-corrupted graphs can still be loaded for inspection.
        """
        kg = cls()

        for node_data in data.get("nodes", []) or []:
            node = KGNode.from_dict(node_data)
            kg.nodes[node.id] = node

        for edge_data in data.get("edges", []) or []:
            edge = KGEdge.from_dict(edge_data)
            # Use list.append directly: round-tripping a stored graph should
            # not re-validate referential integrity (the file itself is the
            # source of truth) and self-loop edges produced by older
            # versions should still load.
            kg.edges.append(edge)

        for cluster_data in data.get("clusters", []) or []:
            cluster = Cluster.from_dict(cluster_data)
            valid_members = {nid for nid in cluster.node_ids if nid in kg.nodes}
            if valid_members:
                cluster.node_ids = valid_members
                kg.clusters[cluster.id] = cluster

        meta = data.get("metadata")
        if isinstance(meta, dict):
            kg.metadata = dict(meta)

        return kg

    @classmethod
    def load_json(cls, filepath: str) -> "EnhancedKG":
        """Load an :class:`EnhancedKG` from a JSON file written by ``save_json``.

        Counterpart to :meth:`save_json`. Used by the incremental ingestion
        layer to re-hydrate a previously-persisted graph before merging new
        document extractions into it.
        """
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        return cls.from_dict(data)

    def query(self):
        """Return a :class:`drg.query.GraphQuery` facade over this graph.

        Convenience only — graph generation behaviour is unchanged.
        """
        from ..query import GraphQuery

        return GraphQuery(self)

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
