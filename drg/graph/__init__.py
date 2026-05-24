"""
Knowledge Graph Module

This module provides:
- Schema generation (dataset-agnostic)
- Relationship modeling (enriched format)
- Knowledge graph core (modular monolith)
- Visualization (Mermaid, PyVis)
- Community reports
"""

# Legacy KG class (from graph.py)
import json
from typing import Any


class KG:
    """Simple Knowledge Graph class."""

    def __init__(self):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []

    @classmethod
    def from_typed(cls, entities_typed: list[tuple[str, str]], triples: list[tuple[str, str, str]]):
        kg = cls()
        for name, etype in entities_typed:
            kg.nodes.setdefault(name, {"type": etype})
        for s, r, o in triples:
            kg.nodes.setdefault(s, {"type": None})
            kg.nodes.setdefault(o, {"type": None})
            kg.edges.append((s, r, o))
        return kg

    @classmethod
    def from_triples(cls, triples: list[tuple[str, str, str]]):
        kg = cls()
        for s, r, o in triples:
            kg.nodes.setdefault(s, {"type": None})
            kg.nodes.setdefault(o, {"type": None})
            kg.edges.append((s, r, o))
        return kg

    def to_json(self, indent: int = 2) -> str:
        data = {
            "nodes": [{"id": n, **attr} for n, attr in self.nodes.items()],
            "edges": [{"source": s, "type": r, "target": o} for s, r, o in self.edges],
        }
        return json.dumps(data, indent=indent)


# Schema Generator
# Community Report
from .community_report import (
    CommunityReport,
    CommunityReportGenerator,
)

# Hub mitigation (export-time graph shaping)
from .hub_mitigation import apply_hub_relation_proxy_split

# KG Core
from .kg_core import (
    Cluster,
    EnhancedKG,
    KGEdge,
    KGNode,
)

# Neo4j Exporter
from .neo4j_exporter import (
    Neo4jConfig,
    Neo4jExporter,
)

# Relationship Model
from .relationship_model import (
    RELATIONSHIP_CATEGORIES,
    EnrichedRelationship,
    RelationshipType,
    RelationshipTypeClassifier,
    create_enriched_relationship,
)
from .schema_generator import (
    DatasetAgnosticSchemaGenerator,
    EntityClassDefinition,
    PropertyDefinition,
    create_default_schema,
)

# Visualization
from .visualization import (
    DEFAULT_EDGE_COLORS,
    DEFAULT_NODE_COLORS,
    KGVisualizer,
)

# Visualization Adapter
from .visualization_adapter import (
    ProvenanceEdge,
    ProvenanceGraph,
    ProvenanceNode,
    VisualizationAdapter,
)

__all__ = [
    "DEFAULT_EDGE_COLORS",
    "DEFAULT_NODE_COLORS",
    # Legacy KG
    "KG",
    "RELATIONSHIP_CATEGORIES",
    "Cluster",
    # Community Report
    "CommunityReport",
    "CommunityReportGenerator",
    "DatasetAgnosticSchemaGenerator",
    "EnhancedKG",
    "EnrichedRelationship",
    "EntityClassDefinition",
    "KGEdge",
    # KG Core
    "KGNode",
    # Visualization
    "KGVisualizer",
    # Neo4j Exporter
    "Neo4jConfig",
    "Neo4jExporter",
    # Schema Generator
    "PropertyDefinition",
    "ProvenanceEdge",
    "ProvenanceGraph",
    # Visualization Adapter
    "ProvenanceNode",
    # Relationship Model
    "RelationshipType",
    "RelationshipTypeClassifier",
    "VisualizationAdapter",
    # Hub mitigation
    "apply_hub_relation_proxy_split",
    "create_default_schema",
    "create_enriched_relationship",
]
