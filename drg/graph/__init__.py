"""
Knowledge Graph Module

This module provides:
- Schema generation (dataset-agnostic)
- Relationship modeling (enriched format)
- Knowledge graph core (modular monolith)
- Visualization (Mermaid, PyVis)
- Community reports
"""

# Legacy KG class — moved to _legacy.py, kept here for backward compatibility.
# Instantiating KG will raise DeprecationWarning; use EnhancedKG instead.
from ._legacy import KG


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
