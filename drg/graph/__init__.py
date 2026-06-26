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

# Graph snapshot diff
from .diff import (
    SnapshotDiff,
    diff_graph_data,
)

# Hub mitigation (export-time graph shaping)
from .hub_mitigation import apply_hub_relation_proxy_split

# Incremental updates (opt-in; safe to import without DSPy)
from .incremental import (
    EdgeMergePolicy,
    GraphMerger,
    KGDiff,
    MergeStrategy,
    NodeMergePolicy,
    merge_graphs,
)

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
    Neo4jSyncPlan,
    build_neo4j_sync_plan,
    sanitize_neo4j_identifier,
    validate_neo4j_config,
)

# Structured provenance metadata helpers
from .provenance import (
    ProvenanceRecord,
    attach_provenance,
    find_text_provenance,
    provenance_from_metadata,
)

# Relationship Model
from .relationship_model import (
    RELATIONSHIP_CATEGORIES,
    EnrichedRelationship,
    RelationshipType,
    RelationshipTypeClassifier,
    create_enriched_relationship,
)

# Graph validation
from .validation import (
    ValidationIssue,
    ValidationReport,
    load_graph_json,
    validate_graph_data,
    validate_graph_file,
)

# Snapshot-based graph versioning
from .versioning import (
    GraphVersion,
    VersionManifest,
    create_snapshot,
    diff_versions,
    list_versions,
    rollback_to_version,
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
    "KG",
    "RELATIONSHIP_CATEGORIES",
    "Cluster",
    "CommunityReport",
    "CommunityReportGenerator",
    "EdgeMergePolicy",
    "EnhancedKG",
    "EnrichedRelationship",
    "GraphMerger",
    "GraphVersion",
    "KGDiff",
    "KGEdge",
    "KGNode",
    "KGVisualizer",
    "MergeStrategy",
    "Neo4jConfig",
    "Neo4jExporter",
    "Neo4jSyncPlan",
    "NodeMergePolicy",
    "ProvenanceEdge",
    "ProvenanceGraph",
    "ProvenanceNode",
    "ProvenanceRecord",
    "RelationshipType",
    "RelationshipTypeClassifier",
    "SnapshotDiff",
    "ValidationIssue",
    "ValidationReport",
    "VersionManifest",
    "VisualizationAdapter",
    "apply_hub_relation_proxy_split",
    "attach_provenance",
    "build_neo4j_sync_plan",
    "create_enriched_relationship",
    "create_snapshot",
    "diff_graph_data",
    "diff_versions",
    "find_text_provenance",
    "list_versions",
    "load_graph_json",
    "merge_graphs",
    "provenance_from_metadata",
    "rollback_to_version",
    "sanitize_neo4j_identifier",
    "validate_graph_data",
    "validate_graph_file",
    "validate_neo4j_config",
]
