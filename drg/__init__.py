"""DRG - Declarative Relationship Generation"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Static type stubs for IDE autocompletion.
# These imports are never executed at runtime (TYPE_CHECKING is False), so they
# don't trigger heavy optional dependencies. They DO make extract_typed,
# KGExtractor, etc. visible to pyright/mypy and IDE intellisense.
if TYPE_CHECKING:
    from .clustering import (  # noqa: F401
        ClusteringAlgorithm,
        ClusterResult,
        LeidenClustering,
        LouvainClustering,
        SpectralClustering,
    )
    from .extract import (
        KGExtractor,
        create_kgedge_from_triple,
        extract_from_chunks,
        extract_from_chunks_async,
        extract_triples,
        extract_typed,
        extract_typed_async,
        generate_schema_from_text,
    )
    from .graph.kg_core import Cluster, EnhancedKG, KGEdge, KGNode  # noqa: F401

# Single source of truth: the version comes from setuptools_scm at build
# time. In editable/installed mode we read it via importlib.metadata. In a
# fresh checkout without a build step (rare; CI does build before test)
# we fall back to "0.0.0+unknown" so imports never crash.
#
# To bump: tag the commit (e.g. `git tag v0.1.0 && git push --tags`).
# Do NOT edit a hard-coded string here.
try:
    from ._version import __version__  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - only hit before first build
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("drg-kg")
    except (ImportError, PackageNotFoundError):  # pragma: no cover
        __version__ = "0.0.0+unknown"

# Core schema types (lightweight - import directly)
# Extraction functions can pull optional heavy dependencies (e.g., DSPy).
# Keep them lazily loaded so visualization/server usage doesn't require extraction deps.
# Legacy graph class (lightweight)
from .graph import KG
from .schema import (
    DRGSchema,
    EnhancedDRGSchema,
    Entity,
    EntityGroup,
    EntityType,
    PropertyGroup,
    Relation,
    RelationGroup,
)

# Heavy modules use lazy loading to avoid importing all dependencies at startup
# These will be imported only when explicitly requested

__all__ = [
    "KG",
    "ConfidenceScore",
    "ConfidenceStrategy",
    "DRGSchema",
    "DefaultConfidenceStrategy",
    "EnhancedDRGSchema",
    "Entity",
    "EntityGroup",
    "EntityType",
    "Event",
    "EventProvenance",
    "EventRole",
    "EventTimestamp",
    "EventTypeDefinition",
    "EventTypeRegistry",
    "EvidenceBundle",
    "Explanation",
    "GraphMetricScore",
    "GraphQuery",
    "InMemoryBackend",
    "KGExtractor",
    "PropertyGroup",
    "QueryBackend",
    "QueryError",
    "Relation",
    "RelationGroup",
    "TextSpan",
    "create_kgedge_from_triple",
    "default_event_registry",
    "example_event_registry",
    "extract_events",
    "extract_from_chunks",
    "extract_from_chunks_async",
    "extract_triples",
    "extract_typed",
    "extract_typed_async",
    "generate_schema_from_text",
]


# Schema loading utility (available via submodule import: drg.schema.load_schema_from_json)
# Not exported at top level to keep core imports lightweight


def __getattr__(name: str):
    """Lazy loading for heavy modules that are not frequently used.

    This allows importing heavy dependencies only when explicitly requested,
    improving startup time and reducing memory usage.

    Note: Imported objects are cached in globals() for subsequent access.
    """
    # Lazy loading mapping: name -> module_path
    lazy_imports = {
        # Confidence framework (lightweight, no heavy deps)
        "ConfidenceScore": ".confidence",
        "ConfidenceStrategy": ".confidence",
        "DefaultConfidenceStrategy": ".confidence",
        # Extraction (may require optional heavy deps)
        "extract_typed": ".extract",
        "extract_typed_async": ".extract",
        "extract_triples": ".extract",
        "extract_from_chunks": ".extract",
        "extract_from_chunks_async": ".extract",
        "create_kgedge_from_triple": ".extract",
        "KGExtractor": ".extract",
        "generate_schema_from_text": ".extract",
        # Graph module (heavy - many submodules)
        "RelationshipType": ".graph",
        "EnrichedRelationship": ".graph",
        "RelationshipTypeClassifier": ".graph",
        "create_enriched_relationship": ".graph",
        "RELATIONSHIP_CATEGORIES": ".graph",
        "KGNode": ".graph",
        "KGEdge": ".graph",
        "Cluster": ".graph",
        "EnhancedKG": ".graph",
        # Incremental updates (opt-in; pure stdlib + entity_resolution)
        "GraphMerger": ".graph",
        "KGDiff": ".graph",
        "MergeStrategy": ".graph",
        "NodeMergePolicy": ".graph",
        "EdgeMergePolicy": ".graph",
        "merge_graphs": ".graph",
        "KGVisualizer": ".graph",
        "DEFAULT_NODE_COLORS": ".graph",
        "DEFAULT_EDGE_COLORS": ".graph",
        "CommunityReport": ".graph",
        "CommunityReportGenerator": ".graph",
        "Neo4jConfig": ".graph",
        "Neo4jExporter": ".graph",
        "ProvenanceNode": ".graph",
        "ProvenanceEdge": ".graph",
        "ProvenanceGraph": ".graph",
        "VisualizationAdapter": ".graph",
        # Chunking module
        "ChunkingStrategy": ".chunking",
        "TokenBasedChunker": ".chunking",
        "SentenceBasedChunker": ".chunking",
        "create_chunker": ".chunking",
        "ChunkValidator": ".chunking",
        "validate_chunks": ".chunking",
        # Embedding module
        "EmbeddingProvider": ".embedding",
        "OpenAIEmbeddingProvider": ".embedding",
        "GeminiEmbeddingProvider": ".embedding",
        "OpenRouterEmbeddingProvider": ".embedding",
        "LocalEmbeddingProvider": ".embedding",
        "create_embedding_provider": ".embedding",
        # Public API focuses on KG extraction and analysis utilities.
        # Clustering module
        "ClusteringAlgorithm": ".clustering",
        "LouvainClustering": ".clustering",
        "LeidenClustering": ".clustering",
        "SpectralClustering": ".clustering",
        "create_clustering_algorithm": ".clustering",
        "ClusterSummarizer": ".clustering",
        "create_summarizer": ".clustering",
        # MCP server is in drg.mcp_server (use: pip install drg-kg[mcp])
        "Event": ".events",
        "EventRole": ".events",
        "EventTypeDefinition": ".events",
        "EventTypeRegistry": ".events",
        "EventTimestamp": ".events",
        "EventProvenance": ".events",
        "TextSpan": ".events",
        "default_event_registry": ".events",
        "example_event_registry": ".events",
        "extract_events": ".events",
        "events_to_kg_nodes_and_edges": ".events",
        "event_to_kg_node": ".events",
        "event_to_role_edges": ".events",
        "is_event_node": ".events",
        "is_event_role_edge": ".events",
        # Query & reasoning layer (pure stdlib + drg.graph; no LLM)
        "GraphQuery": ".query",
        "QueryBackend": ".query",
        "InMemoryBackend": ".query",
        "QueryError": ".query",
        "Explanation": ".query",
        "EvidenceBundle": ".query",
        "EntityView": ".query",
        "EdgeView": ".query",
        "GraphPath": ".query",
        "GraphMetricScore": ".query",
        "QueryAnswer": ".query",
        "NeighborhoodView": ".query",
        "RelatedEntityMatch": ".query",
        "EventView": ".query",
        "CommunityView": ".query",
        "EvidenceItem": ".query",
        "Provenance": ".query",
        "EntityMatch": ".query",
    }

    if name in lazy_imports:
        module_path = lazy_imports[name]
        # Import the module
        import importlib

        module = importlib.import_module(module_path, __name__)
        # Get the requested attribute
        attr = getattr(module, name)
        # Cache it in globals for subsequent access
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
