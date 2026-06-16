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
    from .extract import (  # noqa: F401
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
    # Legacy graph
    "KG",
    "DRGSchema",
    "EnhancedDRGSchema",
    # Core schema classes
    "Entity",
    "EntityGroup",
    "EntityType",
    "KGExtractor",
    "PropertyGroup",
    "Relation",
    "RelationGroup",
    "extract_triples",
    # Core extraction
    "extract_typed",
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
        # Extraction (may require optional heavy deps)
        "extract_typed": ".extract",
        "extract_triples": ".extract",
        "KGExtractor": ".extract",
        "generate_schema_from_text": ".extract",
        # Graph module (heavy - many submodules)
        "PropertyDefinition": ".graph",
        "EntityClassDefinition": ".graph",
        "DatasetAgnosticSchemaGenerator": ".graph",
        "create_default_schema": ".graph",
        "RelationshipType": ".graph",
        "EnrichedRelationship": ".graph",
        "RelationshipTypeClassifier": ".graph",
        "create_enriched_relationship": ".graph",
        "RELATIONSHIP_CATEGORIES": ".graph",
        "KGNode": ".graph",
        "KGEdge": ".graph",
        "Cluster": ".graph",
        "EnhancedKG": ".graph",
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
        # NOTE: This project is not a RAG/retrieval framework. Public API intentionally does not
        # export retrieval/search helpers; focus is KG extraction + analysis utilities.
        # Clustering module
        "ClusteringAlgorithm": ".clustering",
        "LouvainClustering": ".clustering",
        "LeidenClustering": ".clustering",
        "SpectralClustering": ".clustering",
        "create_clustering_algorithm": ".clustering",
        "ClusterSummarizer": ".clustering",
        "create_summarizer": ".clustering",
        # Optimizer module (heavy - requires DSPy)
        "OptimizerConfig": ".optimizer",
        "DRGOptimizer": ".optimizer",
        "create_optimizer": ".optimizer",
        "evaluate_extraction": ".optimizer",
        "ExtractionMetrics": ".optimizer",
        "calculate_metrics": ".optimizer",
        "compare_metrics": ".optimizer",
        # MCP API module (heavy)
        "DRGMCPAPI": ".mcp_api",
        "MCPRequest": ".mcp_api",
        "MCPResponse": ".mcp_api",
        "MCPErrorCode": ".mcp_api",
        "create_mcp_api": ".mcp_api",
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
