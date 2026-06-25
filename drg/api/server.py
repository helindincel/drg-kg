"""
FastAPI Server for DRG Knowledge Graph API

Endpoints:
- GET /api/graph - Full graph data
- GET /api/graph/stats - Graph statistics
- GET /api/communities - Community/cluster data
- GET /api/communities/{cluster_id} - Specific community report
- GET /api/provenance/{query_id} - Query provenance chain
- POST /api/query - Execute query and get provenance
- GET /api/visualization/{format} - Graph visualization data (cytoscape, vis-network, d3)
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Input limits (defence against prompt injection and runaway requests).
# Override via environment variables for custom deployments.
# ---------------------------------------------------------------------------
_MAX_QUERY_CHARS: int = int(os.getenv("DRG_MAX_QUERY_CHARS", "2000"))
_MAX_TEXT_CHARS: int = int(os.getenv("DRG_MAX_TEXT_CHARS", "100_000"))

try:
    from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.security.api_key import APIKeyHeader
    from fastapi.staticfiles import StaticFiles
except ImportError:
    FastAPI = None

from ..graph import (  # noqa: E402
    EnhancedKG,
    Neo4jConfig,
    Neo4jExporter,
    ProvenanceEdge,
    ProvenanceGraph,
    ProvenanceNode,
    VisualizationAdapter,
    build_neo4j_sync_plan,
    validate_neo4j_config,
)
from ..graph.auto_clusters import ensure_clusters  # noqa: E402
from ..graph.query_engine import execute_query as execute_deterministic_query  # noqa: E402
from ..utils.env_loader import load_dotenv  # noqa: E402

# ---------------------------------------------------------------------------
# Optional API-key authentication
# ---------------------------------------------------------------------------
# Set the DRG_API_KEY environment variable to enable authentication.
# When the variable is absent (or empty) the server runs without auth — suitable
# for local / trusted-network deployments only.
#
# Example:
#   export DRG_API_KEY="my-secret-token"
#   curl -H "X-API-Key: my-secret-token" http://localhost:8000/api/graph

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False) if FastAPI else None


def _require_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> None:
    """FastAPI dependency that enforces the DRG_API_KEY when it is configured."""
    expected = os.getenv("DRG_API_KEY", "").strip()
    if not expected:
        # Auth not configured — allow all requests.
        return
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


logger = logging.getLogger(__name__)


# Pydantic models for API
class QueryRequest(BaseModel):
    """Query request model."""

    query: str = Field(..., max_length=2000)
    k_entities: int = Field(10, ge=1, le=100)
    k_reports: int = Field(5, ge=0, le=50)
    k_context_chunks: int = Field(5, ge=0, le=50)

    @field_validator("query")
    @classmethod
    def _strip_and_validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty or whitespace")
        # Reject null bytes and non-printable control characters that have
        # no legitimate use in a KG query but are common in injection payloads.
        if any(ord(c) < 32 and c not in ("\t", "\n") for c in v):
            raise ValueError("query contains invalid control characters")
        return v


class QueryResponse(BaseModel):
    """Query response model."""

    query: str
    answer: str | None = None
    provenance_id: str


class ExtractRequest(BaseModel):
    """Text-to-KG extraction request model."""

    text: str = Field(..., max_length=_MAX_TEXT_CHARS)
    extraction_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    model: str | None = None
    api_key: str | None = Field(default=None, exclude=True)
    base_url: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    document_id: str | None = None
    store_graph: bool = True

    @field_validator("text")
    @classmethod
    def _strip_and_validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty or whitespace")
        if any(ord(c) < 32 and c not in ("\t", "\n", "\r") for c in v):
            raise ValueError("text contains invalid control characters")
        return v


class ExtractResponse(BaseModel):
    """Text-to-KG extraction response model."""

    document_id: str
    entities: list[dict[str, str | None]]
    triples: list[dict[str, str]]
    counts: dict[str, int]
    graph: dict[str, Any]
    stored: bool


class GraphUpdateRequest(ExtractRequest):
    """Incremental graph update request."""

    update_strategy: str = Field(default="prefer_existing")
    edge_policy: str = Field(default="skip")

    @field_validator("update_strategy")
    @classmethod
    def _validate_update_strategy(cls, v: str) -> str:
        allowed = {"prefer_existing", "prefer_new", "union"}
        if v not in allowed:
            raise ValueError(f"update_strategy must be one of {sorted(allowed)}")
        return v

    @field_validator("edge_policy")
    @classmethod
    def _validate_edge_policy(cls, v: str) -> str:
        allowed = {"skip", "append_evidence", "max_confidence"}
        if v not in allowed:
            raise ValueError(f"edge_policy must be one of {sorted(allowed)}")
        return v


class GraphUpdateResponse(BaseModel):
    """Incremental graph update response."""

    document_id: str
    diff: dict[str, Any]
    version: dict[str, Any]
    counts: dict[str, int]
    graph: dict[str, Any]


def _default_extraction_schema():
    from ..schema import DRGSchema, Entity, Relation

    return DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")],
    )


def _schema_from_payload(schema_data: dict[str, Any] | None):
    """Parse API schema payloads without importing optional MCP modules."""

    if not schema_data:
        return _default_extraction_schema()

    from ..schema import (
        DRGSchema,
        EnhancedDRGSchema,
        Entity,
        EntityType,
        Relation,
        RelationGroup,
    )

    if "entity_types" in schema_data:
        entity_types = [
            EntityType(
                name=item["name"],
                description=item.get("description") or f"{item['name']} entities",
                examples=item.get("examples", []),
                properties=item.get("properties", {}),
            )
            for item in schema_data.get("entity_types", [])
        ]
        relation_groups = [
            RelationGroup(
                name=group.get("name", "relations"),
                description=group.get("description", ""),
                relations=[
                    Relation(name=rel["name"], src=rel["src"], dst=rel["dst"])
                    for rel in group.get("relations", [])
                ],
            )
            for group in schema_data.get("relation_groups", [])
        ]
        return EnhancedDRGSchema(
            entity_types=entity_types,
            relation_groups=relation_groups,
            auto_discovery=schema_data.get("auto_discovery", False),
        )

    return DRGSchema(
        entities=[Entity(item["name"]) for item in schema_data.get("entities", [])],
        relations=[
            Relation(name=rel["name"], src=rel["src"], dst=rel["dst"])
            for rel in schema_data.get("relations", [])
        ],
    )


def _build_request_lm(request: ExtractRequest):
    """Build a per-request dspy.LM without touching os.environ.

    Returns a configured ``dspy.LM`` instance when the request supplies
    model or API-key overrides, or ``None`` when no overrides are present
    and the globally configured LM should be used.

    This function is the thread-safe replacement for the former
    ``_apply_extraction_env`` / ``_restore_env`` pair.  Use it with
    ``dspy.context(lm=...)`` which scopes the override to the current
    asyncio task / thread via Python's contextvars machinery.
    """
    if not request.model and not request.api_key and not request.base_url:
        return None
    try:
        import dspy  # optional dep — only needed when extract extra is installed
    except ImportError:
        return None

    model_name = request.model or os.getenv("DRG_MODEL", "openai/gpt-4o-mini")

    kwargs: dict[str, object] = {}
    if request.base_url:
        kwargs["base_url"] = request.base_url
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.api_key:
        kwargs["api_key"] = request.api_key

    try:
        return dspy.LM(model_name, **kwargs)
    except Exception:
        logger.debug("Could not build per-request dspy.LM", exc_info=True)
        return None


def _has_provider_credentials(model: str, *, request_api_key: str | None = None) -> bool:
    if model.lower().startswith("ollama"):
        return True
    if request_api_key:
        return True
    return any(
        os.getenv(key)
        for key in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
        )
    )


def _graph_payload(kg: EnhancedKG) -> dict[str, Any]:
    payload = {
        "nodes": [node.to_dict() for node in kg.nodes.values()],
        "edges": [edge.to_dict() for edge in kg.edges],
        "clusters": [cluster.to_dict() for cluster in kg.clusters.values()],
    }
    if kg.metadata:
        payload["metadata"] = dict(kg.metadata)
    return payload


def create_app(
    kg: EnhancedKG | None = None,
    neo4j_config: Neo4jConfig | None = None,
    provenance_store: dict[str, ProvenanceGraph] | None = None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        kg: EnhancedKG instance
        neo4j_config: Optional Neo4j configuration for persistence
        provenance_store: Optional in-memory provenance store

    Returns:
        FastAPI application instance
    """
    if FastAPI is None:
        raise ImportError(
            "fastapi and uvicorn are required. Install with: pip install fastapi uvicorn"
        )

    if os.getenv("DRG_API_LOAD_ENV", "1").lower() not in {"0", "false", "no"}:
        load_dotenv(".env", override=False)

    app = FastAPI(
        title="DRG Knowledge Graph API",
        description="API for DRG Knowledge Graph visualization and querying",
        version="1.0.0",
        dependencies=[Depends(_require_api_key)],
    )

    # CORS middleware
    # DRG_CORS_ORIGINS: comma-separated list of allowed origins.
    # Defaults to ["*"] for local development; restrict in production:
    #   export DRG_CORS_ORIGINS="https://app.example.com,https://admin.example.com"
    #
    # Security note: allow_credentials=True is incompatible with allow_origins=["*"]
    # per the Fetch specification §3.2 — browsers will reject the combination.
    # We therefore only enable credentials when DRG_CORS_ORIGINS is set to a
    # non-wildcard value, and fall back to credentials=False for the open default.
    _cors_origins_env = os.getenv("DRG_CORS_ORIGINS", "*").strip()
    cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    _cors_wildcard = cors_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        # credentials require explicit origins — wildcard + credentials is rejected by browsers
        allow_credentials=not _cors_wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store app state
    app.state.kg = kg
    app.state.neo4j_config = neo4j_config
    app.state.provenance_store = provenance_store or {}
    app.state.visualization_adapter = VisualizationAdapter(kg) if kg else None
    app.state.graph_versions = []
    app.state.graph_snapshots = {}

    def _record_graph_version(
        kg: EnhancedKG,
        *,
        operation: str,
        document_id: str | None = None,
        diff_summary: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        versions: list[dict[str, Any]] = app.state.graph_versions
        version_id = f"v{kg.metadata.get('version', len(versions) + 1)}"
        existing_ids = {v["version_id"] for v in versions}
        if version_id in existing_ids:
            version_id = f"{version_id}-{len(versions) + 1}"
        entry: dict[str, Any] = {
            "version_id": version_id,
            "parent_version_id": versions[-1]["version_id"] if versions else None,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "operation": operation,
            "document_id": document_id,
            "diff_summary": dict(diff_summary or {}),
        }
        versions.append(entry)
        kg.metadata.setdefault("versions", [])
        kg.metadata["versions"].append(entry)
        app.state.graph_snapshots[version_id] = json.loads(kg.to_json())
        return entry

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
            },
        )
        return response

    # Mount static files (for web UI)
    static_dir = Path(__file__).parent / "static"
    templates_dir = Path(__file__).parent / "templates"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Root endpoint - serve web UI."""
        ui_file = templates_dir / "index.html"
        if ui_file.exists():
            return ui_file.read_text(encoding="utf-8")
        return "<h1>DRG Knowledge Graph API</h1><p>Use /docs for API documentation</p>"

    @app.get("/healthz")
    async def healthz():
        """Liveness probe: the process can serve HTTP."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        """Readiness probe: report loaded graph and optional Neo4j config status."""
        kg_loaded = app.state.kg is not None
        neo4j_errors = (
            validate_neo4j_config(app.state.neo4j_config) if app.state.neo4j_config else []
        )
        return {
            "status": "ready" if kg_loaded else "degraded",
            "kg_loaded": kg_loaded,
            "neo4j_configured": app.state.neo4j_config is not None and not neo4j_errors,
            "neo4j_errors": neo4j_errors,
        }

    @app.get("/api/graph")
    async def get_graph():
        """Get full graph data."""
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        return _graph_payload(kg)

    @app.get("/api/graph/stats")
    async def get_graph_stats():
        """Get graph statistics."""
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        # Communities view expects clusters; generate deterministic clusters if missing
        ensure_clusters(kg)

        # Calculate statistics
        node_types: dict[str, int] = {}
        for node in kg.nodes.values():
            node_type = node.type or "Unknown"
            node_types[node_type] = node_types.get(node_type, 0) + 1

        relationship_types: dict[str, int] = {}
        for edge in kg.edges:
            rel_type = edge.relationship_type
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        return {
            "node_count": len(kg.nodes),
            "edge_count": len(kg.edges),
            "cluster_count": len(kg.clusters),
            "node_types": node_types,
            "relationship_types": relationship_types,
        }

    @app.post("/api/extract", response_model=ExtractResponse)
    async def extract_knowledge_graph(request: ExtractRequest):
        """Extract a Knowledge Graph from raw text.

        This is the API equivalent of the CLI's text -> entities/triples ->
        EnhancedKG path. It uses the configured DSPy/LLM provider, so cloud
        models need an API key or a local model configuration.
        """

        from ..extract import extract_typed
        from ..graph.builders import build_enhanced_kg

        schema = _schema_from_payload(request.extraction_schema)
        document_id = request.document_id or f"api:{uuid.uuid4()}"
        lm_override = _build_request_lm(request)
        model = request.model or os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
        if not _has_provider_credentials(model, request_api_key=request.api_key):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Extraction requires a provider API key or a local Ollama model. "
                    "Pass `api_key`, set provider environment variables, or use an "
                    "ollama model with `base_url`."
                ),
            )
        try:
            import contextlib
            import dspy
            ctx = dspy.context(lm=lm_override) if lm_override is not None else contextlib.nullcontext()
            with ctx:
                entities_typed, triples = await asyncio.to_thread(extract_typed, request.text, schema)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("API extraction failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Extraction failed: {type(e).__name__}"
            ) from e

        triples = list(dict.fromkeys(triples))
        kg = await asyncio.to_thread(
            build_enhanced_kg,
            entities_typed=entities_typed,
            triples=triples,
            schema=schema,
            source_text=request.text,
            document_id=document_id,
        )

        graph = _graph_payload(kg)

        if request.store_graph:
            app.state.kg = kg
            app.state.visualization_adapter = VisualizationAdapter(kg)

        return ExtractResponse(
            document_id=document_id,
            entities=[{"name": name, "type": entity_type} for name, entity_type in entities_typed],
            triples=[
                {"source": source, "relation": relation, "target": target}
                for source, relation, target in triples
            ],
            counts={
                "entities": len(entities_typed),
                "triples": len(triples),
                "nodes": len(kg.nodes),
                "edges": len(kg.edges),
            },
            graph=graph,
            stored=request.store_graph,
        )

    @app.post("/api/graph/update", response_model=GraphUpdateResponse)
    async def update_knowledge_graph(request: GraphUpdateRequest):
        """Incrementally merge extracted facts into the loaded graph."""

        from ..extract import extract_typed
        from ..graph import EdgeMergePolicy, GraphMerger, MergeStrategy, NodeMergePolicy
        from ..graph.builders import build_enhanced_kg

        schema = _schema_from_payload(request.extraction_schema)
        document_id = request.document_id or f"api:{uuid.uuid4()}"
        lm_override = _build_request_lm(request)
        model = request.model or os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
        if not _has_provider_credentials(model, request_api_key=request.api_key):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Graph update requires a provider API key or a local Ollama model. "
                    "Pass `api_key`, set provider environment variables, or use an "
                    "ollama model with `base_url`."
                ),
            )
        try:
            import contextlib
            import dspy
            ctx = dspy.context(lm=lm_override) if lm_override is not None else contextlib.nullcontext()
            with ctx:
                entities_typed, triples = await asyncio.to_thread(extract_typed, request.text, schema)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("API graph update failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Graph update failed: {type(e).__name__}") from e

        triples = list(dict.fromkeys(triples))
        incoming = await asyncio.to_thread(
            build_enhanced_kg,
            entities_typed=entities_typed,
            triples=triples,
            schema=schema,
            source_text=request.text,
            document_id=document_id,
        )
        base = app.state.kg or EnhancedKG()
        strategy = MergeStrategy(
            node_policy=NodeMergePolicy(request.update_strategy),
            edge_policy=EdgeMergePolicy(request.edge_policy),
        )
        diff = await asyncio.to_thread(
            GraphMerger(strategy).merge,
            base,
            incoming,
            document_id=document_id,
        )
        version = _record_graph_version(
            base,
            operation="merge",
            document_id=document_id,
            diff_summary=diff.summary(),
        )
        app.state.kg = base
        app.state.visualization_adapter = VisualizationAdapter(base)
        graph = _graph_payload(base)
        return GraphUpdateResponse(
            document_id=document_id,
            diff=diff.to_dict(),
            version=version,
            counts={
                "nodes": len(base.nodes),
                "edges": len(base.edges),
                "clusters": len(base.clusters),
            },
            graph=graph,
        )

    @app.get("/api/graph/versions")
    async def get_graph_versions():
        """List in-memory graph versions created through the API."""

        return {"versions": list(app.state.graph_versions)}

    @app.get("/api/graph/versions/{version_id}/diff")
    async def diff_graph_version(version_id: str, compare_to: str | None = None):
        """Diff a version against its parent or an explicit version id."""

        from ..graph.diff import diff_graph_data

        snapshots: dict[str, dict[str, Any]] = app.state.graph_snapshots
        versions: list[dict[str, Any]] = app.state.graph_versions
        if version_id not in snapshots:
            raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
        version = next(v for v in versions if v["version_id"] == version_id)
        base_id = compare_to or version.get("parent_version_id")
        if not base_id:
            return {"changed": False, "summary": {}, "diff": None}
        if base_id not in snapshots:
            raise HTTPException(status_code=404, detail=f"Version {base_id} not found")
        diff = diff_graph_data(snapshots[base_id], snapshots[version_id])
        return diff.to_dict()

    @app.post("/api/graph/versions/{version_id}/rollback")
    async def rollback_graph_version(version_id: str):
        """Rollback the in-memory graph to a previously recorded API snapshot."""

        snapshots: dict[str, dict[str, Any]] = app.state.graph_snapshots
        if version_id not in snapshots:
            raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
        restored = EnhancedKG.from_dict(snapshots[version_id])
        app.state.kg = restored
        app.state.visualization_adapter = VisualizationAdapter(restored)
        version = _record_graph_version(restored, operation="rollback", document_id=None)
        return {"rolled_back_to": version_id, "version": version, "graph": _graph_payload(restored)}

    @app.get("/api/provenance/entity/{entity_id}")
    async def get_entity_provenance(entity_id: str):
        """Return provenance metadata for a graph entity."""

        from ..graph.provenance import provenance_from_metadata

        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")
        node = kg.nodes.get(entity_id)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
        return {
            "entity_id": entity_id,
            "provenance": provenance_from_metadata(node.metadata).to_dict(),
            "metadata": dict(node.metadata),
        }

    @app.get("/api/provenance/edge")
    async def get_edge_provenance(source: str, relationship_type: str, target: str):
        """Return provenance metadata for matching graph edges."""

        from ..graph.provenance import provenance_from_metadata

        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")
        matches = [
            edge
            for edge in kg.edges
            if edge.source == source
            and edge.target == target
            and edge.relationship_type.lower() == relationship_type.lower()
        ]
        if not matches:
            raise HTTPException(status_code=404, detail="Edge not found")
        return {
            "edge": {
                "source": source,
                "relationship_type": relationship_type,
                "target": target,
            },
            "provenance": [
                {
                    "provenance": provenance_from_metadata(edge.metadata).to_dict(),
                    "metadata": dict(edge.metadata),
                    "confidence": edge.confidence,
                }
                for edge in matches
            ],
        }

    @app.get("/api/communities")
    async def get_communities():
        """Get all communities/clusters."""
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        # UI expects communities to exist; generate deterministic clusters if missing.
        ensure_clusters(kg)

        return {
            "clusters": [cluster.to_dict() for cluster in kg.clusters.values()],
        }

    @app.get("/api/communities/{cluster_id}")
    async def get_community(cluster_id: str):
        """Get specific community report."""
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        ensure_clusters(kg)

        cluster = kg.clusters.get(cluster_id)
        if cluster is None:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        # Generate community report if generator available
        from ..graph.community_report import CommunityReportGenerator

        report_generator = CommunityReportGenerator(kg)
        report = await asyncio.to_thread(report_generator.generate_report, cluster)

        return report.to_dict()

    @app.get("/api/visualization/{format}")
    async def get_visualization(
        format: str,
        hub_split: int | None = Query(
            None, description="1 to enable UI hub splitting, 0 to disable"
        ),
        hub_split_threshold: int | None = Query(
            None, description="Degree threshold for hub splitting"
        ),
    ):
        """Get graph visualization data in specified format."""
        kg = app.state.kg
        adapter = app.state.visualization_adapter

        if kg is None or adapter is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        format_lower = format.lower()

        if format_lower == "cytoscape":
            data = adapter.kg_to_cytoscape(
                kg,
                hub_split=(bool(hub_split) if hub_split is not None else None),
                hub_split_threshold=hub_split_threshold,
            )
            return {"elements": data}

        elif format_lower == "vis-network" or format_lower == "visnetwork":
            data = adapter.kg_to_vis_network(kg)
            return data

        elif format_lower == "d3":
            data = adapter.kg_to_d3_json(kg)
            return data

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Supported: cytoscape, vis-network, d3",
            )

    @app.get("/api/visualization/communities/{format}")
    async def get_communities_visualization(
        format: str,
        hub_split: int | None = Query(
            None, description="1 to enable UI hub splitting, 0 to disable"
        ),
        hub_split_threshold: int | None = Query(
            None, description="Degree threshold for hub splitting"
        ),
    ):
        """Get communities visualization with color coding."""
        kg = app.state.kg
        adapter = app.state.visualization_adapter

        if kg is None or adapter is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        ensure_clusters(kg)

        format_lower = format.lower()

        if format_lower == "cytoscape":
            data = adapter.communities_to_cytoscape(
                kg,
                hub_split=(bool(hub_split) if hub_split is not None else None),
                hub_split_threshold=hub_split_threshold,
            )
            return {"elements": data}

        else:
            # For other formats, use regular visualization
            return await get_visualization(format)

    @app.post("/api/query")
    async def execute_query(request: QueryRequest):
        """Execute a deterministic KG lookup for the UI query box.

        It performs:
        - entity string matching
        - optional relation filtering (e.g., "(works_at)" or "rel:develops")
        - neighborhood expansion around seed entities
        """
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        q = request.query  # already stripped and validated by QueryRequest

        # Deterministic graph query (run in thread pool to avoid blocking event loop)
        result = await asyncio.to_thread(
            execute_deterministic_query,
            kg=kg,
            query=q,
            k_entities=request.k_entities,
            k_edges=max(20, request.k_entities * 6),
        )

        # Build a minimal provenance graph (optional, but keeps API contract stable)
        prov_id = str(uuid.uuid4())
        prov_nodes: list[ProvenanceNode] = [
            ProvenanceNode(id=f"query:{prov_id}", type="query", label="Query", data={"query": q}),
            ProvenanceNode(
                id=f"answer:{prov_id}",
                type="answer",
                label="Answer",
                data={"answer": result.answer},
            ),
        ]
        prov_edges: list[ProvenanceEdge] = [
            ProvenanceEdge(
                source=f"query:{prov_id}",
                target=f"answer:{prov_id}",
                type="generated_from",
                label="deterministic_lookup",
                weight=1.0,
            )
        ]
        for ent in result.seed_entities[:20]:
            prov_nodes.append(
                ProvenanceNode(id=f"entity:{ent}", type="entity", label=ent, data={"id": ent})
            )
            prov_edges.append(
                ProvenanceEdge(
                    source=f"query:{prov_id}",
                    target=f"entity:{ent}",
                    type="matched_entity",
                    label="entity_match",
                    weight=1.0,
                )
            )

        provenance = ProvenanceGraph(
            nodes=prov_nodes,
            edges=prov_edges,
            query=q,
            answer=result.answer,
            metadata={
                "engine": "deterministic",
                "seed_entities": result.seed_entities,
            },
        )
        app.state.provenance_store[prov_id] = provenance

        return QueryResponse(
            query=q,
            answer=result.answer,
            provenance_id=prov_id,
        )

    @app.get("/api/provenance/{provenance_id}")
    async def get_provenance(provenance_id: str, format: str = "json"):
        """Get query provenance chain."""
        provenance_store = app.state.provenance_store
        adapter = app.state.visualization_adapter

        if provenance_id not in provenance_store:
            raise HTTPException(status_code=404, detail=f"Provenance {provenance_id} not found")

        provenance = provenance_store[provenance_id]

        format_lower = format.lower()

        if format_lower == "cytoscape" and adapter:
            data = adapter.provenance_to_cytoscape(provenance)
            return {"elements": data}

        else:
            if adapter:
                return adapter.provenance_to_json(provenance)
            else:
                return {
                    "query": provenance.query,
                    "answer": provenance.answer,
                    "nodes": [
                        {
                            "id": node.id,
                            "type": node.type,
                            "label": node.label,
                            "data": node.data,
                        }
                        for node in provenance.nodes
                    ],
                    "edges": [
                        {
                            "source": edge.source,
                            "target": edge.target,
                            "type": edge.type,
                            "label": edge.label,
                        }
                        for edge in provenance.edges
                    ],
                }

    @app.get("/api/neo4j/test")
    async def test_neo4j_config():
        """Validate Neo4j configuration and verify connectivity when possible."""
        neo4j_config = app.state.neo4j_config
        config_errors = validate_neo4j_config(neo4j_config)
        if config_errors:
            raise HTTPException(status_code=400, detail={"errors": config_errors})

        exporter = None
        try:
            exporter = Neo4jExporter(neo4j_config)
            return {"status": "ok", "database": neo4j_config.database, "uri": neo4j_config.uri}
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Neo4j connection test failed: {e!s}"
            ) from e
        finally:
            if exporter is not None:
                exporter.close()

    @app.post("/api/neo4j/sync")
    async def sync_to_neo4j(
        clear_existing: bool = Query(False),
        dry_run: bool = Query(False, description="Preview sync without writing to Neo4j"),
    ):
        """Sync knowledge graph to Neo4j."""
        kg = app.state.kg
        neo4j_config = app.state.neo4j_config

        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        config_errors = validate_neo4j_config(neo4j_config)
        if config_errors:
            raise HTTPException(status_code=400, detail={"errors": config_errors})

        plan = build_neo4j_sync_plan(kg)
        if dry_run:
            return {
                "status": "dry_run",
                "clear_existing": clear_existing,
                "plan": plan.to_dict(),
            }

        exporter = None
        try:
            exporter = Neo4jExporter(neo4j_config)
            stats = exporter.sync_kg(kg, clear_existing=clear_existing)

            return {
                "status": "success",
                "plan": plan.to_dict(),
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"Neo4j sync failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Neo4j sync failed: {e!s}") from e
        finally:
            if exporter is not None:
                exporter.close()

    @app.get("/api/neo4j/stats")
    async def get_neo4j_stats():
        """Get Neo4j graph statistics."""
        neo4j_config = app.state.neo4j_config

        config_errors = validate_neo4j_config(neo4j_config)
        if config_errors:
            raise HTTPException(status_code=400, detail={"errors": config_errors})

        exporter = None
        try:
            exporter = Neo4jExporter(neo4j_config)
            stats = exporter.get_graph_stats()

            return stats
        except Exception as e:
            logger.error(f"Failed to get Neo4j stats: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get Neo4j stats: {e!s}") from e
        finally:
            if exporter is not None:
                exporter.close()

    return app


class DRGAPIServer:
    """DRG API Server wrapper class."""

    def __init__(
        self,
        kg: EnhancedKG | None = None,
        neo4j_config: Neo4jConfig | None = None,
    ):
        """Initialize API server.

        Args:
            kg: EnhancedKG instance
            neo4j_config: Optional Neo4j configuration
        """
        self.kg = kg
        self.neo4j_config = neo4j_config
        self.app = create_app(
            kg=kg,
            neo4j_config=neo4j_config,
        )

    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run API server.

        Args:
            host: Host to bind to
            port: Port to bind to
            reload: Enable auto-reload (development)
        """
        try:
            import uvicorn
        except ImportError as err:
            raise ImportError("uvicorn is required. Install with: pip install uvicorn") from err

        uvicorn.run(self.app, host=host, port=port, reload=reload)
