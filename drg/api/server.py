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

import logging
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:
    FastAPI = None

from ..graph import (
    EnhancedKG,
    Neo4jConfig,
    Neo4jExporter,
    ProvenanceEdge,
    ProvenanceGraph,
    ProvenanceNode,
    VisualizationAdapter,
)
from ..graph.auto_clusters import ensure_clusters

# GraphRAG removed - not part of this project
from ..graph.query_engine import execute_query as execute_deterministic_query

logger = logging.getLogger(__name__)


# Pydantic models for API
class QueryRequest(BaseModel):
    """Query request model."""

    query: str
    k_entities: int = 10
    k_reports: int = 5
    k_context_chunks: int = 5


class QueryResponse(BaseModel):
    """Query response model."""

    query: str
    answer: str | None = None
    provenance_id: str
    retrieval_context: dict[str, Any]


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

    app = FastAPI(
        title="DRG Knowledge Graph API",
        description="API for DRG Knowledge Graph visualization and querying",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store app state
    app.state.kg = kg
    app.state.neo4j_config = neo4j_config
    app.state.provenance_store = provenance_store or {}
    app.state.visualization_adapter = VisualizationAdapter(kg) if kg else None

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

    @app.get("/api/graph")
    async def get_graph():
        """Get full graph data."""
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        return {
            "nodes": [node.to_dict() for node in kg.nodes.values()],
            "edges": [edge.to_dict() for edge in kg.edges],
            "clusters": [cluster.to_dict() for cluster in kg.clusters.values()],
        }

    @app.get("/api/graph/stats")
    async def get_graph_stats():
        """Get graph statistics."""
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        # Communities view expects clusters; generate deterministic clusters if missing
        ensure_clusters(kg)

        # Calculate statistics
        node_types = {}
        for node in kg.nodes.values():
            node_type = node.type or "Unknown"
            node_types[node_type] = node_types.get(node_type, 0) + 1

        relationship_types = {}
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
        report = report_generator.generate_report(cluster)

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
        """Execute query (deterministic KG lookup; NO RAG/LLM).

        This endpoint exists to power the UI query box without turning DRG into a retrieval framework.
        It performs:
        - entity string matching
        - optional relation filtering (e.g., "(works_at)" or "rel:develops")
        - neighborhood expansion around seed entities
        """
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        q = (request.query or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Deterministic graph query
        result = execute_deterministic_query(
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
                    type="retrieved_from",
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

        # UI expects: retrieval_context.seed_entities, and optionally .entities / .community_reports
        retrieval_context: dict[str, Any] = {
            "seed_entities": result.seed_entities,
            "entities": result.matched_entities,
            "community_reports": [],
            "matched_edges": result.matched_edges,
        }

        return QueryResponse(
            query=q,
            answer=result.answer,
            provenance_id=prov_id,
            retrieval_context=retrieval_context,
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

    @app.post("/api/neo4j/sync")
    async def sync_to_neo4j(clear_existing: bool = Query(False)):
        """Sync knowledge graph to Neo4j."""
        kg = app.state.kg
        neo4j_config = app.state.neo4j_config

        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")

        if neo4j_config is None:
            raise HTTPException(status_code=404, detail="Neo4j configuration not provided")

        try:
            exporter = Neo4jExporter(neo4j_config)
            stats = exporter.sync_kg(kg, clear_existing=clear_existing)
            exporter.close()

            return {
                "status": "success",
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"Neo4j sync failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Neo4j sync failed: {e!s}") from e

    @app.get("/api/neo4j/stats")
    async def get_neo4j_stats():
        """Get Neo4j graph statistics."""
        neo4j_config = app.state.neo4j_config

        if neo4j_config is None:
            raise HTTPException(status_code=404, detail="Neo4j configuration not provided")

        try:
            exporter = Neo4jExporter(neo4j_config)
            stats = exporter.get_graph_stats()
            exporter.close()

            return stats
        except Exception as e:
            logger.error(f"Failed to get Neo4j stats: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get Neo4j stats: {e!s}") from e

    return app


# GraphRAG removed - _create_provenance_from_retrieval function no longer available


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
