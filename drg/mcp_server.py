"""
DRG MCP Server — built on the official MCP Python SDK.

Requires the ``mcp`` extra::

    pip install drg-kg[mcp]

Usage (stdio transport, for Claude Desktop / any MCP client)::

    python -m drg.mcp_server          # stdio
    python -m drg.mcp_server --http   # HTTP+SSE on http://localhost:8765

Exposed tools
-------------
- ``drg_define_schema``  — define an entity/relation extraction schema
- ``drg_extract``        — extract entities & relations from text
- ``drg_build_kg``       — build an EnhancedKG from extracted data
- ``drg_get_kg``         — retrieve a stored KG as JSON
- ``drg_export_kg``      — export a KG (json / jsonld / enriched)
- ``drg_list_schemas``   — list all registered schemas
- ``drg_list_kgs``       — list all stored knowledge graphs
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as _exc:
    raise ImportError(
        "The official MCP SDK is required for drg.mcp_server. "
        "Install it with: pip install drg-kg[mcp]"
    ) from _exc

from .extract import extract_typed  # noqa: E402
from .graph.kg_core import EnhancedKG, KGEdge, KGNode  # noqa: E402
from .schema import (  # noqa: E402
    DRGSchema,
    EnhancedDRGSchema,
    Entity,
    EntityType,
    Relation,
    RelationGroup,
)

# ---------------------------------------------------------------------------
# In-memory stores (stateful across tool calls within one server session)
# ---------------------------------------------------------------------------
# WARNING: These module-level dicts are process-local.  In a multi-worker
# deployment (e.g. ``uvicorn --workers N``) each worker has its own copy;
# a schema or KG registered in one worker is invisible to others.  For
# single-worker stdio / HTTP+SSE usage (the intended MCP deployment model)
# this is fine.  If you need multi-worker support, swap these dicts for a
# shared backend (Redis, SQLite, etc.) and inject via FastMCP lifespan.
_schemas: dict[str, DRGSchema | EnhancedDRGSchema] = {}
_knowledge_graphs: dict[str, EnhancedKG] = {}

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP("DRG Knowledge Graph")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parse_schema(schema_data: dict[str, Any]) -> DRGSchema | EnhancedDRGSchema:
    """Parse a schema dict into a DRGSchema or EnhancedDRGSchema."""
    if "entity_types" in schema_data:
        entity_types = [
            EntityType(
                name=et["name"],
                description=et.get("description", ""),
                examples=et.get("examples", []),
                properties=et.get("properties", {}),
            )
            for et in schema_data["entity_types"]
        ]
        relation_groups = [
            RelationGroup(
                name=rg["name"],
                description=rg.get("description", ""),
                relations=[
                    Relation(name=r["name"], src=r["src"], dst=r["dst"])
                    for r in rg.get("relations", [])
                ],
            )
            for rg in schema_data.get("relation_groups", [])
        ]
        return EnhancedDRGSchema(
            entity_types=entity_types,
            relation_groups=relation_groups,
            auto_discovery=schema_data.get("auto_discovery", False),
        )
    else:
        return DRGSchema(
            entities=[Entity(e["name"]) for e in schema_data.get("entities", [])],
            relations=[
                Relation(name=r["name"], src=r["src"], dst=r["dst"])
                for r in schema_data.get("relations", [])
            ],
        )


def _schema_summary(schema: DRGSchema | EnhancedDRGSchema) -> dict[str, Any]:
    if isinstance(schema, EnhancedDRGSchema):
        return {
            "type": "enhanced",
            "entity_types": [et.name for et in schema.entity_types],
            "relation_group_count": len(schema.relation_groups),
        }
    return {
        "type": "legacy",
        "entities": [e.name for e in schema.entities],
        "relation_count": len(schema.relations),
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def drg_define_schema(schema_id: str, schema: dict) -> dict:
    """Define an entity/relation extraction schema.

    Args:
        schema_id: Unique identifier for the schema.
        schema: Schema definition dict.  Two formats are supported:

            **Enhanced** (recommended)::

                {
                  "entity_types": [{"name": "Company", "description": "..."}],
                  "relation_groups": [{"name": "g1", "relations": [
                      {"name": "produces", "src": "Company", "dst": "Product"}
                  ]}]
                }

            **Legacy**::

                {
                  "entities": [{"name": "Company"}],
                  "relations": [{"name": "produces", "src": "Company", "dst": "Product"}]
                }
    """
    parsed = _parse_schema(schema)
    _schemas[schema_id] = parsed
    return {"schema_id": schema_id, "status": "defined", "summary": _schema_summary(parsed)}


@mcp.tool()
def drg_extract(text: str, schema_id: str) -> dict:
    """Extract entities and relations from text using a registered schema.

    Args:
        text: Input text to extract from.
        schema_id: ID of a previously defined schema (see drg_define_schema).
    """
    if schema_id not in _schemas:
        raise ValueError(f"Schema '{schema_id}' not found. Call drg_define_schema first.")

    schema = _schemas[schema_id]
    entities, triples = extract_typed(text, schema)

    return {
        "entities": [{"name": e[0], "type": e[1]} for e in entities],
        "triples": [{"source": t[0], "relation": t[1], "target": t[2]} for t in triples],
        "counts": {"entities": len(entities), "triples": len(triples)},
    }


@mcp.tool()
def drg_build_kg(
    kg_id: str,
    entities: list[list[str]],
    triples: list[list[str]],
) -> dict:
    """Build an EnhancedKG from extracted entities and triples.

    Args:
        kg_id: Unique identifier for this knowledge graph.
        entities: List of ``[entity_name, entity_type]`` pairs.
        triples: List of ``[source, relation, target]`` triples.
    """
    kg = EnhancedKG()
    for pair in entities:
        if len(pair) < 2:
            raise ValueError(f"Entity pair must be [name, type], got: {pair}")
        kg.add_node(KGNode(id=pair[0], type=pair[1]))

    for triple in triples:
        if len(triple) < 3:
            raise ValueError(f"Triple must be [source, relation, target], got: {triple}")
        source, relation, target = triple[0], triple[1], triple[2]
        # Ensure nodes exist
        if source not in kg.nodes:
            kg.add_node(KGNode(id=source))
        if target not in kg.nodes:
            kg.add_node(KGNode(id=target))
        kg.add_edge(
            KGEdge(
                source=source,
                target=target,
                relationship_type=relation,
                relationship_detail=relation,
            )
        )

    _knowledge_graphs[kg_id] = kg
    return {
        "kg_id": kg_id,
        "status": "built",
        "node_count": len(kg.nodes),
        "edge_count": len(kg.edges),
    }


@mcp.tool()
def drg_get_kg(kg_id: str) -> dict:
    """Retrieve a stored knowledge graph as JSON.

    Args:
        kg_id: ID of a previously built knowledge graph.
    """
    if kg_id not in _knowledge_graphs:
        raise ValueError(f"Knowledge graph '{kg_id}' not found. Call drg_build_kg first.")
    return json.loads(_knowledge_graphs[kg_id].to_json())


@mcp.tool()
def drg_export_kg(kg_id: str, format: str = "json") -> dict:
    """Export a knowledge graph in a specified format.

    Args:
        kg_id: ID of the knowledge graph.
        format: One of ``json``, ``jsonld``, or ``enriched``.
    """
    if kg_id not in _knowledge_graphs:
        raise ValueError(f"Knowledge graph '{kg_id}' not found.")

    kg = _knowledge_graphs[kg_id]
    fmt = format.lower()

    if fmt == "json":
        return {"format": "json", "data": json.loads(kg.to_json())}
    elif fmt == "jsonld":
        return {"format": "jsonld", "data": json.loads(kg.to_json_ld())}
    elif fmt == "enriched":
        return {"format": "enriched", "data": kg.to_enriched_format()}
    else:
        raise ValueError(f"Unsupported format '{format}'. Choose: json, jsonld, enriched.")


@mcp.tool()
def drg_list_schemas() -> dict:
    """List all registered schemas."""
    return {
        "schemas": [
            {"schema_id": sid, "summary": _schema_summary(s)} for sid, s in _schemas.items()
        ]
    }


@mcp.tool()
def drg_list_kgs() -> dict:
    """List all stored knowledge graphs."""
    return {
        "knowledge_graphs": [
            {
                "kg_id": kid,
                "node_count": len(kg.nodes),
                "edge_count": len(kg.edges),
            }
            for kid, kg in _knowledge_graphs.items()
        ]
    }


# ---------------------------------------------------------------------------
# Factory helper (for embedding in existing apps)
# ---------------------------------------------------------------------------


def create_mcp_server() -> FastMCP:
    """Return the configured FastMCP server instance.

    Example — run via stdio::

        server = create_mcp_server()
        server.run()

    Example — run via HTTP+SSE::

        server = create_mcp_server()
        server.run(transport="sse")
    """
    return mcp


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DRG MCP Server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use HTTP+SSE transport instead of stdio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for HTTP+SSE transport (default: 8765)",
    )
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="sse")
    else:
        mcp.run()
