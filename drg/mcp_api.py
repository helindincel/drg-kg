"""
MCP (Model Context Protocol) API wrapper for DRG.

.. deprecated::
    This module implements MCP manually (JSON-RPC 2.0 dataclasses) and does **not**
    use the official `mcp` Python SDK.  It has no transport layer (stdio / HTTP+SSE)
    and will not work as a real MCP server.

    Use :mod:`drg.mcp_server` instead, which is built on top of the official SDK
    (``pip install drg-kg[mcp]``).  This module is kept for backward compatibility
    only and will be removed in a future major version.

Legacy usage (still works, but deprecated)::

    from drg.mcp_api import DRGMCPAPI  # DeprecationWarning raised at import

For a real MCP server::

    from drg.mcp_server import create_mcp_server
    mcp = create_mcp_server()
    mcp.run()
"""

import warnings

warnings.warn(
    "drg.mcp_api is deprecated and will be removed in a future major version. "
    "Use drg.mcp_server (pip install drg-kg[mcp]) for a real MCP server built on "
    "the official MCP Python SDK.",
    DeprecationWarning,
    stacklevel=2,
)

import json  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from enum import Enum  # noqa: E402
from typing import Any  # noqa: E402

from .extract import extract_typed  # noqa: E402
from .graph.kg_core import EnhancedKG, KGNode  # noqa: E402
from .schema import DRGSchema, EnhancedDRGSchema, Entity, Relation  # noqa: E402


class MCPErrorCode(str, Enum):
    """MCP error codes."""

    INVALID_REQUEST = "invalid_request"
    METHOD_NOT_FOUND = "method_not_found"
    INVALID_PARAMS = "invalid_params"
    INTERNAL_ERROR = "internal_error"
    SCHEMA_ERROR = "schema_error"
    EXTRACTION_ERROR = "extraction_error"


@dataclass
class MCPRequest:
    """MCP-style request object."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | int | None = None
    jsonrpc: str = "2.0"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPRequest":
        return cls(
            method=data.get("method", ""),
            params=data.get("params", {}),
            id=data.get("id"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
        }
        if self.id is not None:
            result["id"] = self.id
        return result


@dataclass
class MCPResponse:
    """MCP-style response object."""

    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    id: str | int | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        result = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            result["id"] = self.id

        if self.error:
            result["error"] = self.error
        else:
            result["result"] = self.result

        return result

    @classmethod
    def success(cls, result: dict[str, Any], request_id: str | int | None = None) -> "MCPResponse":
        return cls(result=result, id=request_id)

    @classmethod
    def error_response(
        cls,
        code: MCPErrorCode,
        message: str,
        data: dict[str, Any] | None = None,
        request_id: str | int | None = None,
    ) -> "MCPResponse":
        error = {
            "code": code.value,
            "message": message,
        }
        if data:
            error["data"] = data
        return cls(error=error, id=request_id)


class DRGMCPAPI:
    """
    MCP-style API wrapper for DRG.

    Provides a standardized interface for AI agents to interact with DRG's
    knowledge graph extraction capabilities.
    """

    def __init__(self):
        self._schemas: dict[str, DRGSchema | EnhancedDRGSchema] = {}
        self._knowledge_graphs: dict[str, EnhancedKG] = {}

    def handle_request(self, request: dict[str, Any] | MCPRequest) -> MCPResponse:
        """
        Handle an MCP request and return a response.

        Args:
            request: MCP request as dict or MCPRequest object

        Returns:
            MCPResponse object
        """
        if isinstance(request, dict):
            request = MCPRequest.from_dict(request)

        method = request.method
        params = request.params
        request_id = request.id

        # Route to appropriate handler
        try:
            if method == "drg/list_tools":
                result = self._list_tools()
            elif method == "drg/define_schema":
                result = self._define_schema(params)
            elif method == "drg/extract":
                result = self._extract(params)
            elif method == "drg/build_kg":
                result = self._build_kg(params)
            elif method == "drg/get_kg":
                result = self._get_kg(params)
            elif method == "drg/export_kg":
                result = self._export_kg(params)
            elif method == "drg/list_schemas":
                result = self._list_schemas()
            elif method == "drg/get_schema":
                result = self._get_schema(params)
            else:
                return MCPResponse.error_response(
                    MCPErrorCode.METHOD_NOT_FOUND,
                    f"Unknown method: {method}",
                    request_id=request_id,
                )

            return MCPResponse.success(result, request_id=request_id)

        except ValueError as e:
            return MCPResponse.error_response(
                MCPErrorCode.INVALID_PARAMS,
                str(e),
                request_id=request_id,
            )
        except Exception as e:
            return MCPResponse.error_response(
                MCPErrorCode.INTERNAL_ERROR,
                f"Internal error: {e!s}",
                {"type": type(e).__name__},
                request_id=request_id,
            )

    def _list_tools(self) -> dict[str, Any]:
        """List available DRG tools/capabilities."""
        return {
            "tools": [
                {
                    "name": "drg/define_schema",
                    "description": "Define a schema for entity and relation extraction",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "schema_id": {
                                "type": "string",
                                "description": "Unique identifier for the schema",
                            },
                            "schema": {
                                "type": "object",
                                "description": "Schema definition (DRGSchema or EnhancedDRGSchema format)",
                            },
                        },
                        "required": ["schema_id", "schema"],
                    },
                },
                {
                    "name": "drg/extract",
                    "description": "Extract entities and relations from text using a defined schema",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Input text to extract from"},
                            "schema_id": {
                                "type": "string",
                                "description": "ID of the schema to use",
                            },
                        },
                        "required": ["text", "schema_id"],
                    },
                },
                {
                    "name": "drg/build_kg",
                    "description": "Build a knowledge graph from extracted entities and relations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "kg_id": {
                                "type": "string",
                                "description": "Unique identifier for the knowledge graph",
                            },
                            "entities": {
                                "type": "array",
                                "description": "List of (entity_name, entity_type) tuples",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                            },
                            "triples": {
                                "type": "array",
                                "description": "List of (source, relation, target) triples",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 3,
                                    "maxItems": 3,
                                },
                            },
                        },
                        "required": ["kg_id", "entities", "triples"],
                    },
                },
                {
                    "name": "drg/get_kg",
                    "description": "Get a knowledge graph by ID",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "kg_id": {"type": "string", "description": "ID of the knowledge graph"},
                        },
                        "required": ["kg_id"],
                    },
                },
                {
                    "name": "drg/export_kg",
                    "description": "Export a knowledge graph in various formats",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "kg_id": {"type": "string", "description": "ID of the knowledge graph"},
                            "format": {
                                "type": "string",
                                "enum": ["json", "jsonld", "enriched"],
                                "description": "Export format",
                            },
                        },
                        "required": ["kg_id", "format"],
                    },
                },
                {
                    "name": "drg/list_schemas",
                    "description": "List all defined schemas",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "drg/get_schema",
                    "description": "Get a schema by ID",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "schema_id": {"type": "string", "description": "ID of the schema"},
                        },
                        "required": ["schema_id"],
                    },
                },
            ]
        }

    def _define_schema(self, params: dict[str, Any]) -> dict[str, Any]:
        """Define a schema."""
        schema_id = params.get("schema_id")
        schema_data = params.get("schema")

        if not schema_id:
            raise ValueError("schema_id is required")
        if not schema_data:
            raise ValueError("schema is required")

        # Parse schema from dict
        schema = self._parse_schema(schema_data)
        self._schemas[schema_id] = schema

        return {
            "schema_id": schema_id,
            "status": "defined",
            "entity_types": self._get_schema_summary(schema),
        }

    def _parse_schema(self, schema_data: dict[str, Any]) -> DRGSchema | EnhancedDRGSchema:
        """Parse schema from dictionary."""
        # Check if it's an EnhancedDRGSchema
        if "entity_types" in schema_data:
            # Enhanced schema
            from .schema import EntityType, RelationGroup

            entity_types = [
                EntityType(
                    name=et["name"],
                    description=et.get("description", ""),
                    examples=et.get("examples", []),
                    properties=et.get("properties", {}),
                )
                for et in schema_data["entity_types"]
            ]

            relation_groups = []
            if "relation_groups" in schema_data:
                for rg_data in schema_data["relation_groups"]:
                    relations = [
                        Relation(
                            name=r["name"],
                            src=r["src"],
                            dst=r["dst"],
                        )
                        for r in rg_data.get("relations", [])
                    ]
                    relation_groups.append(
                        RelationGroup(
                            name=rg_data["name"],
                            description=rg_data.get("description", ""),
                            relations=relations,
                        )
                    )

            return EnhancedDRGSchema(
                entity_types=entity_types,
                relation_groups=relation_groups,
                auto_discovery=schema_data.get("auto_discovery", False),
            )
        else:
            # Legacy DRGSchema
            entities = [Entity(e["name"]) for e in schema_data.get("entities", [])]
            relations = [
                Relation(r["name"], r["src"], r["dst"]) for r in schema_data.get("relations", [])
            ]
            return DRGSchema(entities=entities, relations=relations)

    def _get_schema_summary(self, schema: DRGSchema | EnhancedDRGSchema) -> dict[str, Any]:
        """Get summary of schema."""
        if isinstance(schema, EnhancedDRGSchema):
            return {
                "entity_types": [et.name for et in schema.entity_types],
                "relation_groups": len(schema.relation_groups),
            }
        else:
            return {
                "entity_types": [e.name for e in schema.entities],
                "relations": len(schema.relations),
            }

    def _extract(self, params: dict[str, Any]) -> dict[str, Any]:
        """Extract entities and relations from text."""
        text = params.get("text")
        schema_id = params.get("schema_id")

        if not text:
            raise ValueError("text is required")
        if not schema_id:
            raise ValueError("schema_id is required")

        if schema_id not in self._schemas:
            raise ValueError(
                f"Schema '{schema_id}' not found. Define it first using drg/define_schema"
            )

        schema = self._schemas[schema_id]
        entities, triples = extract_typed(text, schema)

        return {
            "entities": [{"name": e[0], "type": e[1]} for e in entities],
            "triples": [{"source": t[0], "relation": t[1], "target": t[2]} for t in triples],
            "counts": {
                "entities": len(entities),
                "triples": len(triples),
            },
        }

    def _build_kg(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build a knowledge graph from entities and triples.

        Supports enriched metadata including temporal information (start_time, end_time),
        confidence scores, and negation flags. Domain-agnostic - works for any domain.
        """
        from .extract import create_kgedge_from_triple

        kg_id = params.get("kg_id")
        entities_data = params.get("entities", [])
        triples_data = params.get("triples", [])
        enriched_metadata = params.get(
            "enriched_metadata", []
        )  # Optional: list of dicts with temporal, confidence, negation

        if not kg_id:
            raise ValueError("kg_id is required")

        # Convert entities and triples to tuples
        entities = [(e[0], e[1]) for e in entities_data]
        triples = [(t[0], t[1], t[2]) for t in triples_data]

        # Create EnhancedKG
        kg = EnhancedKG()

        # Add nodes
        for entity_name, entity_type in entities:
            kg.add_node(KGNode(id=entity_name, type=entity_type))

        # Add edges with enriched metadata (temporal, confidence, negation)
        for i, (source, relation, target) in enumerate(triples):
            # Get enriched metadata for this triple if available
            enriched_dict = None
            if enriched_metadata and i < len(enriched_metadata):
                enriched_dict = enriched_metadata[i]

            # Use helper function to create KGEdge with temporal information
            edge = create_kgedge_from_triple(
                (source, relation, target), enriched_metadata=enriched_dict
            )
            kg.add_edge(edge)

        self._knowledge_graphs[kg_id] = kg

        return {
            "kg_id": kg_id,
            "status": "built",
            "node_count": len(kg.nodes),
            "edge_count": len(kg.edges),
        }

    def _get_kg(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get a knowledge graph by ID."""
        kg_id = params.get("kg_id")

        if not kg_id:
            raise ValueError("kg_id is required")

        if kg_id not in self._knowledge_graphs:
            raise ValueError(f"Knowledge graph '{kg_id}' not found")

        kg = self._knowledge_graphs[kg_id]
        return json.loads(kg.to_json())

    def _export_kg(self, params: dict[str, Any]) -> dict[str, Any]:
        """Export a knowledge graph in specified format."""
        kg_id = params.get("kg_id")
        format_type = params.get("format", "json")

        if not kg_id:
            raise ValueError("kg_id is required")

        if kg_id not in self._knowledge_graphs:
            raise ValueError(f"Knowledge graph '{kg_id}' not found")

        kg = self._knowledge_graphs[kg_id]

        if format_type == "json":
            return {"format": "json", "data": json.loads(kg.to_json())}
        elif format_type == "jsonld":
            jsonld_str = kg.to_json_ld()
            return {"format": "jsonld", "data": json.loads(jsonld_str)}
        elif format_type == "enriched":
            return {"format": "enriched", "data": kg.to_enriched_format()}
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _list_schemas(self) -> dict[str, Any]:
        """List all defined schemas."""
        return {
            "schemas": [
                {
                    "schema_id": schema_id,
                    "summary": self._get_schema_summary(schema),
                }
                for schema_id, schema in self._schemas.items()
            ]
        }

    def _get_schema(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get a schema by ID."""
        schema_id = params.get("schema_id")

        if not schema_id:
            raise ValueError("schema_id is required")

        if schema_id not in self._schemas:
            raise ValueError(f"Schema '{schema_id}' not found")

        schema = self._schemas[schema_id]
        # Convert schema to dict (simplified representation)
        if isinstance(schema, EnhancedDRGSchema):
            return {
                "schema_id": schema_id,
                "type": "enhanced",
                "entity_types": [
                    {
                        "name": et.name,
                        "description": et.description,
                        "examples": et.examples,
                    }
                    for et in schema.entity_types
                ],
                "relation_groups": [
                    {
                        "name": rg.name,
                        "description": rg.description,
                        "relations": [
                            {"name": r.name, "src": r.src, "dst": r.dst} for r in rg.relations
                        ],
                    }
                    for rg in schema.relation_groups
                ],
            }
        else:
            return {
                "schema_id": schema_id,
                "type": "legacy",
                "entities": [{"name": e.name} for e in schema.entities],
                "relations": [
                    {"name": r.name, "src": r.src, "dst": r.dst} for r in schema.relations
                ],
            }


def create_mcp_api() -> DRGMCPAPI:
    """Create a new DRGMCPAPI instance."""
    return DRGMCPAPI()
