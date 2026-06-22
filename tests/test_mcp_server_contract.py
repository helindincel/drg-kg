"""Contract tests for the official MCP SDK server surface."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from drg import mcp_server


@pytest.fixture(autouse=True)
def _clear_mcp_state():
    mcp_server._schemas.clear()
    mcp_server._knowledge_graphs.clear()
    yield
    mcp_server._schemas.clear()
    mcp_server._knowledge_graphs.clear()


def test_create_mcp_server_returns_configured_instance():
    server = mcp_server.create_mcp_server()
    assert server is mcp_server.mcp


def test_define_schema_and_list_schemas():
    result = mcp_server.drg_define_schema(
        "company_schema",
        {
            "entity_types": [
                {"name": "Company", "description": "Business organizations"},
                {"name": "Product", "description": "Commercial products"},
            ],
            "relation_groups": [
                {
                    "name": "commercial",
                    "relations": [{"name": "produces", "src": "Company", "dst": "Product"}],
                }
            ],
        },
    )

    assert result["status"] == "defined"
    schemas = mcp_server.drg_list_schemas()["schemas"]
    assert schemas[0]["schema_id"] == "company_schema"
    assert schemas[0]["summary"]["type"] == "enhanced"


def test_build_get_and_export_kg_without_llm():
    build = mcp_server.drg_build_kg(
        kg_id="kg1",
        entities=[["Apple", "Company"], ["iPhone", "Product"]],
        triples=[["Apple", "produces", "iPhone"]],
    )
    assert build["node_count"] == 2
    assert build["edge_count"] == 1

    graph = mcp_server.drg_get_kg("kg1")
    assert {node["id"] for node in graph["nodes"]} == {"Apple", "iPhone"}

    exported = mcp_server.drg_export_kg("kg1", "json")
    assert exported["format"] == "json"
    assert len(exported["data"]["edges"]) == 1
    assert mcp_server.drg_list_kgs()["knowledge_graphs"][0]["kg_id"] == "kg1"


def test_missing_kg_raises_clear_error():
    with pytest.raises(ValueError, match="not found"):
        mcp_server.drg_get_kg("missing")
