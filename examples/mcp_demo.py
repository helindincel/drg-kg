#!/usr/bin/env python3
"""Demo for the official DRG MCP server tool contract.

Run the real server with:

    python -m drg.mcp_server

This script calls the same tool functions directly so it can run offline and
without launching an MCP client.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drg.mcp_server import (
    create_mcp_server,
    drg_build_kg,
    drg_define_schema,
    drg_export_kg,
    drg_get_kg,
    drg_list_kgs,
    drg_list_schemas,
)


def print_payload(title: str, payload: dict) -> None:
    """Pretty print a tool result."""
    print(title)
    print("-" * len(title))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()


def main():
    print("=" * 70)
    print("DRG MCP Server Demo")
    print("=" * 70)
    print()

    server = create_mcp_server()
    print(f"Loaded MCP server: {server.name if hasattr(server, 'name') else server!r}")
    print()

    schema_result = drg_define_schema(
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
    print_payload("1. Define schema", schema_result)

    build_result = drg_build_kg(
        kg_id="example_kg",
        entities=[["Apple", "Company"], ["iPhone", "Product"]],
        triples=[["Apple", "produces", "iPhone"]],
    )
    print_payload("2. Build KG", build_result)

    graph = drg_get_kg("example_kg")
    print_payload(
        "3. KG summary",
        {
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
            "clusters": len(graph.get("clusters", [])),
        },
    )

    export = drg_export_kg("example_kg", format="json")
    print_payload(
        "4. Export JSON", {"format": export["format"], "keys": list(export["data"].keys())}
    )

    print_payload("5. List schemas", drg_list_schemas())
    print_payload("6. List KGs", drg_list_kgs())

    print("=" * 70)
    print("Demo completed.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
