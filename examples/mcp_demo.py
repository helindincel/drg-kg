#!/usr/bin/env python3
"""
MCP API Demo - Demonstrates the MCP-style API interface for DRG.

This script shows how AI agents can interact with DRG through the MCP API.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drg.mcp_api import MCPRequest, create_mcp_api


def print_response(response):
    """Pretty print MCP response."""
    print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
    print()


def main():
    print("=" * 70)
    print("🔌 DRG MCP API Demo")
    print("=" * 70)
    print()

    # Create MCP API instance
    api = create_mcp_api()

    # 1. List available tools
    print("1️⃣  Listing available tools...")
    print("-" * 70)
    request = MCPRequest(method="drg/list_tools", id=1)
    response = api.handle_request(request)
    print_response(response)
    print()

    # 2. Define a schema
    print("2️⃣  Defining a schema...")
    print("-" * 70)
    schema_request = MCPRequest(
        method="drg/define_schema",
        params={
            "schema_id": "company_schema",
            "schema": {
                "entities": [
                    {"name": "Company"},
                    {"name": "Product"},
                ],
                "relations": [
                    {"name": "produces", "src": "Company", "dst": "Product"},
                ],
            },
        },
        id=2,
    )
    response = api.handle_request(schema_request)
    print_response(response)
    print()

    # 3. Extract entities and relations from text
    print("3️⃣  Extracting entities and relations...")
    print("-" * 70)
    text = "Apple released the iPhone 16 in September 2025. Samsung also produces the Galaxy S24."
    print(f"📄 Text: {text}\n")

    extract_request = MCPRequest(
        method="drg/extract",
        params={
            "text": text,
            "schema_id": "company_schema",
        },
        id=3,
    )
    response = api.handle_request(extract_request)
    print_response(response)
    print()

    # Extract the entities and triples from response
    if response.result:
        entities_data = response.result.get("entities", [])
        triples_data = response.result.get("triples", [])

        # Convert to format expected by build_kg
        entities = [[e["name"], e["type"]] for e in entities_data]
        triples = [[t["source"], t["relation"], t["target"]] for t in triples_data]

        # 4. Build knowledge graph
        print("4️⃣  Building knowledge graph...")
        print("-" * 70)
        build_kg_request = MCPRequest(
            method="drg/build_kg",
            params={
                "kg_id": "example_kg",
                "entities": entities,
                "triples": triples,
            },
            id=4,
        )
        response = api.handle_request(build_kg_request)
        print_response(response)
        print()

        # 5. Get knowledge graph
        print("5️⃣  Getting knowledge graph...")
        print("-" * 70)
        get_kg_request = MCPRequest(
            method="drg/get_kg",
            params={"kg_id": "example_kg"},
            id=5,
        )
        response = api.handle_request(get_kg_request)
        # Print summary instead of full graph
        if response.result:
            result = response.result
            print("📊 Knowledge Graph Summary:")
            print(f"   Nodes: {len(result.get('nodes', []))}")
            print(f"   Edges: {len(result.get('edges', []))}")
            print(f"   Clusters: {len(result.get('clusters', []))}")
            print()
        print()

        # 6. Export knowledge graph in different formats
        print("6️⃣  Exporting knowledge graph (JSON format)...")
        print("-" * 70)
        export_request = MCPRequest(
            method="drg/export_kg",
            params={
                "kg_id": "example_kg",
                "format": "json",
            },
            id=6,
        )
        response = api.handle_request(export_request)
        if response.result and response.result.get("data"):
            data = response.result["data"]
            print(
                f"✓ Exported {len(data.get('nodes', []))} nodes, {len(data.get('edges', []))} edges"
            )
            print()

        # 7. List all schemas
        print("7️⃣  Listing all schemas...")
        print("-" * 70)
        list_schemas_request = MCPRequest(method="drg/list_schemas", id=7)
        response = api.handle_request(list_schemas_request)
        print_response(response)
        print()

        # 8. Get schema details
        print("8️⃣  Getting schema details...")
        print("-" * 70)
        get_schema_request = MCPRequest(
            method="drg/get_schema",
            params={"schema_id": "company_schema"},
            id=8,
        )
        response = api.handle_request(get_schema_request)
        print_response(response)
        print()

    # 9. Error handling example
    print("9️⃣  Error handling example (invalid schema_id)...")
    print("-" * 70)
    error_request = MCPRequest(
        method="drg/extract",
        params={
            "text": "Test text",
            "schema_id": "nonexistent_schema",
        },
        id=9,
    )
    response = api.handle_request(error_request)
    print_response(response)
    print()

    print("=" * 70)
    print("✅ Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    # Check if API key is available (optional, for actual extraction)
    import os

    api_key = (
        os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    )

    if not api_key:
        print("⚠️  Warning: No API key found. Extraction will use mock mode if available.")
        print(
            "   Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY for actual extraction.\n"
        )

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
