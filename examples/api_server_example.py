"""
DRG API Server Example

This example demonstrates how to:
1. Load a knowledge graph
2. Start the FastAPI server
3. Access the web UI and API endpoints

Usage:
    # Set Neo4j credentials (optional)
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=your_password
    python examples/api_server_example.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drg.api import DRGAPIServer
from drg.graph import EnhancedKG
from drg.graph.kg_core import Cluster, KGEdge, KGNode
from drg.graph.neo4j_exporter import Neo4jConfig


def create_sample_kg() -> EnhancedKG:
    """Create a sample knowledge graph for demonstration."""
    kg = EnhancedKG()

    # Add sample nodes
    nodes = [
        KGNode(id="Apple", type="Company", properties={"industry": "Technology"}),
        KGNode(id="iPhone", type="Product", properties={"category": "Smartphone"}),
        KGNode(id="Tim Cook", type="Person", properties={"role": "CEO"}),
        KGNode(id="Cupertino", type="Location", properties={"type": "City"}),
        KGNode(id="Google", type="Company", properties={"industry": "Technology"}),
        KGNode(id="Android", type="Product", properties={"category": "OS"}),
    ]

    for node in nodes:
        kg.add_node(node)

    # Add sample edges
    edges = [
        KGEdge(
            source="Apple",
            target="iPhone",
            relationship_type="produces",
            relationship_detail="Apple manufactures iPhone",
        ),
        KGEdge(
            source="Tim Cook",
            target="Apple",
            relationship_type="works_for",
            relationship_detail="Tim Cook is CEO of Apple",
        ),
        KGEdge(
            source="Apple",
            target="Cupertino",
            relationship_type="located_in",
            relationship_detail="Apple headquarters is in Cupertino",
        ),
        KGEdge(
            source="Google",
            target="Android",
            relationship_type="produces",
            relationship_detail="Google develops Android",
        ),
    ]

    for edge in edges:
        kg.add_edge(edge)

    # Add sample cluster
    cluster = Cluster(
        id="tech_companies",
        node_ids={"Apple", "Google", "iPhone", "Android"},
        metadata={"theme": "Technology companies and products"},
    )
    kg.add_cluster(cluster)

    return kg


def main():
    """Main function to start the API server."""
    import re
    import sys

    # Optional: --port <N> (or env DRG_API_PORT)
    port = int(os.getenv("DRG_API_PORT", "8000"))
    argv = list(sys.argv[1:])
    if "--port" in argv:
        idx = argv.index("--port")
        if idx + 1 < len(argv):
            port = int(argv[idx + 1])
            del argv[idx : idx + 2]

    # Example name: command line argument > environment variable > auto-detect latest > default
    if len(argv) > 0:
        example_name = argv[0]
        if example_name.isdigit():
            example_name = f"{example_name}example"
    elif os.getenv("DRG_EXAMPLE"):
        example_name = os.getenv("DRG_EXAMPLE")
        if example_name.isdigit():
            example_name = f"{example_name}example"
    else:
        # Auto-detect: Find the most recently modified KG file
        kg_files = list(Path("outputs").glob("*example*_kg.json"))
        if kg_files:
            # Sort by modification time (most recent first)
            latest_kg = max(kg_files, key=lambda p: p.stat().st_mtime)
            # Extract example name from filename (e.g., "outputs/3example_kg.json" -> "3example")
            example_name = latest_kg.stem.replace("_kg", "")
            print(f"🔍 En son güncellenen KG dosyası bulundu: {latest_kg.name}")
        else:
            example_name = "1example"

    print("=" * 70)
    print(f"DRG API Server Example - {example_name.upper()}")
    print("=" * 70)
    print(f"📌 Example seçimi: {example_name}")
    print(f"   (Değiştirmek için: export DRG_EXAMPLE=3example veya python {sys.argv[0]} 3)")

    def _try_load_kg_file(path: Path) -> EnhancedKG | None:
        """Load a KG JSON file into EnhancedKG.

        Supports both:
        - EnhancedKG JSON (nodes/edges/clusters with relationship_type keys)
        - Legacy CLI KG JSON (nodes/edges with edge key 'type')
        """
        if not path.exists():
            return None
        import json

        with open(path, encoding="utf-8") as f:
            kg_data = json.load(f)

        kg = EnhancedKG()

        # Load nodes (both formats share: {"id": "...", "type": "..."} at minimum)
        for node_data in kg_data.get("nodes", []):
            try:
                node = KGNode.from_dict(node_data)
            except Exception:
                node = KGNode(
                    id=node_data.get("id", ""),
                    type=node_data.get("type"),
                    properties=node_data.get("properties", {}) or {},
                    metadata=node_data.get("metadata", {}) or {},
                )
            if node.id:
                kg.add_node(node)

        # Load edges (Enhanced format vs legacy format)
        for edge_data in kg_data.get("edges", []):
            # EnhancedKG edge format
            if "relationship_type" in edge_data:
                edge = KGEdge.from_dict(edge_data)
            # Legacy CLI edge format: {"source": s, "type": r, "target": o}
            else:
                src = edge_data.get("source")
                dst = edge_data.get("target")
                rel = edge_data.get("type") or edge_data.get("relationship_type")
                if not (src and dst and rel):
                    continue
                edge = KGEdge(
                    source=src,
                    target=dst,
                    relationship_type=rel,
                    relationship_detail=f"{src} {rel} {dst}",
                    metadata=edge_data.get("metadata", {}) or {},
                )

            # Ensure nodes exist
            if edge.source not in kg.nodes:
                kg.add_node(KGNode(id=edge.source, type=None))
            if edge.target not in kg.nodes:
                kg.add_node(KGNode(id=edge.target, type=None))
            kg.add_edge(edge)

        # Load clusters if present (only EnhancedKG export includes it)
        for cluster_data in kg_data.get("clusters", []):
            try:
                cluster = Cluster.from_dict(cluster_data)
                kg.add_cluster(cluster)
            except Exception:
                # If cluster is malformed, skip rather than failing the UI.
                continue

        return kg

    # Load knowledge graph from file if exists, otherwise use sample
    kg_path = Path(f"outputs/{example_name}_kg.json")
    kg = _try_load_kg_file(kg_path)

    # Backward compatibility: if full pipeline output doesn't exist, fall back to CLI output:
    # outputs/example{N}.json (where example_name is like "4example")
    if kg is None:
        m = re.match(r"^(\d+)example$", example_name)
        if m:
            legacy_path = Path(f"outputs/example{m.group(1)}.json")
            kg = _try_load_kg_file(legacy_path)
            if kg is not None:
                kg_path = legacy_path

    if kg is not None:
        print("\n1. Loading knowledge graph from file...")
        print(f"   📄 KG file: {kg_path}")
        print(
            f"   ✅ Loaded KG with {len(kg.nodes)} nodes, {len(kg.edges)} edges, {len(kg.clusters)} clusters"
        )
    else:
        print("\n1. Creating sample knowledge graph...")
        kg = create_sample_kg()
        print(
            f"   ✅ Created KG with {len(kg.nodes)} nodes, {len(kg.edges)} edges, {len(kg.clusters)} clusters"
        )

    # Optional: Neo4j configuration (comment out if not using Neo4j)
    print("\n2. Neo4j configuration...")
    neo4j_config = None
    neo4j_uri = os.getenv("NEO4J_URI")
    if neo4j_uri:
        neo4j_config = Neo4jConfig(
            uri=neo4j_uri,
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
        print("   ✅ Neo4j configuration loaded")
    else:
        print("   ℹ️  Neo4j not configured (set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD to enable)")

    # Create API server
    print("\n3. Creating API server...")
    server = DRGAPIServer(
        kg=kg,
        neo4j_config=neo4j_config,
    )
    print("   ✅ API server created")

    # Start server
    print("\n" + "=" * 70)
    print("Starting DRG API Server...")
    print("=" * 70)
    print(f"\n🌐 Web UI: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🔍 Graph API: http://localhost:{port}/api/graph")
    print(f"👥 Communities API: http://localhost:{port}/api/communities")
    print("\nPress Ctrl+C to stop the server\n")

    server.run(host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
