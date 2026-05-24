#!/usr/bin/env python3
"""
Full Pipeline Example: Example 1 için tüm pipeline'ı çalıştırır.

Pipeline adımları:
1. Chunking
2. Schema Generation (eğer yoksa)
3. KG Extraction (cross-chunk relationships ile)
4. EnhancedKG oluşturma
5. Entity Embeddings ekleme
6. Clustering
7. Community Reports
8. Outputs kaydetme

UI'da görüntülemek için API server başlatılabilir.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Project root'a ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx

from drg.chunking import create_chunker
from drg.clustering import create_clustering_algorithm
from drg.embedding import create_embedding_provider
from drg.extract import extract_from_chunks, generate_schema_from_text
from drg.graph.builders import build_enhanced_kg
from drg.graph.community_report import CommunityReportGenerator
from drg.graph.hub_mitigation import apply_hub_relation_proxy_split
from drg.graph.kg_core import EnhancedKG
from drg.schema import DRGSchema, EnhancedDRGSchema, Entity, Relation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_text(example_name: str) -> str:
    """Load text from inputs directory."""
    text_path = project_root / "inputs" / f"{example_name}_text.txt"
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    with open(text_path, encoding="utf-8") as f:
        return f.read()


def load_schema(example_name: str) -> EnhancedDRGSchema | None:
    """Load schema from inputs directory if exists."""
    schema_path = project_root / "inputs" / f"{example_name}_schema.json"
    if not schema_path.exists():
        return None

    with open(schema_path, encoding="utf-8") as f:
        schema_data = json.load(f)

    # EnhancedDRGSchema'ya yükle (eğer DRGSchema formatındaysa convert et)
    try:
        if "entity_types" in schema_data:
            return EnhancedDRGSchema.from_dict(schema_data)
        else:
            # Legacy DRGSchema formatı
            entities = [Entity(e["name"]) for e in schema_data.get("entities", [])]
            relations = [
                Relation(
                    r["name"], r.get("source", r.get("src", "")), r.get("target", r.get("dst", ""))
                )
                for r in schema_data.get("relations", [])
            ]
            return DRGSchema(entities=entities, relations=relations)
    except Exception as e:
        logger.warning(f"Schema loading failed: {e}, will generate schema from text")
        return None


def save_outputs(
    example_name: str, schema: EnhancedDRGSchema | DRGSchema, kg: EnhancedKG, reports: list
):
    """Save outputs to outputs directory."""
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Schema kaydet
    schema_path = outputs_dir / f"{example_name}_schema.json"
    if isinstance(schema, EnhancedDRGSchema):
        schema_dict = schema.to_dict()
    else:
        schema_dict = {
            "entities": [{"name": e.name} for e in schema.entities],
            "relations": [
                {"name": r.name, "source": r.src, "target": r.dst} for r in schema.relations
            ],
        }

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Schema saved to: {schema_path}")

    # KG kaydet
    kg_path = outputs_dir / f"{example_name}_kg.json"
    kg.save_json(str(kg_path), indent=2)
    logger.info(f"Knowledge graph saved to: {kg_path}")

    # Community reports kaydet
    if reports:
        reports_path = outputs_dir / f"{example_name}_community_reports.json"
        reports_dict = [
            report.to_dict() if hasattr(report, "to_dict") else report for report in reports
        ]
        with open(reports_path, "w", encoding="utf-8") as f:
            json.dump(reports_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Community reports saved to: {reports_path}")

    # Summary kaydet
    summary = {
        "example_name": example_name,
        "nodes": len(kg.nodes),
        "edges": len(kg.edges),
        "clusters": len(kg.clusters) if hasattr(kg, "clusters") else 0,
        "community_reports": len(reports),
    }
    summary_path = outputs_dir / f"{example_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to: {summary_path}")

    return kg_path


def run_full_pipeline(example_name: str = "1example"):
    """Run full pipeline for given example."""
    logger.info(f"🚀 Starting full pipeline for: {example_name}")

    # Step 1: Load text
    logger.info("📄 Step 1: Loading text...")
    text = load_text(example_name)
    logger.info(f"Loaded {len(text)} characters of text")

    # Step 2: Load or generate schema
    logger.info("📋 Step 2: Loading/Generating schema...")
    schema = load_schema(example_name)
    if schema is None:
        logger.info("Schema not found, generating from text...")
        # Auto-schema generation can exceed default max token budgets (some providers default ~1500),
        # which leads to truncated JSON and parsing failures. If the user didn't set a budget,
        # pick a safer default.
        if not os.getenv("DRG_MAX_TOKENS"):
            os.environ["DRG_MAX_TOKENS"] = "4096"
        schema = generate_schema_from_text(text)
        logger.info(
            f"Generated schema with {len(schema.entity_types) if isinstance(schema, EnhancedDRGSchema) else len(schema.entities)} entities"
        )

    # Step 3: Chunking
    logger.info("✂️  Step 3: Chunking text...")
    chunker = create_chunker(strategy="token_based", chunk_size=768, overlap_ratio=0.15)
    chunks = chunker.chunk(
        text, origin_dataset=example_name, origin_file=f"{example_name}_text.txt"
    )
    logger.info(f"Created {len(chunks)} chunks")

    # Step 4: KG Extraction (with cross-chunk relationships)
    logger.info("🔍 Step 4: Extracting knowledge graph...")
    logger.info("   - Using two-pass extraction for cross-chunk relationships")
    logger.info("   - Enabling entity resolution")
    logger.info("   - Enabling coreference resolution")

    # Try to create embedding provider for better entity resolution
    embedding_provider = None
    try:
        embedding_provider = create_embedding_provider("local")
        logger.info("   - Using local embedding provider for entity resolution")
    except Exception as e:
        logger.warning(f"   - Local embedding provider not available: {e}")

    entities, triples = extract_from_chunks(
        chunks=[
            {"text": chunk.text, "chunk_id": chunk.chunk_id, "metadata": chunk.metadata}
            for chunk in chunks
        ],
        schema=schema,
        enable_cross_chunk_relationships=True,
        enable_entity_resolution=True,
        enable_coreference_resolution=True,
        two_pass_extraction=True,
        embedding_provider=embedding_provider,
    )
    logger.info(f"Extracted {len(entities)} entities and {len(triples)} relationships")

    # Step 5: Create EnhancedKG
    logger.info("🕸️  Step 5: Creating Enhanced Knowledge Graph...")
    kg = build_enhanced_kg(
        entities_typed=entities,
        triples=triples,
        schema=schema,
        source_text=text,
    )

    logger.info(f"Created KG with {len(kg.nodes)} nodes and {len(kg.edges)} edges")

    # Optional: Export-time hub mitigation (persisted in output KG).
    # This reduces "single-node centered" star layouts (e.g., Company -> many Products)
    # by re-routing high-degree hubs through relation-specific proxy nodes.
    #
    # Enable with:
    #   export DRG_EXPORT_HUB_SPLIT=1
    #   export DRG_EXPORT_HUB_SPLIT_THRESHOLD=8
    export_hub_split = os.getenv("DRG_EXPORT_HUB_SPLIT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if export_hub_split:
        try:
            hub_threshold = int(os.getenv("DRG_EXPORT_HUB_SPLIT_THRESHOLD", "10"))
        except Exception:
            hub_threshold = 10
        stats = apply_hub_relation_proxy_split(kg, hub_degree_threshold=hub_threshold, enabled=True)
        logger.info(
            f"🧩 Hub-split applied (export-time): hubs={stats['hubs']}, "
            f"proxy_nodes={stats['proxy_nodes']}, edges_replaced={stats['edges_replaced']}, "
            f"connector_edges={stats['connector_edges']} (threshold={hub_threshold})"
        )

    # Step 6: Add entity embeddings
    logger.info("🧮 Step 6: Adding entity embeddings...")
    if embedding_provider is None:
        try:
            embedding_provider = create_embedding_provider("local")
        except Exception as e:
            logger.warning(f"Embedding provider not available, skipping embeddings: {e}")
            embedding_provider = None

    if embedding_provider:
        entity_texts = {node_id: node_id for node_id in kg.nodes.keys()}
        kg.add_entity_embeddings(embedding_provider, entity_texts)
        logger.info(f"Added embeddings for {len(kg.nodes)} entities")
    else:
        logger.warning("No embedding provider available, skipping embeddings")

    # Hub dissolution removed - not aligned with project architecture
    # Instead, we rely on improved LLM prompts and extraction logic
    # The extraction should produce balanced graphs from the start
    # Step 7: Clustering
    logger.info("🔗 Step 7: Clustering graph...")
    try:
        clustering_algorithm = create_clustering_algorithm("louvain")

        # Convert EnhancedKG to NetworkX
        G = nx.Graph()
        for node_id in kg.nodes.keys():
            G.add_node(node_id)
        for edge in kg.edges:
            G.add_edge(edge.source, edge.target)

        # Cluster
        clusters = clustering_algorithm.cluster(G)
        logger.info(f"Found {len(clusters)} clusters")

        # Add clusters to KG
        for i, cluster in enumerate(clusters):
            from drg.graph.kg_core import Cluster

            kg_cluster = Cluster(
                id=f"cluster_{i}", node_ids=set(cluster.nodes), metadata={"algorithm": "louvain"}
            )
            kg.add_cluster(kg_cluster)
    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
        clusters = []

    # Step 8: Community Reports
    logger.info("📊 Step 8: Generating community reports...")
    reports = []
    if clusters and len(kg.clusters) > 0:
        try:
            report_generator = CommunityReportGenerator(kg)
            reports = report_generator.generate_all_reports()
            logger.info(f"Generated {len(reports)} community reports")
        except Exception as e:
            logger.warning(f"Community report generation failed: {e}")

    # Step 9: Save outputs
    logger.info("💾 Step 9: Saving outputs...")
    kg_path = save_outputs(example_name, schema, kg, reports)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Pipeline completed successfully!")
    logger.info(f"   Example: {example_name}")
    logger.info(f"   Nodes: {len(kg.nodes)}")
    logger.info(f"   Edges: {len(kg.edges)}")
    logger.info(f"   Clusters: {len(kg.clusters) if hasattr(kg, 'clusters') else 0}")
    logger.info(f"   Community Reports: {len(reports)}")
    logger.info(f"   KG saved to: {kg_path}")
    logger.info("\n🌐 To view in UI, run:")
    logger.info(f"   python examples/api_server_example.py {example_name}")
    logger.info("=" * 60)

    return kg, schema, reports


if __name__ == "__main__":
    example_name = sys.argv[1] if len(sys.argv) > 1 else "1example"
    run_full_pipeline(example_name)
