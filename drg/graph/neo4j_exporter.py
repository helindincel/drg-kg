"""
Neo4j Exporter - Knowledge Graph Persistence Layer

Exports EnhancedKG data to Neo4j graph database:
- Nodes (entities) with type, properties, metadata, embeddings
- Edges (relationships) with type, detail, metadata, weights
- Clusters (communities) with community IDs
- Semantic similarity edges (optional)
"""

import logging
from dataclasses import dataclass
from typing import Any

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

from .kg_core import EnhancedKG

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""

    uri: str
    user: str
    password: str
    database: str = "neo4j"


class Neo4jExporter:
    """
    Neo4j exporter for EnhancedKG.

    Syncs all nodes, relations, weights, similarity values and community IDs
    from EnhancedKG into Neo4j persistent graph store.
    """

    def __init__(self, config: Neo4jConfig):
        """Initialize Neo4j exporter.

        Args:
            config: Neo4j connection configuration
        """
        if GraphDatabase is None:
            raise ImportError("neo4j package is required. Install with: pip install neo4j")

        self.config = config
        self.driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
        logger.info(f"Connected to Neo4j at {config.uri}")

    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def sync_kg(self, kg: EnhancedKG, clear_existing: bool = False) -> dict[str, Any]:
        """Sync entire knowledge graph to Neo4j.

        Args:
            kg: EnhancedKG instance to sync
            clear_existing: If True, clear existing graph data before syncing

        Returns:
            Dictionary with sync statistics
        """
        with self.driver.session(database=self.config.database) as session:
            if clear_existing:
                logger.info("Clearing existing graph data...")
                session.run("MATCH (n) DETACH DELETE n")

            # Sync nodes (entities)
            nodes_synced = self._sync_nodes(session, kg)

            # Sync edges (relationships)
            edges_synced = self._sync_edges(session, kg)

            # Sync clusters (communities)
            clusters_synced = self._sync_clusters(session, kg)

            # Sync semantic similarity edges (if embeddings exist)
            similarity_edges_synced = self._sync_similarity_edges(session, kg)

            stats = {
                "nodes_synced": nodes_synced,
                "edges_synced": edges_synced,
                "clusters_synced": clusters_synced,
                "similarity_edges_synced": similarity_edges_synced,
            }

            logger.info(f"KG sync completed: {stats}")
            return stats

    def _sync_nodes(self, session, kg: EnhancedKG) -> int:
        """Sync nodes to Neo4j.

        Creates nodes with:
        - Labels: Entity type (if available) + "Entity"
        - Properties: id, type, properties dict, metadata dict
        - Embedding: stored as property (if available)
        """
        count = 0
        for node in kg.nodes.values():
            # Build labels: use entity type as label, add "Entity" as base label
            labels = ["Entity"]
            if node.type:
                labels.append(node.type)

            # Build properties
            properties = {
                "id": node.id,
                "type": node.type or "Unknown",
                "properties": node.properties,
                "metadata": node.metadata,
            }

            # First-class confidence property — surfaced for Cypher filters
            # (e.g. ``MATCH (n:Entity) WHERE n.confidence > 0.8``) instead of
            # being buried inside the metadata bag.
            if node.confidence is not None:
                properties["confidence"] = float(node.confidence)

            # Add embedding if available (store as list property)
            if node.embedding:
                properties["embedding"] = node.embedding
                properties["embedding_dimension"] = len(node.embedding)

            # Create or update node
            query = """
            MERGE (n:Entity {id: $id})
            SET n = $properties
            """
            if node.type:
                query = f"""
                MERGE (n:Entity:{node.type} {{id: $id}})
                SET n = $properties
                """

            session.run(query, id=node.id, properties=properties)
            count += 1

        logger.info(f"Synced {count} nodes to Neo4j")
        return count

    def _sync_edges(self, session, kg: EnhancedKG) -> int:
        """Sync edges (relationships) to Neo4j.

        Creates relationships with:
        - Type: relationship_type
        - Properties: relationship_detail, metadata, confidence (if available)
        - Weight: derived from metadata or default 1.0
        """
        count = 0
        for edge in kg.edges:
            # Build properties
            properties = {
                "relationship_type": edge.relationship_type,
                "relationship_detail": edge.relationship_detail,
                "metadata": edge.metadata,
            }

            # Extract weight from metadata or use default. Prefer the
            # first-class ``KGEdge.confidence`` attribute when present;
            # fall back to the legacy metadata-based location for KGs
            # built before the confidence framework existed.
            weight = edge.metadata.get("weight", 1.0)
            if edge.confidence is not None:
                weight = edge.confidence
            elif "confidence" in edge.metadata:
                weight = edge.metadata["confidence"]
            properties["weight"] = float(weight)
            # Persist confidence as a dedicated property too, so callers
            # can filter on it independently of any weighting semantics.
            if edge.confidence is not None:
                properties["confidence"] = float(edge.confidence)

            # Create relationship
            # Use relationship_type as the Neo4j relationship type. Event role
            # edges (``role:<name>``) and any other colon-bearing types are
            # collapsed to underscores so Cypher does not interpret the colon
            # as a label separator.
            rel_type = (
                edge.relationship_type.upper().replace(" ", "_").replace(":", "_")
            )
            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r = $properties
            """

            try:
                session.run(
                    query, source_id=edge.source, target_id=edge.target, properties=properties
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to sync edge {edge.source}->{edge.target}: {e}")
                # Fallback: use generic RELATES_TO relationship type
                query_fallback = """
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                MERGE (source)-[r:RELATES_TO]->(target)
                SET r = $properties
                """
                session.run(
                    query_fallback,
                    source_id=edge.source,
                    target_id=edge.target,
                    properties=properties,
                )
                count += 1

        logger.info(f"Synced {count} edges to Neo4j")
        return count

    def _sync_clusters(self, session, kg: EnhancedKG) -> int:
        """Sync clusters (communities) to Neo4j.

        Creates cluster nodes and connects entities to clusters via MEMBER_OF relationship.
        """
        count = 0
        for cluster_id, cluster in kg.clusters.items():
            # Create cluster node
            cluster_properties = {
                "cluster_id": cluster.id,
                "node_count": len(cluster.node_ids),
                "metadata": cluster.metadata,
            }

            query_cluster = """
            MERGE (c:Cluster {cluster_id: $cluster_id})
            SET c = $properties
            """

            session.run(query_cluster, cluster_id=cluster_id, properties=cluster_properties)

            # Connect entities to cluster
            for node_id in cluster.node_ids:
                query_member = """
                MATCH (e:Entity {id: $node_id})
                MATCH (c:Cluster {cluster_id: $cluster_id})
                MERGE (e)-[:MEMBER_OF]->(c)
                """
                session.run(query_member, node_id=node_id, cluster_id=cluster_id)

            count += 1

        logger.info(f"Synced {count} clusters to Neo4j")
        return count

    def _sync_similarity_edges(
        self, session, kg: EnhancedKG, similarity_threshold: float = 0.7
    ) -> int:
        """Sync semantic similarity edges between entities with embeddings.

        Creates SIMILAR_TO relationships between entities with high semantic similarity.
        Only creates edges above similarity_threshold to avoid excessive connections.

        Args:
            session: Neo4j session
            kg: EnhancedKG instance
            similarity_threshold: Minimum similarity to create edge (0.0-1.0)

        Returns:
            Number of similarity edges created
        """
        # Get all nodes with embeddings
        nodes_with_embeddings = [
            (node_id, node.embedding)
            for node_id, node in kg.nodes.items()
            if node.embedding is not None
        ]

        if len(nodes_with_embeddings) < 2:
            logger.info("Not enough nodes with embeddings for similarity edges")
            return 0

        count = 0
        # Calculate pairwise similarities and create edges
        for i, (node_id1, embedding1) in enumerate(nodes_with_embeddings):
            for node_id2, embedding2 in nodes_with_embeddings[i + 1 :]:
                similarity = self._cosine_similarity(embedding1, embedding2)

                if similarity >= similarity_threshold:
                    properties = {
                        "similarity": float(similarity),
                        "weight": float(similarity),
                        "type": "semantic_similarity",
                    }

                    query = """
                    MATCH (e1:Entity {id: $node_id1})
                    MATCH (e2:Entity {id: $node_id2})
                    MERGE (e1)-[r:SIMILAR_TO]->(e2)
                    SET r = $properties
                    """

                    session.run(query, node_id1=node_id1, node_id2=node_id2, properties=properties)
                    count += 1

        logger.info(f"Synced {count} similarity edges to Neo4j")
        return count

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0.0-1.0)
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the graph in Neo4j.

        Returns:
            Dictionary with graph statistics
        """
        with self.driver.session(database=self.config.database) as session:
            # Count nodes
            node_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]

            # Count edges
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

            # Count clusters
            cluster_count = session.run("MATCH (c:Cluster) RETURN count(c) as count").single()[
                "count"
            ]

            # Get node types
            node_types = session.run("""
                MATCH (n:Entity)
                WHERE n.type IS NOT NULL
                RETURN n.type as type, count(*) as count
                ORDER BY count DESC
            """).data()

            # Get relationship types
            rel_types = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
            """).data()

            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "cluster_count": cluster_count,
                "node_types": node_types,
                "relationship_types": rel_types,
            }
