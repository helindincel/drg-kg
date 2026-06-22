"""Unit tests for Neo4j exporter UX helpers."""

from __future__ import annotations

from drg.graph.kg_core import Cluster, EnhancedKG, KGEdge, KGNode
from drg.graph.neo4j_exporter import (
    Neo4jConfig,
    build_neo4j_sync_plan,
    sanitize_neo4j_identifier,
    validate_neo4j_config,
)


def _kg() -> EnhancedKG:
    kg = EnhancedKG()
    kg.add_node(KGNode(id="a", type="Company; DROP", embedding=[0.1, 0.2]))
    kg.add_node(KGNode(id="b", type="Product", embedding=[0.1, 0.3]))
    kg.add_edge(
        KGEdge(
            source="a",
            target="b",
            relationship_type="role:manufacturer",
            relationship_detail="a manufactures b",
        )
    )
    kg.add_cluster(Cluster(id="c1", node_ids={"a", "b"}))
    return kg


def test_sanitize_neo4j_identifier_removes_cypher_unsafe_chars():
    assert sanitize_neo4j_identifier("role:manufacturer") == "ROLE_MANUFACTURER"
    assert sanitize_neo4j_identifier("Company; DROP") == "COMPANY_DROP"
    assert sanitize_neo4j_identifier("123 type") == "_123_TYPE"
    assert sanitize_neo4j_identifier("", fallback="relates_to") == "RELATES_TO"


def test_validate_neo4j_config_reports_all_missing_fields():
    errors = validate_neo4j_config(Neo4jConfig(uri="", user="", password="", database=""))
    assert "Neo4j URI is required" in errors
    assert "Neo4j user is required" in errors
    assert "Neo4j password is required" in errors
    assert "Neo4j database is required" in errors


def test_validate_neo4j_config_accepts_bolt_uri():
    assert (
        validate_neo4j_config(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="password")
        )
        == []
    )


def test_build_neo4j_sync_plan_is_write_free_and_sanitized():
    plan = build_neo4j_sync_plan(_kg())
    data = plan.to_dict()
    assert data["node_count"] == 2
    assert data["edge_count"] == 1
    assert data["cluster_count"] == 1
    assert data["similarity_edge_candidates"] == 1
    assert "COMPANY_DROP" in data["labels"]
    assert "ROLE_MANUFACTURER" in data["relationship_types"]
    assert "MEMBER_OF" in data["relationship_types"]
