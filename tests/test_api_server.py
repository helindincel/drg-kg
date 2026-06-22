"""Contract tests for drg.api.server — uses FastAPI TestClient, no live LLM.

Covers:
- GET /api/graph          happy-path + 404 when no KG loaded
- GET /api/graph/stats    happy-path
- GET /api/communities    happy-path
- POST /api/query         happy-path, empty-query 422, long-query 422
- GET /api/visualization  cytoscape, vis-network, d3 formats + bad format 400
- Authentication          401 when DRG_API_KEY is set and key is wrong/missing
"""

from __future__ import annotations

import pytest

# Skip the entire module if fastapi / httpx (required by TestClient) are not installed.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from drg.api.server import create_app
from drg.graph.kg_core import Cluster, EnhancedKG, KGEdge, KGNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_api_dotenv_loading(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DRG_API_LOAD_ENV", "0")


def _make_sample_kg() -> EnhancedKG:
    """Return a small but complete EnhancedKG."""
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Alice", type="Person"))
    kg.add_node(KGNode(id="Acme", type="Company"))
    kg.add_node(KGNode(id="Widget", type="Product"))
    kg.add_edge(
        KGEdge(
            source="Alice",
            target="Acme",
            relationship_type="works_at",
            relationship_detail="Alice works at Acme",
        )
    )
    kg.add_edge(
        KGEdge(
            source="Acme",
            target="Widget",
            relationship_type="produces",
            relationship_detail="Acme produces Widget",
        )
    )
    kg.add_cluster(Cluster(id="c0", node_ids={"Alice", "Acme", "Widget"}))
    return kg


@pytest.fixture()
def client() -> TestClient:
    """TestClient with a pre-loaded KG and no API-key requirement."""
    app = create_app(kg=_make_sample_kg())
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def client_no_kg() -> TestClient:
    """TestClient with *no* KG loaded."""
    app = create_app()
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# GET /api/graph
# ---------------------------------------------------------------------------


class TestGetGraph:
    def test_returns_200_with_kg_loaded(self, client: TestClient):
        resp = client.get("/api/graph")
        assert resp.status_code == 200

    def test_response_has_nodes_edges_clusters(self, client: TestClient):
        data = client.get("/api/graph").json()
        assert "nodes" in data
        assert "edges" in data
        assert "clusters" in data

    def test_nodes_contain_expected_ids(self, client: TestClient):
        nodes = client.get("/api/graph").json()["nodes"]
        ids = {n["id"] for n in nodes}
        assert {"Alice", "Acme", "Widget"}.issubset(ids)

    def test_edges_present(self, client: TestClient):
        edges = client.get("/api/graph").json()["edges"]
        assert len(edges) == 2

    def test_returns_404_when_no_kg(self, client_no_kg: TestClient):
        assert client_no_kg.get("/api/graph").status_code == 404


class TestOperationalEndpoints:
    def test_healthz_returns_ok(self, client_no_kg: TestClient):
        resp = client_no_kg.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readyz_reports_loaded_graph(self, client: TestClient):
        data = client.get("/readyz").json()
        assert data["status"] == "ready"
        assert data["kg_loaded"] is True

    def test_readyz_degraded_without_graph(self, client_no_kg: TestClient):
        data = client_no_kg.get("/readyz").json()
        assert data["status"] == "degraded"
        assert data["kg_loaded"] is False

    def test_request_id_header_is_echoed(self, client: TestClient):
        resp = client.get("/api/graph", headers={"X-Request-ID": "req-123"})
        assert resp.headers["X-Request-ID"] == "req-123"


# ---------------------------------------------------------------------------
# GET /api/graph/stats
# ---------------------------------------------------------------------------


class TestGetGraphStats:
    def test_returns_200(self, client: TestClient):
        assert client.get("/api/graph/stats").status_code == 200

    def test_stats_structure(self, client: TestClient):
        data = client.get("/api/graph/stats").json()
        assert data["node_count"] == 3
        assert data["edge_count"] == 2
        assert "node_types" in data
        assert "relationship_types" in data

    def test_returns_404_when_no_kg(self, client_no_kg: TestClient):
        assert client_no_kg.get("/api/graph/stats").status_code == 404


class TestExtractEndpoint:
    def test_extract_builds_graph_and_stores_it(self, monkeypatch: pytest.MonkeyPatch):
        import drg.extract as extract_mod

        def fake_extract(text, schema):
            assert "TechCorp" in text
            return (
                [("TechCorp", "Company"), ("Jane Doe", "Person")],
                [("TechCorp", "founded_by", "Jane Doe")],
            )

        monkeypatch.setattr(extract_mod, "extract_typed", fake_extract)
        app = create_app()
        c = TestClient(app)

        resp = c.post(
            "/api/extract",
            json={
                "text": "TechCorp was founded by Jane Doe.",
                "schema": {
                    "entity_types": [
                        {"name": "Company", "description": "Companies"},
                        {"name": "Person", "description": "People"},
                    ],
                    "relation_groups": [
                        {
                            "name": "founding",
                            "relations": [
                                {"name": "founded_by", "src": "Company", "dst": "Person"}
                            ],
                        }
                    ],
                },
                "model": "ollama_chat/llama3",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"]["nodes"] == 2
        assert data["counts"]["edges"] == 1
        assert data["stored"] is True
        assert c.get("/api/graph").status_code == 200

    def test_extract_can_skip_storing_graph(self, monkeypatch: pytest.MonkeyPatch):
        import drg.extract as extract_mod

        monkeypatch.setattr(
            extract_mod,
            "extract_typed",
            lambda text, schema: ([("Acme", "Company")], []),
        )
        c = TestClient(create_app())

        resp = c.post(
            "/api/extract",
            json={
                "text": "Acme builds widgets.",
                "store_graph": False,
                "model": "ollama_chat/llama3",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["stored"] is False
        assert c.get("/api/graph").status_code == 404

    def test_extract_without_provider_credentials_returns_400(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        for key in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "DRG_MODEL",
        ):
            monkeypatch.delenv(key, raising=False)
        c = TestClient(create_app())

        resp = c.post("/api/extract", json={"text": "Acme builds widgets."})

        assert resp.status_code == 400
        assert "API key" in resp.json()["detail"]

    def test_extract_empty_text_returns_422(self, client_no_kg: TestClient):
        resp = client_no_kg.post("/api/extract", json={"text": "   "})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/communities
# ---------------------------------------------------------------------------


class TestGetCommunities:
    def test_returns_200(self, client: TestClient):
        assert client.get("/api/communities").status_code == 200

    def test_response_has_clusters_key(self, client: TestClient):
        data = client.get("/api/communities").json()
        assert "clusters" in data
        assert isinstance(data["clusters"], list)

    def test_returns_404_when_no_kg(self, client_no_kg: TestClient):
        assert client_no_kg.get("/api/communities").status_code == 404


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------


class TestPostQuery:
    def test_happy_path_returns_200(self, client: TestClient):
        resp = client.post("/api/query", json={"query": "Alice"})
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client: TestClient):
        data = client.post("/api/query", json={"query": "Acme"}).json()
        assert "query" in data
        assert "provenance_id" in data
        removed_context_field = "query" + "_context"
        assert removed_context_field not in data

    def test_empty_query_returns_422(self, client: TestClient):
        resp = client.post("/api/query", json={"query": ""})
        assert resp.status_code == 422

    def test_whitespace_only_query_returns_422(self, client: TestClient):
        resp = client.post("/api/query", json={"query": "   "})
        assert resp.status_code == 422

    def test_query_too_long_returns_422(self, client: TestClient):
        # QueryRequest.query has max_length=2000
        resp = client.post("/api/query", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_query_with_null_byte_returns_422(self, client: TestClient):
        resp = client.post("/api/query", json={"query": "Alice\x00inject"})
        assert resp.status_code == 422

    def test_no_kg_returns_404(self, client_no_kg: TestClient):
        resp = client_no_kg.post("/api/query", json={"query": "Alice"})
        assert resp.status_code == 404

    def test_k_entities_out_of_range_returns_422(self, client: TestClient):
        resp = client.post("/api/query", json={"query": "Alice", "k_entities": 0})
        assert resp.status_code == 422

    def test_k_entities_too_large_returns_422(self, client: TestClient):
        resp = client.post("/api/query", json={"query": "Alice", "k_entities": 200})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/visualization/{format}
# ---------------------------------------------------------------------------


class TestGetVisualization:
    def test_cytoscape_returns_200(self, client: TestClient):
        resp = client.get("/api/visualization/cytoscape")
        assert resp.status_code == 200

    def test_cytoscape_has_elements(self, client: TestClient):
        data = client.get("/api/visualization/cytoscape").json()
        assert "elements" in data

    def test_vis_network_returns_200(self, client: TestClient):
        assert client.get("/api/visualization/vis-network").status_code == 200

    def test_d3_returns_200(self, client: TestClient):
        assert client.get("/api/visualization/d3").status_code == 200

    def test_unsupported_format_returns_400(self, client: TestClient):
        assert client.get("/api/visualization/graphml").status_code == 400

    def test_returns_404_when_no_kg(self, client_no_kg: TestClient):
        assert client_no_kg.get("/api/visualization/cytoscape").status_code == 404


# ---------------------------------------------------------------------------
# Authentication (DRG_API_KEY)
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    def test_no_auth_required_when_env_var_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("DRG_API_KEY", raising=False)
        app = create_app(kg=_make_sample_kg())
        c = TestClient(app)
        assert c.get("/api/graph").status_code == 200

    def test_valid_key_grants_access(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DRG_API_KEY", "secret-token")
        app = create_app(kg=_make_sample_kg())
        c = TestClient(app)
        assert c.get("/api/graph", headers={"X-API-Key": "secret-token"}).status_code == 200

    def test_wrong_key_returns_401(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DRG_API_KEY", "secret-token")
        app = create_app(kg=_make_sample_kg())
        c = TestClient(app)
        assert c.get("/api/graph", headers={"X-API-Key": "wrong"}).status_code == 401

    def test_missing_key_returns_401(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DRG_API_KEY", "secret-token")
        app = create_app(kg=_make_sample_kg())
        c = TestClient(app)
        assert c.get("/api/graph").status_code == 401


class TestNeo4jEndpoints:
    def test_sync_dry_run_returns_plan(self):
        from drg.graph.neo4j_exporter import Neo4jConfig

        app = create_app(
            kg=_make_sample_kg(),
            neo4j_config=Neo4jConfig(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
            ),
        )
        c = TestClient(app)
        resp = c.post("/api/neo4j/sync?dry_run=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "dry_run"
        assert data["plan"]["node_count"] == 3
        assert "WORKS_AT" in data["plan"]["relationship_types"]

    def test_sync_returns_config_errors_before_connecting(self):
        from drg.graph.neo4j_exporter import Neo4jConfig

        app = create_app(
            kg=_make_sample_kg(),
            neo4j_config=Neo4jConfig(uri="", user="", password=""),
        )
        c = TestClient(app)
        resp = c.post("/api/neo4j/sync")
        assert resp.status_code == 400
        assert "errors" in resp.json()["detail"]
