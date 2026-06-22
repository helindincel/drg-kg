from __future__ import annotations

import json
import sys

import pytest

from drg import cli as cli_mod
from drg.graph.builders import build_enhanced_kg
from drg.graph.incremental import GraphMerger
from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode
from drg.graph.versioning import create_snapshot, diff_versions, list_versions, rollback_to_version
from drg.query import GraphQuery


def test_builder_attaches_structured_provenance_and_query_reads_it():
    kg = build_enhanced_kg(
        entities_typed=[("Alice", "Person"), ("Acme", "Company")],
        triples=[("Alice", "works_at", "Acme")],
        source_text="Alice works at Acme. Bob works elsewhere.",
        document_id="doc_1",
    )

    edge = kg.edges[0]
    provenance = edge.metadata["provenance"]
    assert provenance["document_id"] == "doc_1"
    assert provenance["sentence_id"] == "s0"
    assert provenance["snippet"] == "Alice works at Acme."

    bundle = GraphQuery(kg).evidence_for("Alice", "works_at", "Acme")
    assert bundle.source_documents == ("doc_1",)
    assert bundle.evidence[0].snippet == "Alice works at Acme."


def test_versioning_snapshot_diff_and_rollback(tmp_path):
    graph_path = tmp_path / "global_kg.json"

    base = EnhancedKG()
    base.add_node(KGNode(id="Alice", type="Person"))
    GraphMerger().merge(base, EnhancedKG(), document_id="init")
    base.save_json(str(graph_path))
    v1 = create_snapshot(base, graph_path, operation="init", document_id="init")

    base.add_node(KGNode(id="Acme", type="Company"))
    base.add_edge(KGEdge("Alice", "Acme", "works_at", "Alice works at Acme"))
    GraphMerger().merge(base, EnhancedKG(), document_id="doc_2")
    base.save_json(str(graph_path))
    v2 = create_snapshot(base, graph_path, operation="merge", document_id="doc_2")

    versions = list_versions(graph_path)
    assert [v.version_id for v in versions] == [v1.version_id, v2.version_id]

    diff = diff_versions(graph_path, v1.version_id, v2.version_id)
    assert "Acme" in diff.added_nodes

    rollback = rollback_to_version(graph_path, v1.version_id)
    assert rollback.operation == "rollback"
    restored = json.loads(graph_path.read_text())
    assert {n["id"] for n in restored["nodes"]} == {"Alice"}


def test_cli_versions_list_diff_and_rollback(monkeypatch, tmp_path, capsys):
    graph_path = tmp_path / "global_kg.json"
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Alice", type="Person"))
    kg.metadata = {"version": 1}
    kg.save_json(str(graph_path))
    v1 = create_snapshot(kg, graph_path, operation="init")

    kg.add_node(KGNode(id="Acme", type="Company"))
    kg.metadata["version"] = 2
    kg.save_json(str(graph_path))
    v2 = create_snapshot(kg, graph_path, operation="merge")

    monkeypatch.setattr(sys, "argv", ["drg", "versions", "list", str(graph_path), "--json"])
    cli_mod.main()
    listed = json.loads(capsys.readouterr().out)
    assert [v["version_id"] for v in listed] == [v1.version_id, v2.version_id]

    monkeypatch.setattr(
        sys,
        "argv",
        ["drg", "versions", "diff", str(graph_path), v1.version_id, v2.version_id, "--json"],
    )
    cli_mod.main()
    diff = json.loads(capsys.readouterr().out)
    assert diff["summary"]["added_nodes"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["drg", "versions", "rollback", str(graph_path), v1.version_id],
    )
    cli_mod.main()
    assert "Rolled back" in capsys.readouterr().out
    restored = json.loads(graph_path.read_text())
    assert {n["id"] for n in restored["nodes"]} == {"Alice"}


def test_api_update_versions_and_provenance(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    import drg.extract as extract_mod
    from fastapi.testclient import TestClient

    from drg.api.server import create_app

    calls = iter(
        [
            ([("Alice", "Person"), ("Acme", "Company")], [("Alice", "works_at", "Acme")]),
            ([("Alice", "Person"), ("Widget", "Product")], [("Acme", "produces", "Widget")]),
        ]
    )

    def fake_extract(text, schema):
        return next(calls)

    monkeypatch.setattr(extract_mod, "extract_typed", fake_extract)
    client = TestClient(create_app())

    first = client.post(
        "/api/graph/update",
        json={"text": "Alice works at Acme.", "model": "ollama_chat/llama3", "document_id": "doc_1"},
    )
    assert first.status_code == 200
    assert first.json()["version"]["version_id"] == "v1"

    second = client.post(
        "/api/graph/update",
        json={"text": "Acme produces Widget.", "model": "ollama_chat/llama3", "document_id": "doc_2"},
    )
    assert second.status_code == 200

    versions = client.get("/api/graph/versions").json()["versions"]
    assert len(versions) == 2

    edge_prov = client.get(
        "/api/provenance/edge",
        params={"source": "Alice", "relationship_type": "works_at", "target": "Acme"},
    )
    assert edge_prov.status_code == 200
    assert edge_prov.json()["provenance"][0]["provenance"]["document_id"] == "doc_1"

    diff = client.get(f"/api/graph/versions/{versions[1]['version_id']}/diff")
    assert diff.status_code == 200
    assert diff.json()["summary"]["added_nodes"] >= 1

    rollback = client.post(f"/api/graph/versions/{versions[0]['version_id']}/rollback")
    assert rollback.status_code == 200
    ids = {node["id"] for node in rollback.json()["graph"]["nodes"]}
    assert ids == {"Alice", "Acme"}
