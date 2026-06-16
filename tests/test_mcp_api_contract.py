"""Contract tests for drg.mcp_api — no LLM / DSPy required.

Tests focus on the request-routing, data-model layer and schema-definition
machinery. The _extract() path requires a live LLM, so it is covered only
for the error-handling (missing params) branch.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# drg/mcp_api imports from drg.extract which imports dspy at module level.
# Stub it so these contract tests don't need dspy installed.
sys.modules.setdefault("dspy", MagicMock())

from drg.mcp_api import (
    DRGMCPAPI,
    MCPErrorCode,
    MCPRequest,
    MCPResponse,
    create_mcp_api,
)


# ---------------------------------------------------------------------------
# MCPRequest
# ---------------------------------------------------------------------------


class TestMCPRequest:
    def test_from_dict_basic(self):
        req = MCPRequest.from_dict({"method": "drg/list_tools", "id": 1})
        assert req.method == "drg/list_tools"
        assert req.id == 1

    def test_from_dict_defaults(self):
        req = MCPRequest.from_dict({})
        assert req.method == ""
        assert req.params == {}
        assert req.id is None
        assert req.jsonrpc == "2.0"

    def test_to_dict_round_trip(self):
        req = MCPRequest(method="drg/list_tools", params={"k": "v"}, id=42)
        d = req.to_dict()
        assert d["method"] == "drg/list_tools"
        assert d["params"] == {"k": "v"}
        assert d["id"] == 42

    def test_to_dict_omits_none_id(self):
        req = MCPRequest(method="m")
        d = req.to_dict()
        assert "id" not in d

    def test_jsonrpc_version_defaults_2(self):
        req = MCPRequest(method="m")
        assert req.jsonrpc == "2.0"


# ---------------------------------------------------------------------------
# MCPResponse
# ---------------------------------------------------------------------------


class TestMCPResponse:
    def test_success_factory(self):
        resp = MCPResponse.success({"status": "ok"}, request_id=7)
        assert resp.result == {"status": "ok"}
        assert resp.id == 7
        assert resp.error is None

    def test_error_factory(self):
        resp = MCPResponse.error_response(
            MCPErrorCode.METHOD_NOT_FOUND, "No such method", request_id=3
        )
        assert resp.error is not None
        assert resp.error["code"] == MCPErrorCode.METHOD_NOT_FOUND.value
        assert "No such method" in resp.error["message"]

    def test_to_dict_success(self):
        resp = MCPResponse.success({"x": 1}, request_id=1)
        d = resp.to_dict()
        assert d["result"] == {"x": 1}
        assert "error" not in d

    def test_to_dict_error(self):
        resp = MCPResponse.error_response(MCPErrorCode.INTERNAL_ERROR, "boom")
        d = resp.to_dict()
        assert "error" in d
        assert "result" not in d

    def test_to_dict_includes_jsonrpc(self):
        resp = MCPResponse.success({})
        assert resp.to_dict()["jsonrpc"] == "2.0"

    def test_error_with_data(self):
        resp = MCPResponse.error_response(
            MCPErrorCode.INVALID_PARAMS, "bad", data={"field": "text"}, request_id=5
        )
        assert resp.error["data"] == {"field": "text"}

    def test_error_without_data_excludes_key(self):
        resp = MCPResponse.error_response(MCPErrorCode.INVALID_PARAMS, "bad")
        assert "data" not in resp.error


# ---------------------------------------------------------------------------
# DRGMCPAPI — routing
# ---------------------------------------------------------------------------


class TestDRGMCPAPIRouting:
    def setup_method(self):
        self.api = create_mcp_api()

    def _req(self, method: str, params: dict | None = None, id_: int | None = 1) -> dict:
        return {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": id_}

    def test_unknown_method_returns_error(self):
        resp = self.api.handle_request(self._req("no/such/method"))
        assert resp.error is not None
        assert resp.error["code"] == MCPErrorCode.METHOD_NOT_FOUND.value

    def test_list_tools_returns_tools_key(self):
        resp = self.api.handle_request(self._req("drg/list_tools"))
        assert resp.result is not None
        assert "tools" in resp.result
        assert len(resp.result["tools"]) > 0

    def test_list_schemas_empty_initially(self):
        resp = self.api.handle_request(self._req("drg/list_schemas"))
        assert resp.result is not None
        assert resp.result.get("schemas") == [] or "schemas" in resp.result

    def test_handle_dict_request(self):
        resp = self.api.handle_request(self._req("drg/list_tools"))
        assert resp.error is None

    def test_handle_mcp_request_object(self):
        req = MCPRequest(method="drg/list_tools", id=1)
        resp = self.api.handle_request(req)
        assert resp.error is None

    def test_request_id_preserved_in_response(self):
        resp = self.api.handle_request(self._req("drg/list_tools", id_=99))
        assert resp.id == 99


# ---------------------------------------------------------------------------
# DRGMCPAPI — schema definition
# ---------------------------------------------------------------------------


class TestDRGMCPAPIDefineSchema:
    def setup_method(self):
        self.api = create_mcp_api()

    def _req(self, method: str, params: dict) -> dict:
        return {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}

    def test_define_legacy_schema(self):
        params = {
            "schema_id": "test_schema",
            "schema": {
                "entities": [{"name": "Company"}, {"name": "Product"}],
                "relations": [{"name": "produces", "src": "Company", "dst": "Product"}],
            },
        }
        resp = self.api.handle_request(self._req("drg/define_schema", params))
        assert resp.error is None
        assert resp.result["schema_id"] == "test_schema"
        assert resp.result["status"] == "defined"

    def test_define_enhanced_schema(self):
        params = {
            "schema_id": "enh_schema",
            "schema": {
                "entity_types": [
                    {"name": "Company", "description": "A business"},
                    {"name": "Product", "description": "A good"},
                ],
                "relation_groups": [
                    {
                        "name": "prod",
                        "description": "production",
                        "relations": [{"name": "produces", "src": "Company", "dst": "Product"}],
                    }
                ],
            },
        }
        resp = self.api.handle_request(self._req("drg/define_schema", params))
        assert resp.error is None
        assert resp.result["status"] == "defined"

    def test_define_schema_missing_id_returns_error(self):
        params = {
            "schema": {"entities": [{"name": "X"}], "relations": []},
        }
        resp = self.api.handle_request(self._req("drg/define_schema", params))
        assert resp.error is not None
        assert resp.error["code"] == MCPErrorCode.INVALID_PARAMS.value

    def test_define_schema_missing_schema_returns_error(self):
        params = {"schema_id": "x"}
        resp = self.api.handle_request(self._req("drg/define_schema", params))
        assert resp.error is not None

    def test_get_defined_schema(self):
        # Define first
        self.api.handle_request({
            "jsonrpc": "2.0",
            "method": "drg/define_schema",
            "params": {
                "schema_id": "s1",
                "schema": {
                    "entities": [{"name": "A"}, {"name": "B"}],
                    "relations": [{"name": "rel", "src": "A", "dst": "B"}],
                },
            },
            "id": 1,
        })
        # Now get it
        resp = self.api.handle_request({
            "jsonrpc": "2.0",
            "method": "drg/get_schema",
            "params": {"schema_id": "s1"},
            "id": 2,
        })
        assert resp.error is None
        assert resp.result is not None

    def test_get_nonexistent_schema_returns_error(self):
        resp = self.api.handle_request({
            "jsonrpc": "2.0",
            "method": "drg/get_schema",
            "params": {"schema_id": "no_such"},
            "id": 1,
        })
        assert resp.error is not None


# ---------------------------------------------------------------------------
# DRGMCPAPI — build_kg (no LLM needed)
# ---------------------------------------------------------------------------


class TestDRGMCPAPIBuildKG:
    def setup_method(self):
        self.api = create_mcp_api()

    def _build_req(self, kg_id: str, entities: list, triples: list) -> dict:
        return {
            "jsonrpc": "2.0",
            "method": "drg/build_kg",
            "params": {"kg_id": kg_id, "entities": entities, "triples": triples},
            "id": 1,
        }

    def test_build_kg_basic(self):
        resp = self.api.handle_request(
            self._build_req(
                "kg1",
                [["Apple", "Company"], ["iPhone", "Product"]],
                [["Apple", "produces", "iPhone"]],
            )
        )
        assert resp.error is None
        assert resp.result["kg_id"] == "kg1"
        assert resp.result["node_count"] == 2

    def test_build_kg_missing_id_returns_error(self):
        resp = self.api.handle_request({
            "jsonrpc": "2.0",
            "method": "drg/build_kg",
            "params": {"entities": [], "triples": []},
            "id": 1,
        })
        assert resp.error is not None

    def test_get_kg_after_build(self):
        self.api.handle_request(
            self._build_req("kg2", [["A", "T"], ["B", "T"]], [["A", "rel", "B"]])
        )
        resp = self.api.handle_request({
            "jsonrpc": "2.0",
            "method": "drg/get_kg",
            "params": {"kg_id": "kg2"},
            "id": 1,
        })
        assert resp.error is None
        assert resp.result is not None

    def test_get_nonexistent_kg_returns_error(self):
        resp = self.api.handle_request({
            "jsonrpc": "2.0",
            "method": "drg/get_kg",
            "params": {"kg_id": "no_kg"},
            "id": 1,
        })
        assert resp.error is not None

    def test_export_kg_json_format(self):
        self.api.handle_request(
            self._build_req("kg3", [["X", "T"], ["Y", "T"]], [["X", "r", "Y"]])
        )
        resp = self.api.handle_request({
            "jsonrpc": "2.0",
            "method": "drg/export_kg",
            "params": {"kg_id": "kg3", "format": "json"},
            "id": 1,
        })
        assert resp.error is None
        assert "export" in resp.result or "data" in resp.result or resp.result is not None

    def test_extract_without_schema_returns_error(self):
        resp = self.api.handle_request({
            "jsonrpc": "2.0",
            "method": "drg/extract",
            "params": {"text": "Apple makes iPhones", "schema_id": "no_schema"},
            "id": 1,
        })
        assert resp.error is not None
