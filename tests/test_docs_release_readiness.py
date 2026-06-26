from __future__ import annotations

from pathlib import Path


def test_mcp_integration_docs_only_reference_existing_public_server_module():
    docs = Path("docs/mcp_integration.md").read_text(encoding="utf-8")

    assert "drg.mcp_server" in docs
    assert "drg.mcp_api" not in docs
