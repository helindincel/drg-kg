"""Top-level package API consistency tests."""

from __future__ import annotations

import drg


def test_type_checked_extraction_exports_are_runtime_accessible():
    names = [
        "create_kgedge_from_triple",
        "extract_from_chunks",
        "extract_from_chunks_async",
        "extract_typed_async",
    ]

    for name in names:
        assert name in drg.__all__
        assert getattr(drg, name) is not None
