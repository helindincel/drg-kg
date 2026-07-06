"""Top-level package API consistency tests."""

from __future__ import annotations

import sys
from pathlib import Path

import drg

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def test_base_dependencies_cover_top_level_import_runtime_requirements():
    with Path("pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.lower().startswith("numpy") for dep in dependencies)


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
