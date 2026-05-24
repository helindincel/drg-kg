"""Regression tests anchored on hand-written fixtures.

These tests do NOT require any LLM API key. They exist to lock down the JSON
shape of `EnhancedDRGSchema` and `KG` so that accidental schema/serialization
breakage is caught early in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drg import KG, EnhancedDRGSchema

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


@pytest.fixture(scope="module")
def minimal_schema_dict() -> dict:
    return _load_fixture("minimal_schema.json")


@pytest.fixture(scope="module")
def minimal_kg_dict() -> dict:
    return _load_fixture("minimal_kg.json")


class TestMinimalSchemaFixture:
    """Lock down `EnhancedDRGSchema.from_dict` against a known-good fixture."""

    def test_loads_without_error(self, minimal_schema_dict: dict) -> None:
        schema = EnhancedDRGSchema.from_dict(minimal_schema_dict)
        assert schema is not None

    def test_entity_types_preserved(self, minimal_schema_dict: dict) -> None:
        schema = EnhancedDRGSchema.from_dict(minimal_schema_dict)
        names = {et.name for et in schema.entity_types}
        assert names == {"Company", "Product"}

    def test_relation_lookup(self, minimal_schema_dict: dict) -> None:
        schema = EnhancedDRGSchema.from_dict(minimal_schema_dict)
        assert schema.is_valid_relation("produces", "Company", "Product")
        assert not schema.is_valid_relation("produces", "Product", "Company")

    def test_roundtrip(self, minimal_schema_dict: dict) -> None:
        """Schema -> dict -> Schema should preserve all entity/relation names."""
        schema = EnhancedDRGSchema.from_dict(minimal_schema_dict)
        dumped = schema.to_dict()
        reloaded = EnhancedDRGSchema.from_dict(dumped)
        assert {et.name for et in reloaded.entity_types} == {et.name for et in schema.entity_types}
        assert len(reloaded.get_all_relations()) == len(schema.get_all_relations())


class TestMinimalKGFixture:
    """Lock down `KG` JSON shape against the reference fixture."""

    def test_kg_from_typed_matches_fixture(self, minimal_kg_dict: dict) -> None:
        entities = [("Apple", "Company"), ("iPhone", "Product")]
        triples = [("Apple", "produces", "iPhone")]

        kg = KG.from_typed(entities, triples)
        result = json.loads(kg.to_json())

        assert result == minimal_kg_dict

    def test_kg_nodes_have_types(self, minimal_kg_dict: dict) -> None:
        node_types = {n["id"]: n.get("type") for n in minimal_kg_dict["nodes"]}
        assert node_types["Apple"] == "Company"
        assert node_types["iPhone"] == "Product"

    def test_kg_edges_are_directed(self, minimal_kg_dict: dict) -> None:
        edges = minimal_kg_dict["edges"]
        assert len(edges) == 1
        edge = edges[0]
        assert edge["source"] == "Apple"
        assert edge["target"] == "iPhone"
        assert edge["type"] == "produces"
