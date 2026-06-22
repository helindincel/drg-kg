"""Unit tests for drg.extract core pipeline — no LLM / DSPy required.

Covers the pure-Python branches of extract_typed and extract_from_chunks:
- Empty / whitespace-only input short-circuit
- Input length guard (DRG_MAX_TEXT_CHARS)
- Mock-mode empty-extraction path (no LM configured)
- Schema filtering (_filter_against_schema)
- Async wrappers (extract_typed_async / extract_from_chunks_async)
- create_kgedge_from_triple (pure Python, no LLM)
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

# Stub dspy before any drg import so the module loads without a real install.
sys.modules.setdefault("dspy", MagicMock())

from drg.extract import (  # noqa: E402
    create_kgedge_from_triple,
    extract_from_chunks,
    extract_from_chunks_async,
    extract_typed,
    extract_typed_async,
)
from drg.schema import DRGSchema, Entity, Relation  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_schema() -> DRGSchema:
    return DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")],
    )


@pytest.fixture()
def rich_schema() -> DRGSchema:
    return DRGSchema(
        entities=[Entity("Person"), Entity("Company"), Entity("Product")],
        relations=[
            Relation("works_at", "Person", "Company"),
            Relation("produces", "Company", "Product"),
        ],
    )


# ---------------------------------------------------------------------------
# Empty / whitespace input short-circuit
# ---------------------------------------------------------------------------


class TestEmptyInputShortCircuit:
    def test_empty_string_returns_empty(self, simple_schema: DRGSchema):
        entities, triples = extract_typed("", simple_schema)
        assert entities == []
        assert triples == []

    def test_whitespace_only_returns_empty(self, simple_schema: DRGSchema):
        entities, triples = extract_typed("   \n\t  ", simple_schema)
        assert entities == []
        assert triples == []

    def test_empty_with_return_enriched(self, simple_schema: DRGSchema):
        result = extract_typed("", simple_schema, return_enriched=True)
        assert len(result) == 3
        assert result == ([], [], [])


# ---------------------------------------------------------------------------
# Input length guard
# ---------------------------------------------------------------------------


class TestInputLengthGuard:
    def test_text_within_limit_is_accepted(self, simple_schema: DRGSchema, monkeypatch):
        monkeypatch.setenv("DRG_MAX_TEXT_CHARS", "100")
        # Short text should not raise (will hit mock-mode empty extraction)
        result = extract_typed("short text", simple_schema)
        assert isinstance(result, tuple)

    def test_text_exceeding_limit_raises(self, simple_schema: DRGSchema, monkeypatch):
        monkeypatch.setenv("DRG_MAX_TEXT_CHARS", "10")
        with pytest.raises(ValueError, match="too long"):
            extract_typed("this text is longer than 10 characters", simple_schema)

    def test_error_message_includes_char_counts(self, simple_schema: DRGSchema, monkeypatch):
        monkeypatch.setenv("DRG_MAX_TEXT_CHARS", "5")
        with pytest.raises(ValueError, match="chars"):
            extract_typed("123456789", simple_schema)


# ---------------------------------------------------------------------------
# Mock-mode: no LM configured → empty extraction
# ---------------------------------------------------------------------------


class TestMockModeNoLM:
    def test_returns_empty_when_no_lm(self, simple_schema: DRGSchema):
        # conftest.py stubs dspy so there is no configured LM by default.
        entities, triples = extract_typed("Apple produces iPhone.", simple_schema)
        assert isinstance(entities, list)
        assert isinstance(triples, list)

    def test_extract_from_chunks_returns_empty_when_no_lm(self, simple_schema: DRGSchema):
        chunks = [{"text": "Apple produces iPhone."}, {"text": "Google makes Android."}]
        entities, triples = extract_from_chunks(chunks, simple_schema)
        assert isinstance(entities, list)
        assert isinstance(triples, list)


# ---------------------------------------------------------------------------
# create_kgedge_from_triple — pure Python, no LLM
# ---------------------------------------------------------------------------


class TestCreateKGEdgeFromTriple:
    def test_basic_triple(self):
        edge = create_kgedge_from_triple(("Apple", "produces", "iPhone"))
        assert edge.source == "Apple"
        assert edge.target == "iPhone"
        assert edge.relationship_type == "produces"

    def test_default_relationship_detail(self):
        edge = create_kgedge_from_triple(("A", "rel", "B"))
        assert edge.relationship_detail == "A rel B"

    def test_custom_relationship_detail(self):
        edge = create_kgedge_from_triple(("A", "rel", "B"), relationship_detail="custom detail")
        assert edge.relationship_detail == "custom detail"

    def test_temporal_metadata_extracted(self):
        meta = {"temporal": {"start": "2020-01", "end": "2022-12"}}
        edge = create_kgedge_from_triple(("A", "rel", "B"), enriched_metadata=meta)
        assert edge.start_time == "2020-01"
        assert edge.end_time == "2022-12"

    def test_confidence_extracted(self):
        meta = {"confidence": 0.85}
        edge = create_kgedge_from_triple(("A", "rel", "B"), enriched_metadata=meta)
        assert edge.confidence == 0.85

    def test_negation_extracted(self):
        meta = {"is_negated": True}
        edge = create_kgedge_from_triple(("A", "rel", "B"), enriched_metadata=meta)
        assert edge.is_negated is True

    def test_no_metadata_defaults(self):
        edge = create_kgedge_from_triple(("A", "rel", "B"))
        assert edge.confidence is None
        assert edge.start_time is None
        assert edge.end_time is None
        assert edge.is_negated is False


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


class TestAsyncWrappers:
    def test_extract_typed_async_returns_coroutine(self, simple_schema: DRGSchema):
        coro = extract_typed_async("text", simple_schema)
        assert asyncio.iscoroutine(coro)
        # Clean up without running
        coro.close()

    def test_extract_typed_async_runs(self, simple_schema: DRGSchema):
        result = asyncio.get_event_loop().run_until_complete(extract_typed_async("", simple_schema))
        assert isinstance(result, tuple)

    def test_extract_from_chunks_async_runs(self, simple_schema: DRGSchema):
        chunks: list[dict] = [{"text": ""}]
        result = asyncio.get_event_loop().run_until_complete(
            extract_from_chunks_async(chunks, simple_schema)
        )
        assert isinstance(result, tuple)

    def test_extract_typed_async_respects_length_limit(self, simple_schema: DRGSchema, monkeypatch):
        monkeypatch.setenv("DRG_MAX_TEXT_CHARS", "5")
        with pytest.raises(ValueError, match="too long"):
            asyncio.get_event_loop().run_until_complete(
                extract_typed_async("123456789", simple_schema)
            )


# ---------------------------------------------------------------------------
# Schema filtering (_filter_against_schema via extract_typed)
# ---------------------------------------------------------------------------


class TestSchemaFiltering:
    """Verify that extraction respects schema entity/relation constraints.

    We inject a mock extractor that returns out-of-schema entities/triples
    and check that extract_typed discards them.
    """

    def _mock_extractor_result(self, entities: list, relations: list) -> Any:
        result = MagicMock()
        result.entities = entities
        result.relations = relations
        result.enriched_relations = None
        return result

    def test_out_of_schema_entity_types_filtered(self, simple_schema: DRGSchema):
        mock_result = self._mock_extractor_result(
            entities=[("Apple", "Company"), ("Elon", "Person")],  # Person not in schema
            relations=[],
        )
        mock_extractor = Mock(return_value=mock_result)

        with (
            patch("drg.extract._get_extractor", return_value=mock_extractor),
            patch("drg.extract.dspy") as mock_dspy,
        ):
            # Simulate a configured LM so mock-mode is not triggered
            mock_dspy.settings.lm = Mock()
            entities, _triples = extract_typed("text", simple_schema)

        # Only Company is in schema; Person should be filtered
        entity_types = {etype for _, etype in entities}
        assert "Person" not in entity_types
        assert "Company" in entity_types

    def test_out_of_schema_relations_filtered(self, simple_schema: DRGSchema):
        mock_result = self._mock_extractor_result(
            entities=[("Apple", "Company"), ("iPhone", "Product")],
            relations=[
                ("Apple", "produces", "iPhone"),  # valid
                ("Apple", "invented_by", "Jobs"),  # not in schema
            ],
        )
        mock_extractor = Mock(return_value=mock_result)

        with (
            patch("drg.extract._get_extractor", return_value=mock_extractor),
            patch("drg.extract.dspy") as mock_dspy,
        ):
            mock_dspy.settings.lm = Mock()
            _entities, triples = extract_typed("text", simple_schema)

        relation_names = {r[1] for r in triples}
        assert "invented_by" not in relation_names

    def test_relation_metadata_is_preserved_for_valid_triples(self, simple_schema: DRGSchema):
        mock_result = self._mock_extractor_result(
            entities=[("Apple", "Company"), ("iPhone", "Product")],
            relations=[("Apple", "produces", "iPhone")],
        )
        mock_result.enriched_relations = [
            {
                "relation": ("Apple", "produces", "iPhone"),
                "confidence": 0.91,
                "evidence": "Apple produces iPhone.",
                "temporal": {"start": "2007", "precision": "year"},
                "is_negated": False,
                "metadata": {"chunk_id": "c1"},
            }
        ]
        mock_extractor = Mock(return_value=mock_result)

        with (
            patch("drg.extract._get_extractor", return_value=mock_extractor),
            patch("drg.extract.resolve_entities_and_relations", None),
            patch("drg.extract.dspy") as mock_dspy,
        ):
            mock_dspy.settings.lm = Mock()
            entities, triples, enriched = extract_typed(
                "Apple produces iPhone.",
                simple_schema,
                return_enriched=True,
                enable_implicit_relationships=False,
            )

        assert entities == [("Apple", "Company"), ("iPhone", "Product")]
        assert triples == [("Apple", "produces", "iPhone")]
        assert enriched[0]["confidence"] == 0.91
        assert enriched[0]["evidence"] == "Apple produces iPhone."
        assert enriched[0]["temporal"]["start"] == "2007"
        assert enriched[0]["metadata"]["chunk_id"] == "c1"
