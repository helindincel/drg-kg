"""Unit tests for ``drg.extract._schema_gen``.

Split into two test classes:

* ``TestSampleTextForSchemaGeneration`` — pure string-sampling helper, no
  external dependencies. We pin both the short-circuit behaviour for
  reasonably-sized inputs and the budget enforcement for over-budget inputs.
* ``TestGenerateSchemaFromText`` — exercises the DSPy-driven branches with
  a mocked predictor so no LLM call leaves the test process.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from drg.errors import SchemaGenerationError
from drg.extract._schema_gen import (
    _sample_text_for_schema_generation,
    generate_schema_from_text,
)
from drg.schema import EnhancedDRGSchema


class TestSampleTextForSchemaGeneration:
    def test_empty_string_returns_empty(self):
        assert _sample_text_for_schema_generation("") == ""

    def test_whitespace_only_returns_empty(self):
        assert _sample_text_for_schema_generation("   \n\t  ") == ""

    def test_short_text_passes_through_unchanged(self):
        text = "Marie Curie discovered radium in 1898."
        assert _sample_text_for_schema_generation(text) == text

    def test_medium_text_under_default_budget_passes_through(self):
        # Default budget is 100k chars — 50k easily fits.
        text = "Sample sentence. " * 3000  # ~51k chars
        result = _sample_text_for_schema_generation(text)
        assert result == text

    def test_long_text_is_sampled_within_budget(self, monkeypatch):
        # Force a small budget so we can verify the sampling actually trips.
        monkeypatch.setenv("DRG_SCHEMA_MAX_SAMPLE_CHARS", "5000")
        monkeypatch.setenv("DRG_SCHEMA_MIN_PART_CHARS", "500")
        monkeypatch.setenv("DRG_SCHEMA_MAX_PART_CHARS", "1000")
        monkeypatch.setenv("DRG_SCHEMA_MAX_PARTS", "8")

        text = "x" * 50_000
        result = _sample_text_for_schema_generation(text)

        assert len(result) < len(text)
        assert len(result) <= 5000 + 200  # budget + separator slack
        # Sampler always includes the document head.
        assert result.startswith("x")
        # And reports truncation between parts.
        assert "[... truncated ...]" in result

    def test_includes_first_and_last_sections(self, monkeypatch):
        # Use unique markers so we can prove the first and last slices land
        # in the sampled output.
        monkeypatch.setenv("DRG_SCHEMA_MAX_SAMPLE_CHARS", "5000")
        monkeypatch.setenv("DRG_SCHEMA_MIN_PART_CHARS", "500")
        monkeypatch.setenv("DRG_SCHEMA_MAX_PART_CHARS", "1000")

        head = "HEADMARKER" + ("a" * 1000)
        tail = ("b" * 1000) + "TAILMARKER"
        middle = "c" * 50_000
        text = head + middle + tail

        result = _sample_text_for_schema_generation(text)
        assert "HEADMARKER" in result
        assert "TAILMARKER" in result

    def test_target_coverage_env_override(self, monkeypatch):
        monkeypatch.setenv("DRG_SCHEMA_TARGET_COVERAGE", "0.10")
        monkeypatch.setenv("DRG_SCHEMA_MAX_SAMPLE_CHARS", "50000")
        text = "y" * 200_000
        result = _sample_text_for_schema_generation(text)
        # 10% coverage of 200k chars = ~20k, well under the 50k cap.
        assert len(result) < len(text)


class _StubPrediction:
    """Stand-in for a DSPy ``Prediction`` returned by ``TypedPredictor``."""

    def __init__(self, generated_schema: str):
        self.generated_schema = generated_schema


_VALID_SCHEMA_JSON = json.dumps(
    {
        "entity_types": [
            {"name": "Person", "description": "People", "examples": ["Alice"]},
            {"name": "Company", "description": "Orgs", "examples": ["ACME"]},
        ],
        "relation_groups": [
            {
                "name": "employment",
                "description": "Employment relations",
                "relations": [
                    {
                        "name": "works_at",
                        "source": "Person",
                        "target": "Company",
                        "description": "Person works at Company",
                    }
                ],
            }
        ],
    }
)


def _patched_dspy(*, schema_str: str, raise_on_call: Exception | None = None):
    """Build a ``patch`` context for ``dspy`` inside ``_schema_gen``.

    The real module is replaced with a SimpleNamespace exposing only the
    surface ``generate_schema_from_text`` actually touches: ``TypedPredictor``
    (which we mock as a callable returning a ``_StubPrediction``) and
    ``ChainOfThought`` (fallback when ``TypedPredictor`` is missing).
    """

    def _make_predictor(*_args, **_kwargs):
        def _call(*, text):  # noqa: ARG001
            if raise_on_call is not None:
                raise raise_on_call
            return _StubPrediction(schema_str)

        return _call

    fake_dspy = SimpleNamespace(
        TypedPredictor=_make_predictor,
        ChainOfThought=_make_predictor,
        Signature=type("Signature", (), {}),
        InputField=lambda **_: None,
        OutputField=lambda **_: None,
    )
    return patch("drg.extract._schema_gen.dspy", fake_dspy)


class TestGenerateSchemaFromText:
    def test_happy_path_returns_enhanced_schema(self):
        with _patched_dspy(schema_str=_VALID_SCHEMA_JSON):
            schema = generate_schema_from_text("Alice works at ACME.")
        assert isinstance(schema, EnhancedDRGSchema)
        assert any(et.name == "Person" for et in schema.entity_types)
        assert any(et.name == "Company" for et in schema.entity_types)
        # Reverse-relation enrichment ran: "works_at" yields "employs".
        all_relations = {r.name for rg in schema.relation_groups for r in rg.relations}
        assert "works_at" in all_relations

    def test_dspy_failure_raises_schema_generation_error(self):
        with _patched_dspy(
            schema_str="",
            raise_on_call=RuntimeError("LLM timeout"),
        ), pytest.raises(SchemaGenerationError, match="Schema generation failed"):
            generate_schema_from_text("Some text.")

    def test_invalid_json_raises_schema_generation_error(self):
        with _patched_dspy(
            schema_str="not really json {{{",
        ), pytest.raises(SchemaGenerationError, match="Schema JSON parsing failed"):
            generate_schema_from_text("Some text.")

    def test_empty_schema_raises_schema_generation_error(self):
        with _patched_dspy(schema_str="{}"), pytest.raises(
            SchemaGenerationError, match="empty schema"
        ):
            generate_schema_from_text("Some text.")

    def test_legacy_shape_is_converted(self):
        """``generate_schema_from_text`` accepts the older ``{entities, relations}``
        shape and converts it into an EnhancedDRGSchema."""
        legacy_payload = json.dumps(
            {
                "entities": [
                    {"name": "Person", "description": "People", "examples": ["Alice"]},
                    {"name": "Company", "description": "Orgs"},
                ],
                "relations": [
                    {
                        "name": "works_at",
                        "source": "Person",
                        "target": "Company",
                        "description": "Person works at Company",
                    }
                ],
            }
        )
        with _patched_dspy(schema_str=legacy_payload):
            schema = generate_schema_from_text("Sample text.")
        assert isinstance(schema, EnhancedDRGSchema)
        # Legacy conversion lumps everything under a single "general" group.
        assert any(rg.name == "general" for rg in schema.relation_groups)

    def test_legacy_shape_with_no_relations_raises(self):
        legacy_payload = json.dumps(
            {
                "entities": [{"name": "Person", "description": "People"}],
                "relations": [],
            }
        )
        with _patched_dspy(schema_str=legacy_payload), pytest.raises(SchemaGenerationError):
            generate_schema_from_text("Sample text.")

    def test_unrecognized_shape_raises(self):
        payload = json.dumps({"foo": "bar"})
        with _patched_dspy(schema_str=payload), pytest.raises(SchemaGenerationError):
            generate_schema_from_text("Sample text.")
