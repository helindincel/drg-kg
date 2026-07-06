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
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from drg.errors import SchemaGenerationError
from drg.extract._schema_gen import (
    _merge_additional_relation_groups,
    _sample_text_for_schema_generation,
    generate_schema_from_text,
)
from drg.schema import EnhancedDRGSchema, EntityType, Relation, RelationGroup


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

    def test_sample_fraction_env_override(self, monkeypatch):
        monkeypatch.setenv("DRG_SCHEMA_SAMPLE_FRACTION", "0.10")
        monkeypatch.setenv("DRG_SCHEMA_MAX_SAMPLE_CHARS", "50000")
        text = "y" * 200_000
        result = _sample_text_for_schema_generation(text)
        assert len(result) < len(text)

    def test_truncates_on_sentence_boundary(self):
        text = "First sentence here. " + ("word " * 200) + "Final sentence ends."
        from drg.extract._schema_gen import _truncate_text_at_boundary

        truncated = _truncate_text_at_boundary(text, 30)
        assert truncated == "First sentence here."
        assert len(truncated) <= 30


def test_schema_max_tokens_defaults_and_retry_bump(monkeypatch):
    monkeypatch.delenv("DRG_SCHEMA_MAX_TOKENS", raising=False)
    monkeypatch.setenv("DRG_MAX_TOKENS", "4096")
    from drg.extract._schema_gen import _schema_max_tokens

    assert _schema_max_tokens() == 8192
    assert _schema_max_tokens(attempt_idx=2) == 12288


def test_generate_schema_from_text_does_not_mutate_drg_max_tokens(monkeypatch):
    monkeypatch.setenv("DRG_MAX_TOKENS", "4096")
    monkeypatch.setenv("DRG_SCHEMA_COVERAGE_PASS", "0")
    from drg.extract._schema_gen import generate_schema_from_text

    with _patched_dspy(payload=_VALID_SCHEMA_PAYLOAD):
        generate_schema_from_text("Alice works at ACME.")
    assert os.getenv("DRG_MAX_TOKENS") == "4096"


def test_schema_output_limits_defaults(monkeypatch):
    monkeypatch.delenv("DRG_SCHEMA_MAX_ENTITY_TYPES", raising=False)
    monkeypatch.delenv("DRG_SCHEMA_MAX_RELATION_GROUPS", raising=False)
    monkeypatch.delenv("DRG_SCHEMA_MAX_RELATIONS", raising=False)
    from drg.extract._schema_gen import _schema_output_limits

    limits = _schema_output_limits()
    assert limits["max_entity_types"] == 10
    assert limits["max_relation_groups"] == 6
    assert limits["max_relations"] == 32


def test_construction_principles_emphasize_density_over_count():
    from drg.graph.construction_principles import (
        KG_CONSTRUCTION_PRINCIPLES,
        SCHEMA_GENERATION_PRINCIPLES,
    )

    assert "Information density" in KG_CONSTRUCTION_PRINCIPLES
    assert "shortcut" in KG_CONSTRUCTION_PRINCIPLES.lower()
    assert "reusable semantics" in SCHEMA_GENERATION_PRINCIPLES
    assert "canonical relation" in SCHEMA_GENERATION_PRINCIPLES.lower()

    from drg.graph.construction_principles import RELATION_PREFLIGHT_CHECKLIST

    assert "unique semantic fact" in RELATION_PREFLIGHT_CHECKLIST
    assert "semantic precision" in RELATION_PREFLIGHT_CHECKLIST


def test_schema_generation_prompt_preserves_semantic_quality_goals():
    from drg.extract._schema_prompts import (
        SCHEMA_CORE_RULES,
        SCHEMA_COVERAGE_AUDIT_INSTRUCTIONS,
        SCHEMA_GENERATION_INSTRUCTIONS,
        SCHEMA_REVIEW_INSTRUCTIONS,
    )

    for prompt in (
        SCHEMA_CORE_RULES,
        SCHEMA_GENERATION_INSTRUCTIONS,
        SCHEMA_REVIEW_INSTRUCTIONS,
        SCHEMA_COVERAGE_AUDIT_INSTRUCTIONS,
    ):
        assert "ontology_budget" in prompt or "Obey ontology_budget" in prompt
        assert "domain-agnostic" in prompt

    assert "LegalCase" in SCHEMA_CORE_RULES
    assert "interaction_families" in SCHEMA_CORE_RULES
    assert "detail" in SCHEMA_CORE_RULES
    assert "Post-processing removes" in SCHEMA_CORE_RULES

    assert SCHEMA_CORE_RULES in SCHEMA_GENERATION_INSTRUCTIONS
    assert SCHEMA_CORE_RULES in SCHEMA_REVIEW_INSTRUCTIONS
    assert SCHEMA_CORE_RULES in SCHEMA_COVERAGE_AUDIT_INSTRUCTIONS
    assert "additional_entity_budget" in SCHEMA_COVERAGE_AUDIT_INSTRUCTIONS
    assert "synonyms" in SCHEMA_COVERAGE_AUDIT_INSTRUCTIONS

    combined = "\n".join(
        (
            SCHEMA_GENERATION_INSTRUCTIONS,
            SCHEMA_REVIEW_INSTRUCTIONS,
            SCHEMA_COVERAGE_AUDIT_INSTRUCTIONS,
        )
    )
    assert combined.count(SCHEMA_CORE_RULES) == 3
    assert "SCHEMA_GENERATION_PRINCIPLES" not in combined


class _StubPrediction:
    """Stand-in for a DSPy structured ``Prediction``."""

    def __init__(
        self,
        *,
        entity_types=None,
        relation_groups=None,
        additional_relation_groups=None,
        generated_schema: str | None = None,
    ):
        self.entity_types = entity_types
        self.relation_groups = relation_groups
        self.additional_relation_groups = additional_relation_groups
        if generated_schema is not None:
            self.generated_schema = generated_schema


def test_merge_additional_relation_groups_dedupes_and_respects_budget():
    base = EnhancedDRGSchema(
        entity_types=[
            EntityType(name="Organization", description="Orgs"),
            EntityType(name="Product", description="Products"),
        ],
        relation_groups=[
            RelationGroup(
                name="core",
                description="Core",
                relations=[
                    Relation("develops", "Organization", "Product", description="builds"),
                ],
            )
        ],
    )
    merged = _merge_additional_relation_groups(
        base,
        [
            {
                "name": "extensions",
                "description": "More",
                "relations": [
                    {
                        "name": "develops",
                        "source": "Organization",
                        "target": "Product",
                        "description": "duplicate",
                    },
                    {
                        "name": "acquired",
                        "source": "Organization",
                        "target": "Organization",
                        "description": "bought",
                        "detail": "Anthropic acquired Bun.",
                    },
                    {
                        "name": "released",
                        "source": "Organization",
                        "target": "Product",
                        "description": "launched",
                        "detail": "Anthropic released Claude 4.",
                    },
                ],
            }
        ],
        max_relations=2,
    )
    names = {rel.name for rg in merged.relation_groups for rel in rg.relations}
    assert names == {"develops", "acquired"}


def test_merge_additional_relation_groups_allows_same_name_different_endpoints():
    base = EnhancedDRGSchema(
        entity_types=[
            EntityType(name="Organization", description="Orgs"),
            EntityType(name="Product", description="Products"),
            EntityType(name="Person", description="People"),
        ],
        relation_groups=[
            RelationGroup(
                name="core",
                description="Core",
                relations=[
                    Relation("develops", "Organization", "Product", description="builds"),
                ],
            )
        ],
    )
    merged = _merge_additional_relation_groups(
        base,
        [
            {
                "name": "extensions",
                "description": "More",
                "relations": [
                    {
                        "name": "develops",
                        "source": "Person",
                        "target": "Product",
                        "description": "person-authored product",
                    },
                ],
            }
        ],
        max_relations=4,
    )
    keys = {
        (rel.name, rel.src, rel.dst)
        for rg in merged.relation_groups
        for rel in rg.relations
    }
    assert ("develops", "Organization", "Product") in keys
    assert ("develops", "Person", "Product") in keys


_VALID_SCHEMA_PAYLOAD = {
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


def _patched_dspy(
    *,
    payload: dict | None = None,
    legacy_schema_str: str | None = None,
    raise_on_call: Exception | None = None,
):
    """Build a ``patch`` context for ``dspy`` inside ``_schema_gen``.

    The real module is replaced with a SimpleNamespace exposing only the
    surface ``generate_schema_from_text`` touches. The happy path returns
    typed fields; `legacy_schema_str` exercises the compatibility fallback.
    """

    def _make_predictor(*_args, **_kwargs):
        def _call(**kwargs):
            if raise_on_call is not None:
                raise raise_on_call
            if legacy_schema_str is not None:
                return _StubPrediction(generated_schema=legacy_schema_str)
            payload_data = payload or {}
            return _StubPrediction(
                entity_types=payload_data.get("entity_types"),
                relation_groups=payload_data.get("relation_groups"),
            )

        return _call

    fake_dspy = SimpleNamespace(
        Predict=_make_predictor,
        ChainOfThought=_make_predictor,
        Signature=type("Signature", (), {}),
        InputField=lambda **_: None,
        OutputField=lambda **_: None,
    )
    return patch("drg.extract._schema_gen.dspy", fake_dspy)


class TestGenerateSchemaFromText:
    @pytest.fixture(autouse=True)
    def _disable_coverage_pass(self, monkeypatch):
        monkeypatch.setenv("DRG_SCHEMA_COVERAGE_PASS", "0")

    def test_happy_path_returns_enhanced_schema(self):
        with _patched_dspy(payload=_VALID_SCHEMA_PAYLOAD):
            schema = generate_schema_from_text("Alice works at ACME.")
        assert isinstance(schema, EnhancedDRGSchema)
        assert any(et.name == "Person" for et in schema.entity_types)
        # "Company" is canonicalized to "Organization" by the sanitizer.
        assert any(et.name == "Organization" for et in schema.entity_types)
        assert not any(et.name == "Company" for et in schema.entity_types)
        all_relations = {r.name for rg in schema.relation_groups for r in rg.relations}
        assert "works_at" in all_relations
        assert "employs" not in all_relations

    def test_dspy_failure_raises_schema_generation_error(self):
        with (
            _patched_dspy(
                payload={},
                raise_on_call=RuntimeError("LLM timeout"),
            ),
            pytest.raises(SchemaGenerationError, match="Schema generation failed"),
        ):
            generate_schema_from_text("Some text.")

    def test_invalid_json_raises_schema_generation_error(self):
        with (
            _patched_dspy(
                legacy_schema_str="not really json {{{",
            ),
            pytest.raises(SchemaGenerationError, match="Legacy schema JSON parsing failed"),
        ):
            generate_schema_from_text("Some text.")

    def test_empty_schema_raises_schema_generation_error(self):
        with (
            _patched_dspy(payload={}),
            pytest.raises(SchemaGenerationError, match="empty schema"),
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
        with _patched_dspy(legacy_schema_str=legacy_payload):
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
        with _patched_dspy(legacy_schema_str=legacy_payload), pytest.raises(SchemaGenerationError):
            generate_schema_from_text("Sample text.")

    def test_unrecognized_shape_raises(self):
        with _patched_dspy(payload={"foo": "bar"}), pytest.raises(SchemaGenerationError):
            generate_schema_from_text("Sample text.")

    def test_coverage_pass_merges_additional_relations(self, monkeypatch):
        monkeypatch.setenv("DRG_SCHEMA_COVERAGE_PASS", "1")

        coverage_payload = {
            "additional_relation_groups": [
                {
                    "name": "coverage",
                    "description": "Gap fill",
                    "relations": [
                        {
                            "name": "acquired",
                            # The coverage LLM sees the already-sanitized schema,
                            # where "Company" has been canonicalized to "Organization".
                            "source": "Organization",
                            "target": "Organization",
                            "description": "One company acquired another",
                            "detail": "Anthropic acquired Bun.",
                        }
                    ],
                }
            ]
        }

        call_count = {"n": 0}

        def _make_predictor(*_args, **_kwargs):
            def _call(**kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _StubPrediction(
                        entity_types=_VALID_SCHEMA_PAYLOAD["entity_types"],
                        relation_groups=_VALID_SCHEMA_PAYLOAD["relation_groups"],
                    )
                return _StubPrediction(**coverage_payload)

            return _call

        fake_dspy = SimpleNamespace(
            Predict=_make_predictor,
            ChainOfThought=_make_predictor,
            Signature=type("Signature", (), {}),
            InputField=lambda **_: None,
            OutputField=lambda **_: None,
        )
        with patch("drg.extract._schema_gen.dspy", fake_dspy):
            schema = generate_schema_from_text("Anthropic acquired Bun.")
        all_relations = {r.name for rg in schema.relation_groups for r in rg.relations}
        assert "works_at" in all_relations
        assert "acquired" in all_relations
        assert call_count["n"] == 2
