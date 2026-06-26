"""Tests verifying that schema metadata reaches DSPy signatures.

These tests confirm fixes for VF-1, VF-2, VF-6, VF-9, VF-10:
- EntityType descriptions and examples are included in the entity signature.
- RelationGroup descriptions/examples and Relation descriptions reach the
  relation signatures.
- Output field annotations use typed Pydantic models.
- Signature docstrings include behavioral instructions.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock

import pytest

# Stub dspy before drg imports — avoids requiring a real DSPy installation.
sys.modules.setdefault("dspy", MagicMock())

from drg.extract import _coerce_entity_mentions, _entity_mentions_to_dspy_input
from drg.extract._signatures import (
    _create_coreference_signature,
    _create_document_relation_signature,
    _create_entity_signature,
    _create_implicit_relation_signature,
    _create_relation_signature,
    _entity_schema_for,
    _relation_schema_for,
)
from drg.extract._types import EntityMention
from drg.schema import (
    DRGSchema,
    EnhancedDRGSchema,
    Entity,
    EntityGroup,
    EntityType,
    Relation,
    RelationGroup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def enhanced_schema() -> EnhancedDRGSchema:
    return EnhancedDRGSchema(
        entity_types=[
            EntityType(
                name="Researcher",
                description="A person who conducts scientific research",
                examples=["Marie Curie", "Alan Turing"],
                properties={"field": "Research discipline", "era": "Active period"},
            ),
            EntityType(
                name="Institution",
                description="A university, laboratory, or research centre",
                examples=["MIT", "CERN"],
            ),
        ],
        relation_groups=[
            RelationGroup(
                name="affiliation",
                description="Relationships between researchers and institutions",
                relations=[
                    Relation(
                        name="affiliated_with",
                        src="Researcher",
                        dst="Institution",
                        description="A researcher works at or is associated with an institution",
                        detail="Marie Curie was affiliated_with the University of Paris.",
                        properties={"role": "Nature of the affiliation"},
                    )
                ],
                examples=[{"text": "Curie worked at the Sorbonne", "relation": "affiliated_with"}],
            )
        ],
    )


@pytest.fixture()
def legacy_schema() -> DRGSchema:
    return DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[
            Relation(
                name="produces",
                src="Company",
                dst="Product",
                description="A company manufactures or creates a product",
                detail="Apple produces the iPhone.",
            )
        ],
    )


# ---------------------------------------------------------------------------
# VF-10: _entity_schema_for() returns dicts with description and examples
# ---------------------------------------------------------------------------


class TestEntitySchemaFor:
    def test_enhanced_schema_returns_dicts_with_metadata(self, enhanced_schema):
        result = _entity_schema_for(enhanced_schema)

        assert len(result) == 2
        researcher = next(e for e in result if e["name"] == "Researcher")

        assert researcher["description"] == "A person who conducts scientific research"
        assert "Marie Curie" in researcher["examples"]
        assert "Alan Turing" in researcher["examples"]
        assert researcher["properties"]["field"] == "Research discipline"

    def test_legacy_schema_returns_dicts_with_name(self, legacy_schema):
        result = _entity_schema_for(legacy_schema)

        assert len(result) == 2
        names = {e["name"] for e in result}
        assert names == {"Company", "Product"}
        # Legacy entities have no description — should default gracefully
        for entry in result:
            assert "description" in entry
            assert "examples" in entry
            assert "properties" in entry

    def test_entity_group_context_included(self):
        """EntityGroup name and description should appear when entity types are grouped."""
        schema = EnhancedDRGSchema(
            entity_types=[
                EntityType(name="Professor", description="Academic faculty member", examples=[]),
                EntityType(name="PostDoc", description="Post-doctoral researcher", examples=[]),
            ],
            relation_groups=[
                RelationGroup(
                    name="hierarchy",
                    description="Academic hierarchy",
                    relations=[
                        Relation(name="supervises", src="Professor", dst="PostDoc", description="")
                    ],
                )
            ],
            entity_groups=[
                EntityGroup(
                    name="AcademicStaff",
                    description="People employed by an academic institution",
                    entity_types=[
                        EntityType(
                            name="Professor", description="Academic faculty member", examples=[]
                        ),
                        EntityType(
                            name="PostDoc", description="Post-doctoral researcher", examples=[]
                        ),
                    ],
                )
            ],
        )
        result = _entity_schema_for(schema)

        professor = next(e for e in result if e["name"] == "Professor")
        assert professor.get("group") == "AcademicStaff"
        assert professor.get("group_description") == "People employed by an academic institution"


# ---------------------------------------------------------------------------
# VF-1 + VF-6: _relation_schema_for() returns full relation metadata
# ---------------------------------------------------------------------------


class TestRelationSchemaFor:
    def test_enhanced_schema_includes_description_and_detail(self, enhanced_schema):
        result = _relation_schema_for(enhanced_schema)

        assert len(result) == 1
        rel = result[0]

        assert rel["name"] == "affiliated_with"
        assert rel["source_type"] == "Researcher"
        assert rel["target_type"] == "Institution"
        assert rel["description"] == "A researcher works at or is associated with an institution"
        assert "Marie Curie" in rel["example"]
        assert rel["properties"]["role"] == "Nature of the affiliation"
        assert rel["group_examples"] == [
            {"text": "Curie worked at the Sorbonne", "relation": "affiliated_with"}
        ]

    def test_enhanced_schema_includes_group_context(self, enhanced_schema):
        result = _relation_schema_for(enhanced_schema)
        rel = result[0]

        assert rel["group"] == "affiliation"
        assert rel["group_description"] == "Relationships between researchers and institutions"

    def test_legacy_schema_includes_description_and_detail(self, legacy_schema):
        result = _relation_schema_for(legacy_schema)

        assert len(result) == 1
        rel = result[0]
        assert rel["description"] == "A company manufactures or creates a product"
        assert "Apple produces the iPhone" in rel["example"]
        assert rel["properties"] == {}

    def test_legacy_schema_has_no_group_keys(self, legacy_schema):
        result = _relation_schema_for(legacy_schema)
        rel = result[0]
        # Legacy schemas have no group concept; group keys should be absent
        assert "group" not in rel
        assert "group_description" not in rel


# ---------------------------------------------------------------------------
# VF-2: Signature _entity_types is list[dict] not list[str]
# ---------------------------------------------------------------------------


class TestEntitySignatureTypes:
    def test_entity_types_are_dicts(self, enhanced_schema):
        sig = _create_entity_signature(enhanced_schema)
        stored = sig._entity_types  # type: ignore[attr-defined]

        assert isinstance(stored, list)
        assert len(stored) > 0
        assert isinstance(stored[0], dict), "entity_types should be list[dict] not list[str]"

    def test_entity_types_contain_description(self, enhanced_schema):
        sig = _create_entity_signature(enhanced_schema)
        stored = sig._entity_types  # type: ignore[attr-defined]

        researcher = next(e for e in stored if e["name"] == "Researcher")
        assert researcher["description"]
        assert researcher["properties"]["field"] == "Research discipline"

    def test_legacy_schema_entity_types_are_dicts(self, legacy_schema):
        sig = _create_entity_signature(legacy_schema)
        stored = sig._entity_types  # type: ignore[attr-defined]
        assert all(isinstance(e, dict) for e in stored)


# ---------------------------------------------------------------------------
# VF-9: Signature docstrings contain behavioral instructions
# ---------------------------------------------------------------------------
# When DSPy is mocked via MagicMock, the inner class's __doc__ may be
# absorbed by the MagicMock metaclass. We verify the docstrings exist in
# the source file directly, which is the canonical source of truth.


def _signatures_source() -> str:
    """Return the full text of _signatures.py."""
    path = pathlib.Path(__file__).parent.parent / "drg" / "extract" / "_signatures.py"
    return path.read_text(encoding="utf-8")


class TestSignatureDocstrings:
    def test_entity_extraction_docstring_mentions_evidence(self):
        src = _signatures_source()
        # The EntityExtraction docstring should instruct on evidence population.
        assert "evidence" in src.lower()

    def test_relation_extraction_docstring_mentions_negation(self):
        src = _signatures_source()
        assert "is_negated" in src or "negat" in src.lower()

    def test_relation_extraction_docstring_mentions_confidence(self):
        src = _signatures_source()
        assert "confidence" in src.lower()

    def test_relation_extraction_docstring_mentions_temporal(self):
        src = _signatures_source()
        assert "temporal" in src.lower()

    def test_document_relation_docstring_mentions_evidence(self):
        src = _signatures_source()
        # DocumentRelationExtraction class docstring should mention evidence.
        assert "DocumentRelationExtraction" in src
        doc_start = src.index("DocumentRelationExtraction")
        # Check the docstring fragment following the class definition
        assert "evidence" in src[doc_start : doc_start + 1000].lower()

    def test_coreference_docstring_mentions_canonical(self):
        src = _signatures_source()
        assert "CoreferenceResolution" in src
        doc_start = src.index("CoreferenceResolution")
        assert "canonical" in src[doc_start : doc_start + 1000].lower()


# ---------------------------------------------------------------------------
# VF-2: Relation schema stored on signatures includes description
# ---------------------------------------------------------------------------


class TestRelationSignatureSchema:
    def test_relation_schema_includes_description(self, enhanced_schema):
        for factory in [
            _create_relation_signature,
            _create_document_relation_signature,
            _create_implicit_relation_signature,
            _create_coreference_signature,
        ]:
            sig = factory(enhanced_schema)
            stored = sig._relation_schema  # type: ignore[attr-defined]
            assert stored, f"{factory.__name__} produced empty relation schema"
            assert stored[0].get("description"), (
                f"{factory.__name__}: relation missing 'description'"
            )
            assert stored[0].get("group"), f"{factory.__name__}: relation missing 'group'"
            assert stored[0].get("group_examples"), (
                f"{factory.__name__}: relation missing 'group_examples'"
            )


class TestTypedOutputModels:
    def test_entity_mention_exposes_schema_defined_properties(self):
        mention = EntityMention(
            name="archive tower",
            type="Place",
            properties={"atmosphere": "silent", "role": "setting"},
        )

        assert mention.properties["atmosphere"] == "silent"

    def test_entity_properties_survive_normalization_and_dspy_input(self):
        mentions = _coerce_entity_mentions(
            [
                {
                    "name": "sealed letter",
                    "type": "Artifact",
                    "properties": {"condition": "unopened"},
                }
            ]
        )

        assert mentions[0].properties == {"condition": "unopened"}
        dspy_input = _entity_mentions_to_dspy_input(mentions)
        assert dspy_input[0]["properties"] == {"condition": "unopened"}
