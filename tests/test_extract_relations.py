"""Unit tests for drg.extract._relations — no LLM / DSPy required."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# drg/extract/__init__.py imports dspy at module level.
# Stub it so these pure-Python tests don't need dspy installed.
sys.modules.setdefault("dspy", MagicMock())

from drg.extract._relations import (
    REVERSE_RELATION_PATTERNS,
    _add_reverse_relations,
    _infer_reverse_relation_name,
    _normalize_schema,
)
from drg.schema import (
    DRGSchema,
    EnhancedDRGSchema,
    Entity,
    EntityType,
    Relation,
    RelationGroup,
)


# ---------------------------------------------------------------------------
# REVERSE_RELATION_PATTERNS — basic sanity
# ---------------------------------------------------------------------------


class TestReverseRelationPatterns:
    def test_produces_has_inverse(self):
        assert REVERSE_RELATION_PATTERNS["produces"] == "produced_by"
        assert REVERSE_RELATION_PATTERNS["produced_by"] == "produces"

    def test_owns_has_inverse(self):
        assert REVERSE_RELATION_PATTERNS["owns"] == "owned_by"
        assert REVERSE_RELATION_PATTERNS["owned_by"] == "owns"

    def test_located_in_contains_pair(self):
        assert REVERSE_RELATION_PATTERNS["located_in"] == "contains"
        assert REVERSE_RELATION_PATTERNS["contains"] == "located_in"

    def test_symmetric_relations_map_to_themselves(self):
        for sym in ("related_to", "connected_to", "partners_with", "collaborates_with"):
            assert REVERSE_RELATION_PATTERNS[sym] == sym

    def test_all_reverse_values_are_strings(self):
        for k, v in REVERSE_RELATION_PATTERNS.items():
            assert isinstance(k, str) and isinstance(v, str), f"Bad entry: {k!r}: {v!r}"

    def test_table_is_non_empty(self):
        assert len(REVERSE_RELATION_PATTERNS) > 10


# ---------------------------------------------------------------------------
# _infer_reverse_relation_name
# ---------------------------------------------------------------------------


class TestInferReverseRelationName:
    # "_by" suffix branch
    def test_by_suffix_with_ed_base_strips_by(self):
        # "created_by" → base "created" ends with "ed" → returns "created" ... wait,
        # the logic returns base (strips _by), and if it ends with "ed" returns base.
        # "created_by" → base = "created" → ends with "ed" → return "created"
        result = _infer_reverse_relation_name("created_by")
        assert result == "created"

    def test_by_suffix_without_ed_adds_s(self):
        # "managed_by" → base = "managed" → ends with "ed" → return "managed"
        result = _infer_reverse_relation_name("managed_by")
        assert result == "managed"

    def test_by_suffix_base_already_ends_s(self):
        # e.g. "owns_by" is unusual but test defensive path: base="owns" doesn't end "ed",
        # already ends "s" → return base unchanged
        result = _infer_reverse_relation_name("owns_by")
        assert result == "owns"

    # "_of" suffix branch
    def test_of_suffix_produces_has_prefix(self):
        assert _infer_reverse_relation_name("part_of") == "has_part"
        assert _infer_reverse_relation_name("member_of") == "has_member"

    # "_from" suffix branch
    def test_from_suffix_strips_from(self):
        # "originates_from" → base = "originates" ends with "s" → return unchanged
        result = _infer_reverse_relation_name("originates_from")
        assert result == "originates"

    def test_from_suffix_base_no_s_adds_s(self):
        # "derive_from" → base = "derive" doesn't end "s" → "derives"
        result = _infer_reverse_relation_name("derive_from")
        assert result == "derives"

    # Direct action verb branch
    def test_verb_ending_s_produces_ed_by(self):
        # "produces"[:-1] = "produce" (strips the trailing 's')
        # then appends "ed_by" → "produceed_by"
        # This reflects the actual heuristic: strip final 's', add 'ed_by'
        result = _infer_reverse_relation_name("produces")
        assert result == "produceed_by"

    def test_verb_not_ending_s_produces_d_by(self):
        # "employ" → doesn't end with "s" → "employd_by"
        result = _infer_reverse_relation_name("employ")
        assert result == "employd_by"

    # Suffix that returns None (_in / _at)
    def test_in_suffix_returns_none(self):
        assert _infer_reverse_relation_name("located_in") is None

    def test_at_suffix_returns_none(self):
        assert _infer_reverse_relation_name("located_at") is None


# ---------------------------------------------------------------------------
# _normalize_schema
# ---------------------------------------------------------------------------


class TestNormalizeSchema:
    def test_drg_schema_passthrough(self):
        schema = DRGSchema(
            entities=[Entity("Company"), Entity("Product")],
            relations=[Relation("produces", "Company", "Product")],
        )
        result = _normalize_schema(schema)
        assert result is schema

    def test_enhanced_schema_converts_to_legacy(self):
        schema = EnhancedDRGSchema(
            entity_types=[
                EntityType("Company", "A business organization"),
                EntityType("Product", "A manufactured good"),
            ],
            relation_groups=[
                RelationGroup(
                    "production",
                    "Production relationships",
                    [Relation("produces", "Company", "Product")],
                )
            ],
        )
        result = _normalize_schema(schema)
        assert isinstance(result, DRGSchema)
        entity_names = {e.name for e in result.entities}
        assert "Company" in entity_names
        assert "Product" in entity_names
        relation_names = {r.name for r in result.relations}
        assert "produces" in relation_names


# ---------------------------------------------------------------------------
# _add_reverse_relations
# ---------------------------------------------------------------------------


class TestAddReverseRelations:
    def _make_entity_types(self, *names: str) -> list[EntityType]:
        return [EntityType(n, f"Type {n}") for n in names]

    def _make_relation_group(
        self, name: str, relations: list[Relation]
    ) -> RelationGroup:
        return RelationGroup(name, f"Group {name}", relations)

    def test_adds_reverse_for_known_pattern(self):
        entity_types = self._make_entity_types("Company", "Product")
        rg = self._make_relation_group(
            "prod", [Relation("produces", "Company", "Product")]
        )
        result = _add_reverse_relations([rg], entity_types)
        all_names = {r.name for r in result[0].relations}
        assert "produces" in all_names
        assert "produced_by" in all_names

    def test_reverse_has_swapped_src_dst(self):
        entity_types = self._make_entity_types("Company", "Product")
        rg = self._make_relation_group(
            "prod", [Relation("produces", "Company", "Product")]
        )
        result = _add_reverse_relations([rg], entity_types)
        rev = next(r for r in result[0].relations if r.name == "produced_by")
        assert rev.src == "Product"
        assert rev.dst == "Company"

    def test_does_not_duplicate_existing_reverse(self):
        entity_types = self._make_entity_types("Company", "Product")
        rg = self._make_relation_group(
            "prod",
            [
                Relation("produces", "Company", "Product"),
                Relation("produced_by", "Product", "Company"),
            ],
        )
        result = _add_reverse_relations([rg], entity_types)
        names = [r.name for r in result[0].relations]
        assert names.count("produced_by") == 1

    def test_skips_reverse_when_entity_type_missing(self):
        # Only "Company" exists — "Product" is not an entity type
        entity_types = self._make_entity_types("Company")
        rg = self._make_relation_group(
            "prod", [Relation("produces", "Company", "Product")]
        )
        result = _add_reverse_relations([rg], entity_types)
        # produced_by would require "Product" as src which is not in entity_types
        all_names = {r.name for r in result[0].relations}
        assert "produced_by" not in all_names

    def test_no_known_reverse_does_not_add_relation(self):
        entity_types = self._make_entity_types("A", "B")
        rg = self._make_relation_group(
            "misc", [Relation("custom_relation_xyz", "A", "B")]
        )
        result = _add_reverse_relations([rg], entity_types)
        # custom_relation_xyz is not in REVERSE_RELATION_PATTERNS
        assert len(result[0].relations) == 1

    def test_handles_multiple_groups(self):
        entity_types = self._make_entity_types("Company", "Person", "City")
        rg1 = self._make_relation_group(
            "work", [Relation("employs", "Company", "Person")]
        )
        rg2 = self._make_relation_group(
            "loc", [Relation("located_in", "Company", "City")]
        )
        result = _add_reverse_relations([rg1, rg2], entity_types)
        assert len(result) == 2
        rg1_names = {r.name for r in result[0].relations}
        rg2_names = {r.name for r in result[1].relations}
        assert "works_at" in rg1_names
        assert "contains" in rg2_names

    def test_empty_groups_returns_empty(self):
        entity_types = self._make_entity_types("A", "B")
        result = _add_reverse_relations([], entity_types)
        assert result == []

    def test_symmetric_relation_not_duplicated(self):
        entity_types = self._make_entity_types("Person", "Person")
        # "related_to" maps to itself — should not duplicate
        rg = self._make_relation_group(
            "sym", [Relation("related_to", "Person", "Person")]
        )
        result = _add_reverse_relations([rg], entity_types)
        names = [r.name for r in result[0].relations]
        assert names.count("related_to") == 1
