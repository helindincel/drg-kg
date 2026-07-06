"""Unit tests for :mod:`drg.extract._schema_sanitizer`.

Focus is on the canonical-vocabulary passes (0a/0b/0c) added to fix the
ontology-audit findings, plus regression coverage for the pre-existing
structural passes and idempotency.
"""

from __future__ import annotations

import pytest

from drg.extract._schema_sanitizer import SchemaSanitizer
from drg.schema import EnhancedDRGSchema, EntityType, Relation, RelationGroup


def _schema(entity_types, relation_groups):
    return EnhancedDRGSchema(
        entity_types=entity_types,
        relation_groups=relation_groups,
    )


def _rel_names(schema):
    return {r.name for rg in schema.relation_groups for r in rg.relations}


def _entity_names(schema):
    return {et.name for et in schema.entity_types}


def _group_names(schema):
    return {rg.name for rg in schema.relation_groups}


# ---------------------------------------------------------------------------
# Pass 0a — canonical relation names
# ---------------------------------------------------------------------------


class TestRelationNameCanonicalization:
    def test_develops_synonyms_collapse_to_single_canonical(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Product", description="products"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("developed", "Organization", "Product", description="d"),
                        Relation("created", "Organization", "Product", description="c"),
                        Relation("produces", "Organization", "Product", description="p"),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        names = _rel_names(clean)
        # All three synonyms collapse onto the endpoint-free 'develops', deduped.
        assert names == {"develops"}

    def test_partnership_synonyms_collapse(self):
        schema = _schema(
            [EntityType(name="Organization", description="orgs")],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("partnered_with", "Organization", "Organization", description="1"),
                        Relation(
                            "formed_agreement_with", "Organization", "Organization", description="2"
                        ),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert _rel_names(clean) == {"partnered_with"}

    def test_headquarters_synonym_canonicalized(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Location", description="places"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation(
                            "has_headquarters_in", "Organization", "Location", description="hq"
                        ),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert _rel_names(clean) == {"headquartered_in"}

    def test_disabled_flag_leaves_names_untouched(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Product", description="products"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("developed", "Organization", "Product", description="d"),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer(canonicalize_names=False).sanitize(schema)
        assert _rel_names(clean) == {"developed"}


class TestEndpointFreeRelationNames:
    def test_endpoint_type_stripped_from_name(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Product", description="products"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("develops_product", "Organization", "Product", description="d"),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert _rel_names(clean) == {"develops"}

    def test_role_prefix_and_suffix_stripped(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Person", description="people"),
                EntityType(name="Field_Of_Study", description="fields"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation(
                            "person_contributed_to_field",
                            "Person",
                            "Field_Of_Study",
                            description="c",
                        ),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert _rel_names(clean) == {"contributed_to"}

    def test_endpoint_free_name_shared_across_targets_survives(self):
        # organization_monitors_{person,technology,concept} -> a single
        # 'monitors' name spanning three targets; endpoint-aware Pass 3 keeps all.
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Person", description="people"),
                EntityType(name="Technology", description="tech"),
                EntityType(name="Concept", description="concepts"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation(
                            "organization_monitors_person",
                            "Organization",
                            "Person",
                            description="1",
                        ),
                        Relation(
                            "organization_monitors_technology",
                            "Organization",
                            "Technology",
                            description="2",
                        ),
                        Relation(
                            "organization_monitors_concept",
                            "Organization",
                            "Concept",
                            description="3",
                        ),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert _rel_names(clean) == {"monitors"}
        # All three endpoint pairs are preserved under the single canonical name.
        monitors_targets = {
            r.dst for rg in clean.relation_groups for r in rg.relations if r.name == "monitors"
        }
        assert monitors_targets == {"Person", "Technology", "Concept"}

    def test_strip_skipped_when_only_light_verb_remains(self):
        # Stripping the "product" role token would leave the meaningless "has";
        # the light-verb guard must keep the original name instead.
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Product", description="products"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("has_product", "Organization", "Product", description="e"),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert _rel_names(clean) == {"has_product"}

    def test_employment_synonyms_canonicalized(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Person", description="people"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("has_employee", "Organization", "Person", description="e"),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert _rel_names(clean) == {"employs"}

    def test_names_without_role_tokens_unchanged(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Person", description="people"),
                EntityType(name="Location", description="places"),
                EntityType(name="Product", description="products"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("founded_by", "Organization", "Person", description="1"),
                        Relation("works_at", "Person", "Organization", description="2"),
                        Relation("is_version_of", "Product", "Product", description="3"),
                        Relation("has_facility_in", "Organization", "Location", description="4"),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert _rel_names(clean) == {"founded_by", "works_at", "is_version_of", "has_facility_in"}


# ---------------------------------------------------------------------------
# Pass 0b — canonical entity types
# ---------------------------------------------------------------------------


class TestEntityTypeCanonicalization:
    def test_company_renamed_to_organization(self):
        schema = _schema(
            [
                EntityType(name="Company", description="a business", examples=["ACME"]),
                EntityType(name="Person", description="a human", examples=["Alice"]),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("founded_by", "Company", "Person", description="f"),
                    ],
                )
            ],
        )
        clean, report = SchemaSanitizer().sanitize(schema)
        assert "Organization" in _entity_names(clean)
        assert "Company" not in _entity_names(clean)
        # Relation endpoint rewired.
        rel = clean.relation_groups[0].relations[0]
        assert rel.src == "Organization"
        assert "Company" in report.removed_entity_types

    def test_artifact_subtypes_collapse_into_product(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Product", description="products", examples=["Windows"]),
                EntityType(name="OperatingSystem", description="os", examples=["MS-DOS"]),
                EntityType(name="Hardware", description="hw", examples=["Xbox"]),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("develops_product", "Organization", "Product", description="d"),
                        Relation(
                            "develops_os", "Organization", "OperatingSystem", description="d2"
                        ),
                        Relation("makes_hw", "Organization", "Hardware", description="d3"),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert _entity_names(clean) == {"Organization", "Product"}
        # Examples unioned onto the surviving Product type.
        product = next(et for et in clean.entity_types if et.name == "Product")
        assert {"Windows", "MS-DOS", "Xbox"}.issubset(set(product.examples))

    def test_artifact_collapse_can_be_disabled(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Product", description="products"),
                EntityType(name="OperatingSystem", description="os"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("develops_product", "Organization", "Product", description="d"),
                        Relation(
                            "develops_os", "Organization", "OperatingSystem", description="d2"
                        ),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer(collapse_artifact_subtypes=False).sanitize(schema)
        assert "OperatingSystem" in _entity_names(clean)

    def test_canonical_named_description_wins_on_merge(self):
        # Product appears after Hardware; the genuine Product description must win.
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Hardware", description="hardware desc", examples=["Xbox"]),
                EntityType(
                    name="Product", description="the real product desc", examples=["Windows"]
                ),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("makes_hw", "Organization", "Hardware", description="d"),
                        Relation("develops_product", "Organization", "Product", description="d2"),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        product = next(et for et in clean.entity_types if et.name == "Product")
        assert product.description == "the real product desc"


# ---------------------------------------------------------------------------
# Pass 0c — merge fragmented relation groups
# ---------------------------------------------------------------------------


class TestRelationGroupMerge:
    def test_product_groups_merge_into_one_family(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Product", description="products"),
            ],
            [
                RelationGroup(
                    name="Product and Technology Lifecycle",
                    description="lifecycle",
                    relations=[Relation("released", "Organization", "Product", description="r")],
                ),
                RelationGroup(
                    name="Product Development",
                    description="dev",
                    relations=[
                        Relation("develops_product", "Organization", "Product", description="d")
                    ],
                ),
                RelationGroup(
                    name="Product Usage",
                    description="usage",
                    relations=[Relation("used_by", "Product", "Organization", description="u")],
                ),
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert _group_names(clean) == {"Product & Technology"}
        # All relations survive under the merged family (develops_product -> develops).
        assert {"released", "develops", "used_by"}.issubset(_rel_names(clean))

    def test_merge_can_be_disabled(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Product", description="products"),
            ],
            [
                RelationGroup(
                    name="Product Development",
                    description="dev",
                    relations=[
                        Relation("develops_product", "Organization", "Product", description="d")
                    ],
                ),
                RelationGroup(
                    name="Product Usage",
                    description="usage",
                    relations=[Relation("used_by", "Product", "Organization", description="u")],
                ),
            ],
        )
        clean, _ = SchemaSanitizer(merge_relation_groups=False).sanitize(schema)
        assert _group_names(clean) == {"Product Development", "Product Usage"}


# ---------------------------------------------------------------------------
# Pass 10 — new document-specific / narrative patterns
# ---------------------------------------------------------------------------


class TestNarrativePruning:
    @pytest.mark.parametrize(
        "rel_name,src,dst",
        [
            ("inspired_by", "Person", "Product"),
            ("used_in_operation", "Product", "Location"),
            ("moved_headquarters_to", "Organization", "Location"),
            ("replaced_as_ceo", "Person", "Person"),
            ("is_brand_of", "Product", "Organization"),
        ],
    )
    def test_narrative_relations_pruned(self, rel_name, src, dst):
        entities = {
            "Person": EntityType(name="Person", description="p"),
            "Product": EntityType(name="Product", description="pr"),
            "Location": EntityType(name="Location", description="loc"),
            "Organization": EntityType(name="Organization", description="org"),
        }
        # Keep a harmless anchor relation so the group is never empty pre-prune.
        schema = _schema(
            list(entities.values()),
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("founded_by", "Organization", "Person", description="anchor"),
                        Relation(rel_name, src, dst, description="x"),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert rel_name not in _rel_names(clean)
        assert "founded_by" in _rel_names(clean)


# ---------------------------------------------------------------------------
# Regression: pre-existing structural passes still fire
# ---------------------------------------------------------------------------


class TestStructuralRegression:
    def test_primitive_entity_type_removed(self):
        schema = _schema(
            [
                EntityType(name="Organization", description="orgs"),
                EntityType(name="Product", description="products"),
                EntityType(name="Date", description="a date"),
            ],
            [
                RelationGroup(
                    name="A",
                    description="a",
                    relations=[
                        Relation("develops_product", "Organization", "Product", description="d"),
                        Relation("released_on", "Organization", "Date", description="ro"),
                    ],
                )
            ],
        )
        clean, _ = SchemaSanitizer().sanitize(schema)
        assert "Date" not in _entity_names(clean)
        assert "released_on" not in _rel_names(clean)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_sanitize_is_idempotent():
    schema = _schema(
        [
            EntityType(name="Company", description="a business", examples=["ACME"]),
            EntityType(name="Product", description="products", examples=["Widget"]),
            EntityType(name="OperatingSystem", description="os", examples=["MS-DOS"]),
            EntityType(name="Person", description="humans", examples=["Alice"]),
        ],
        [
            RelationGroup(
                name="Product Development",
                description="dev",
                relations=[
                    Relation("developed", "Company", "Product", description="d"),
                    Relation("created", "Company", "OperatingSystem", description="c"),
                    Relation("inspired_by", "Person", "Product", description="i"),
                ],
            ),
            RelationGroup(
                name="Product Usage",
                description="usage",
                relations=[Relation("founded_by", "Company", "Person", description="f")],
            ),
        ],
    )
    sanitizer = SchemaSanitizer()
    once, _ = sanitizer.sanitize(schema)
    twice, report2 = sanitizer.sanitize(once)

    assert _entity_names(once) == _entity_names(twice)
    assert _rel_names(once) == _rel_names(twice)
    assert _group_names(once) == _group_names(twice)
    assert report2.total_changes == 0
