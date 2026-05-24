"""Regression tests for the refactored drg.graph.relationship_model package.

Verifies:
- Public API is preserved (imports + attributes match the legacy module).
- Rule-based classification reaches the expected high-confidence results.
- Schema constraint filtering tightens the result set.
- EnrichedRelationship validation still raises ValueError.
"""

from __future__ import annotations

import pytest

from drg.graph.relationship_model import (
    DSPY_AVAILABLE,
    RELATIONSHIP_CATEGORIES,
    EnrichedRelationship,
    RelationshipType,
    RelationshipTypeClassifier,
    build_classification_input,
    create_enriched_relationship,
    create_relationship_classifier,
)
from drg.graph.relationship_model._rule_based import (
    apply_schema_constraints,
    build_schema_indexes,
    classify_rule_based,
)
from drg.schema import EnhancedDRGSchema, EntityType, Relation, RelationGroup

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_public_api_present():
    """Every symbol the legacy module exposed must still be importable."""
    assert RelationshipType.CAUSES.value == "causes"
    assert "causal" in RELATIONSHIP_CATEGORIES
    assert callable(create_enriched_relationship)
    assert callable(create_relationship_classifier)
    assert callable(build_classification_input)
    assert isinstance(DSPY_AVAILABLE, bool)


def test_relationship_categories_cover_taxonomy():
    """Every category lists only valid RelationshipType members."""
    for category, types in RELATIONSHIP_CATEGORIES.items():
        for rt in types:
            assert isinstance(rt, RelationshipType), f"Non-enum in category {category}: {rt}"


# ---------------------------------------------------------------------------
# EnrichedRelationship dataclass
# ---------------------------------------------------------------------------


def test_enriched_relationship_round_trip():
    rel = create_enriched_relationship(
        source="A",
        target="B",
        relationship_type=RelationshipType.OWNS,
        relationship_detail="A owns B",
        confidence=0.9,
        source_ref="chunk_3",
    )
    data = rel.to_dict()
    rebuilt = EnrichedRelationship.from_dict(data)
    assert rebuilt == rel
    assert rel.to_enriched_format()["source_ref"] == "chunk_3"


def test_enriched_relationship_empty_source_raises():
    with pytest.raises(ValueError, match="Source"):
        EnrichedRelationship(
            source="",
            target="B",
            relationship_type=RelationshipType.OWNS,
            relationship_detail="x",
        )


def test_enriched_relationship_self_loop_raises():
    with pytest.raises(ValueError, match="cannot be the same"):
        EnrichedRelationship(
            source="A",
            target="A",
            relationship_type=RelationshipType.OWNS,
            relationship_detail="x",
        )


def test_enriched_relationship_confidence_bounds_raise():
    with pytest.raises(ValueError, match="Confidence"):
        EnrichedRelationship(
            source="A",
            target="B",
            relationship_type=RelationshipType.OWNS,
            relationship_detail="x",
            confidence=1.5,
        )


# ---------------------------------------------------------------------------
# Rule-based classification (extracted module)
# ---------------------------------------------------------------------------


def test_classify_rule_based_keyword_hit():
    results = classify_rule_based(raw_relation_text="A causes B")
    rel_types = {rt for rt, _ in results}
    assert RelationshipType.CAUSES in rel_types


def test_classify_rule_based_falls_back_when_empty():
    """No keyword and no type info → must still return at least one tuple."""
    results = classify_rule_based()
    assert results
    rel_types = [rt for rt, _ in results]
    assert RelationshipType.RELATED_TO in rel_types


def test_classify_rule_based_person_pair_default():
    """Person↔Person with no text → social/known fallbacks."""
    results = classify_rule_based(source_type="Person", target_type="Person")
    rel_types = {rt for rt, _ in results}
    assert RelationshipType.RELATED_TO in rel_types or RelationshipType.KNOWS in rel_types


# ---------------------------------------------------------------------------
# Schema-aware constraint filtering
# ---------------------------------------------------------------------------


def _build_minimal_schema():
    return EnhancedDRGSchema(
        entity_types=[
            EntityType(name="Person", description="People"),
            EntityType(name="Company", description="Companies"),
        ],
        relation_groups=[
            RelationGroup(
                name="ownership",
                description="ownership",
                relations=[
                    Relation(
                        name="owns",
                        src="Person",
                        dst="Company",
                        description="ownership",
                        detail="x owns y",
                    ),
                ],
            ),
        ],
    )


def test_build_schema_indexes_picks_up_relation_groups():
    schema = _build_minimal_schema()
    index = build_schema_indexes(schema)
    assert ("Person", "Company") in index
    assert "owns" in index[("Person", "Company")]


def test_apply_schema_constraints_keeps_valid_candidate():
    schema = _build_minimal_schema()
    index = build_schema_indexes(schema)

    filtered = apply_schema_constraints(
        [(RelationshipType.OWNS, 0.9), (RelationshipType.LIKES, 0.7)],
        source_type="Person",
        target_type="Company",
        valid_relations=index,
    )

    rel_types = [rt for rt, _ in filtered]
    assert RelationshipType.OWNS in rel_types
    assert RelationshipType.LIKES not in rel_types


def test_apply_schema_constraints_passes_through_when_pair_absent():
    schema = _build_minimal_schema()
    index = build_schema_indexes(schema)
    candidates = [(RelationshipType.RELATED_TO, 0.3)]

    filtered = apply_schema_constraints(
        candidates,
        source_type="Unknown",
        target_type="Other",
        valid_relations=index,
    )
    assert filtered == candidates


# ---------------------------------------------------------------------------
# Classifier dispatcher
# ---------------------------------------------------------------------------


def test_classifier_short_circuits_on_high_confidence_rule():
    """When rules already produce ≥0.8 confidence, LLM path is skipped."""
    clf = RelationshipTypeClassifier(use_llm=False)
    results = clf.classify(
        "Alice",
        "Acme",
        source_type="Person",
        target_type="Company",
        raw_relation_text="Alice owns Acme",
    )
    assert results
    top_type, top_conf = results[0]
    assert top_type == RelationshipType.OWNS
    assert top_conf >= 0.8


def test_classifier_uses_schema_index():
    schema = _build_minimal_schema()
    clf = RelationshipTypeClassifier(schema=schema, use_llm=False)
    assert ("Person", "Company") in clf._valid_relations


def test_classifier_build_schema_indexes_method_back_compat():
    """Legacy callers used the protected _build_schema_indexes; keep it working."""
    clf = RelationshipTypeClassifier(use_llm=False)
    clf._build_schema_indexes()
    assert clf._valid_relations == {}


def test_classifier_disables_llm_when_unavailable():
    """use_llm=True with DSPY_AVAILABLE=False still works (LLM path skipped)."""
    from drg.graph.relationship_model import _llm_based as _llm

    original = _llm.DSPY_AVAILABLE
    _llm.DSPY_AVAILABLE = False
    try:
        clf = RelationshipTypeClassifier(use_llm=True)
        assert clf.use_llm is False
    finally:
        _llm.DSPY_AVAILABLE = original
