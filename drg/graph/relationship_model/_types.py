"""Relationship type taxonomy.

Centralizes the standard, domain-agnostic set of relationship types DRG uses
across datasets. Keeping the enum and the category index in one module makes
it the single place to extend the taxonomy — both ``RelationshipType`` and
``RELATIONSHIP_CATEGORIES`` are re-exported at the package level.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["RELATIONSHIP_CATEGORIES", "RelationshipType"]


class RelationshipType(str, Enum):
    """Standard relationship type taxonomy.

    Domain-agnostic — works across business, scientific, medical and social
    datasets. New types belong here; the rule-based classifier in
    ``_rule_based`` reads category groupings from :data:`RELATIONSHIP_CATEGORIES`.
    """

    # Causal
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    TRIGGERS = "triggers"
    RESULTS_IN = "results_in"

    # Spatial
    LOCATED_AT = "located_at"
    CONTAINS = "contains"
    NEAR = "near"
    INSIDE = "inside"
    OUTSIDE = "outside"

    # Temporal
    OCCURS_BEFORE = "occurs_before"
    OCCURS_AFTER = "occurs_after"
    OCCURS_DURING = "occurs_during"
    FOLLOWS = "follows"

    # Social / interaction
    INFLUENCES = "influences"
    INFLUENCED_BY = "influenced_by"
    COLLABORATES_WITH = "collaborates_with"
    WORKS_WITH = "works_with"
    OWNS = "owns"
    BELONGS_TO = "belongs_to"
    MEMBER_OF = "member_of"

    # Hierarchical
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    PART_OF = "part_of"
    HAS_PART = "has_part"

    # Similarity / equivalence
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"
    EQUIVALENT_TO = "equivalent_to"

    # Action
    CREATES = "creates"
    DESTROYS = "destroys"
    MODIFIES = "modifies"
    PRODUCES = "produces"
    CONSUMES = "consumes"

    # Communication
    COMMUNICATES_WITH = "communicates_with"
    INFORMS = "informs"
    REQUESTS = "requests"
    RESPONDS_TO = "responds_to"

    # Emotional / subjective
    LIKES = "likes"
    DISLIKES = "dislikes"
    LOVES = "loves"
    HATES = "hates"
    FEARS = "fears"
    TRUSTS = "trusts"

    # Misc
    KNOWS = "knows"
    MEETS = "meets"
    VISITS = "visits"
    LEAVES = "leaves"
    RETURNS_TO = "returns_to"


RELATIONSHIP_CATEGORIES: dict[str, list[RelationshipType]] = {
    "causal": [
        RelationshipType.CAUSES,
        RelationshipType.CAUSED_BY,
        RelationshipType.TRIGGERS,
        RelationshipType.RESULTS_IN,
    ],
    "spatial": [
        RelationshipType.LOCATED_AT,
        RelationshipType.CONTAINS,
        RelationshipType.NEAR,
        RelationshipType.INSIDE,
        RelationshipType.OUTSIDE,
    ],
    "temporal": [
        RelationshipType.OCCURS_BEFORE,
        RelationshipType.OCCURS_AFTER,
        RelationshipType.OCCURS_DURING,
        RelationshipType.FOLLOWS,
    ],
    "social": [
        RelationshipType.INFLUENCES,
        RelationshipType.INFLUENCED_BY,
        RelationshipType.COLLABORATES_WITH,
        RelationshipType.WORKS_WITH,
        RelationshipType.OWNS,
        RelationshipType.BELONGS_TO,
        RelationshipType.MEMBER_OF,
    ],
    "hierarchical": [
        RelationshipType.PARENT_OF,
        RelationshipType.CHILD_OF,
        RelationshipType.PART_OF,
        RelationshipType.HAS_PART,
    ],
    "action": [
        RelationshipType.CREATES,
        RelationshipType.DESTROYS,
        RelationshipType.MODIFIES,
        RelationshipType.PRODUCES,
        RelationshipType.CONSUMES,
    ],
    "communication": [
        RelationshipType.COMMUNICATES_WITH,
        RelationshipType.INFORMS,
        RelationshipType.REQUESTS,
        RelationshipType.RESPONDS_TO,
    ],
    "emotional": [
        RelationshipType.LIKES,
        RelationshipType.DISLIKES,
        RelationshipType.LOVES,
        RelationshipType.HATES,
        RelationshipType.FEARS,
        RelationshipType.TRUSTS,
    ],
}
