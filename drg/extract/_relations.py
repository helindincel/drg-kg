"""Relation helpers: schema normalization, reverse-relation handling.

These helpers contain an optional reverse-relation pattern table. The table is
kept for explicit compatibility paths, not as the default extraction contract.
"""

from __future__ import annotations

import logging

from ..schema import DRGSchema, EnhancedDRGSchema, EntityType, Relation, RelationGroup

logger = logging.getLogger(__name__)


# Canonical bidirectional relation patterns used only when callers explicitly
# opt into reverse-relation normalization.
REVERSE_RELATION_PATTERNS: dict[str, str] = {
    # Production / creation
    "produces": "produced_by",
    "produced_by": "produces",
    "creates": "created_by",
    "created_by": "creates",
    "created": "created_by",
    "manufactures": "manufactured_by",
    "manufactured_by": "manufactures",
    "builds": "built_by",
    "built_by": "builds",
    "makes": "made_by",
    "made_by": "makes",
    # Ownership
    "owns": "owned_by",
    "owned_by": "owns",
    "possesses": "possessed_by",
    "possessed_by": "possesses",
    # Founding / establishment
    "founds": "founded_by",
    "founded_by": "founds",
    "founded": "founded_by",
    "establishes": "established_by",
    "established_by": "establishes",
    # Design / development
    "designs": "designed_by",
    "designed_by": "designs",
    "designed": "designed_by",
    "develops": "developed_by",
    "developed_by": "develops",
    "programs": "programmed_by",
    "programmed_by": "programs",
    # Location
    "located_in": "contains",
    "contains": "located_in",
    "located_at": "hosts",
    "hosts": "located_at",
    "situated_in": "contains",
    # Employment / work
    "works_at": "employs",
    "employs": "works_at",
    "works_for": "employs",
    "employed_by": "employs",
    # Membership
    "member_of": "has_member",
    "has_member": "member_of",
    "part_of": "has_part",
    "has_part": "part_of",
    "belongs_to": "has_member",
    # Hierarchy
    "parent_of": "child_of",
    "child_of": "parent_of",
    "manager_of": "reports_to",
    "reports_to": "manager_of",
    "supervisor_of": "reports_to",
    "subordinate_of": "supervises",
    # Symmetric
    "related_to": "related_to",
    "connected_to": "connected_to",
    "partners_with": "partners_with",
    "collaborates_with": "collaborates_with",
    # Generic action
    "operates": "operated_by",
    "operated_by": "operates",
    "manages": "managed_by",
    "managed_by": "manages",
    "controls": "controlled_by",
    "controlled_by": "controls",
}


def _normalize_schema(schema: DRGSchema | EnhancedDRGSchema) -> DRGSchema:
    """Convert EnhancedDRGSchema to DRGSchema for internal use."""
    if isinstance(schema, EnhancedDRGSchema):
        return schema.to_legacy_schema()
    return schema


def _infer_reverse_relation_name(relation_name: str) -> str | None:
    """Infer reverse relation name from relation name (domain-agnostic).

    Generic reverse-relation detection for relations not in the pattern table.
    Works across domains by detecting common suffix patterns.
    """
    relation_lower = relation_name.lower()

    # "_by" suffix → remove it
    if relation_lower.endswith("_by"):
        base = relation_lower[:-3]
        if base.endswith("ed"):
            return base
        return base + "s" if not base.endswith("s") else base

    # "_of" suffix → "has_<base>"
    if relation_lower.endswith("_of"):
        base = relation_lower[:-3]
        return f"has_{base}"

    # "_from" suffix → remove + try action verb form
    if relation_lower.endswith("_from"):
        base = relation_lower[:-5]
        return base + "s" if not base.endswith("s") else base

    # Direct action verbs → passive form
    if not relation_lower.endswith(("_by", "_of", "_from", "_in", "_at")):
        if relation_lower.endswith("s"):
            return relation_lower[:-1] + "ed_by"
        return relation_lower + "d_by"

    return None


def _add_reverse_relations(
    relation_groups: list[RelationGroup],
    entity_types: list[EntityType],
) -> list[RelationGroup]:
    """Automatically add reverse relations for common bidirectional relationships.

    This allows extraction of relations from both directions, e.g.:
      - "produces" → "produced_by"
      - "owns" → "owned_by"
      - "located_in" → "contains"
    """
    entity_names = {et.name for et in entity_types}
    new_relation_groups: list[RelationGroup] = []

    for rg in relation_groups:
        new_relations = list(rg.relations)
        added_reverse_relations: set[str] = set()

        for rel in rg.relations:
            reverse_name = REVERSE_RELATION_PATTERNS.get(rel.name)
            if reverse_name and reverse_name not in added_reverse_relations:
                exists = any(
                    r.name == reverse_name and r.src == rel.dst and r.dst == rel.src
                    for r in new_relations
                )
                if not exists and rel.dst in entity_names and rel.src in entity_names:
                    new_relations.append(
                        Relation(
                            name=reverse_name,
                            src=rel.dst,
                            dst=rel.src,
                            description=f"Reverse of {rel.name}: {rel.description}",
                            detail=f"Reverse relationship: {rel.detail}",
                        )
                    )
                    added_reverse_relations.add(reverse_name)
                    logger.debug(f"Added reverse relation: {reverse_name} ({rel.dst} -> {rel.src})")

        new_relation_groups.append(
            RelationGroup(
                name=rg.name,
                description=rg.description,
                relations=new_relations,
                examples=rg.examples,
            )
        )

    return new_relation_groups
