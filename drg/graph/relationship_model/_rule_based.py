"""Rule-based relationship classification.

Cheap, deterministic scoring using keyword patterns + (source_type, target_type)
heuristics. The classifier consults this first; only when results are weak
does it fall through to LLM-based classification.

Schema constraints (when a schema is present) filter the keyword scores so
the classifier never proposes a relation the schema disallows for the given
entity-type pair.
"""

from __future__ import annotations

from ._types import RelationshipType

__all__ = [
    "apply_schema_constraints",
    "build_schema_indexes",
    "classify_rule_based",
]


def build_schema_indexes(schema) -> dict[tuple[str, str], set[str]]:
    """Return a ``(source_type, target_type) -> {relation_names}`` index.

    Works for both :class:`drg.schema.EnhancedDRGSchema` and the legacy
    :class:`drg.schema.DRGSchema`.
    """
    valid_relations: dict[tuple[str, str], set[str]] = {}
    if schema is None:
        return valid_relations

    try:
        if hasattr(schema, "relation_groups"):
            for rg in schema.relation_groups:
                for rel in rg.relations:
                    key = (rel.src, rel.dst)
                    valid_relations.setdefault(key, set()).add(rel.name.lower())
    except AttributeError:
        pass

    if not valid_relations and hasattr(schema, "relations"):
        for rel in schema.relations:
            key = (rel.src, rel.dst)
            valid_relations.setdefault(key, set()).add(rel.name.lower())

    return valid_relations


# Keyword → (RelationshipType, base_confidence) groupings. Defined once at
# module load so the per-classify cost is just a dict lookup + substring scan.
_KEYWORD_RULES: tuple[tuple[tuple[str, ...], RelationshipType, float], ...] = (
    # Causal
    (("causes", "caused", "leads to", "results in"), RelationshipType.CAUSES, 0.9),
    (("because", "due to", "result of"), RelationshipType.CAUSED_BY, 0.9),
    # Spatial
    (("located", "at", "in", "place"), RelationshipType.LOCATED_AT, 0.8),
    (("contains", "includes", "has"), RelationshipType.CONTAINS, 0.8),
    # Temporal
    (("before", "prior", "earlier"), RelationshipType.OCCURS_BEFORE, 0.8),
    (("after", "later", "subsequent"), RelationshipType.OCCURS_AFTER, 0.8),
    # Social
    (("influences", "affects", "impacts"), RelationshipType.INFLUENCES, 0.85),
    (("collaborates", "works with", "partners"), RelationshipType.COLLABORATES_WITH, 0.85),
    (("owns", "possesses", "has ownership"), RelationshipType.OWNS, 0.9),
    (("member", "belongs", "part of group"), RelationshipType.MEMBER_OF, 0.85),
    # Hierarchical
    (("parent", "father", "mother"), RelationshipType.PARENT_OF, 0.9),
    (("child", "son", "daughter"), RelationshipType.CHILD_OF, 0.9),
    (("part of", "component", "belongs to"), RelationshipType.PART_OF, 0.8),
)


def classify_rule_based(
    *,
    source_type: str | None = None,
    target_type: str | None = None,
    raw_relation_text: str | None = None,
) -> list[tuple[RelationshipType, float]]:
    """Score candidate relationship types from keyword patterns and type pairs.

    The scoring is intentionally cheap and explainable: keyword hits in
    ``raw_relation_text`` always win over the type-based fallbacks, and the
    type-based fallbacks only fire when there are no text matches.

    Returns:
        Up to a handful of ``(RelationshipType, confidence)`` tuples,
        unsorted. The caller (:class:`RelationshipTypeClassifier`) is
        responsible for ranking + de-duplication.
    """
    results: list[tuple[RelationshipType, float]] = []

    if raw_relation_text:
        text_lower = raw_relation_text.lower()
        for keywords, rel_type, base_conf in _KEYWORD_RULES:
            if any(word in text_lower for word in keywords):
                results.append((rel_type, base_conf))

    # Type-pair heuristics fire only when keyword matching produced nothing.
    if source_type and target_type and not results:
        if source_type == "Person" and target_type == "Person":
            results.append((RelationshipType.RELATED_TO, 0.5))
            results.append((RelationshipType.KNOWS, 0.4))
        elif source_type == "Person" and target_type == "Location":
            results.append((RelationshipType.LOCATED_AT, 0.6))
            results.append((RelationshipType.VISITS, 0.5))
        elif source_type == "Event" and target_type == "Person":
            results.append((RelationshipType.INFLUENCES, 0.5))
            results.append((RelationshipType.RESULTS_IN, 0.4))

    # Universal fallback — never return empty.
    if not results:
        results.append((RelationshipType.RELATED_TO, 0.3))

    return results


def apply_schema_constraints(
    candidates: list[tuple[RelationshipType, float]],
    *,
    source_type: str,
    target_type: str,
    valid_relations: dict[tuple[str, str], set[str]],
) -> list[tuple[RelationshipType, float]]:
    """Filter candidates against a pre-built schema index.

    Exact name matches keep their confidence; partial matches get a 0.8x
    discount (the relation is in the schema but not what the classifier
    proposed). If the schema has no entry for ``(source_type, target_type)``,
    we leave the candidates untouched — better to surface something than
    silently strip everything.
    """
    if not valid_relations:
        return candidates

    valid_relation_names = valid_relations.get((source_type, target_type), set())
    if not valid_relation_names:
        return candidates

    filtered: list[tuple[RelationshipType, float]] = []
    for rel_type, conf in candidates:
        rel_name = rel_type.value.lower()
        if rel_name in valid_relation_names:
            filtered.append((rel_type, conf))
        elif any(rel_name in valid or valid in rel_name for valid in valid_relation_names):
            filtered.append((rel_type, conf * 0.8))

    return filtered if filtered else candidates
