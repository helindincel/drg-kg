"""Relationship type classifier — dispatcher between rule-based and LLM paths.

The class itself stays small: it owns the schema index, decides when to
escalate from rules to LLM, and merges results. Both classification engines
live in their own modules so they can be tested in isolation.
"""

from __future__ import annotations

from typing import Any

from ...utils.logging import get_logger
from . import _llm_based
from ._rule_based import (
    apply_schema_constraints,
    build_schema_indexes,
    classify_rule_based,
)
from ._types import RelationshipType

logger = get_logger(__name__)


class RelationshipTypeClassifier:
    """Determine relationship types using rules + (optional) LLM.

    The dispatcher is conservative about escalating to the LLM: if the
    rule-based engine returns a result with confidence ≥ 0.8 we stop there
    (cheap, deterministic). Below that, we merge in LLM-classified candidates
    and rank by confidence.
    """

    def __init__(self, schema: Any | None = None, use_llm: bool = True):
        self.schema = schema
        self.use_llm = use_llm and _llm_based.is_available()
        self._llm_classifier = None  # lazy
        self._valid_relations: dict[tuple[str, str], set] = build_schema_indexes(schema)

    # Backward-compat shim: the legacy class exposed this method for tests.
    def _build_schema_indexes(self) -> None:
        self._valid_relations = build_schema_indexes(self.schema)

    def classify(
        self,
        source: str,
        target: str,
        source_type: str | None = None,
        target_type: str | None = None,
        raw_relation_text: str | None = None,
        context: str | None = None,
    ) -> list[tuple[RelationshipType, float]]:
        rule_results = classify_rule_based(
            source_type=source_type,
            target_type=target_type,
            raw_relation_text=raw_relation_text,
        )

        if self.schema and source_type and target_type:
            rule_results = apply_schema_constraints(
                rule_results,
                source_type=source_type,
                target_type=target_type,
                valid_relations=self._valid_relations,
            )

        # High-confidence rule-based result is enough — skip the LLM round-trip.
        if rule_results and rule_results[0][1] >= 0.8:
            return rule_results[:3]

        llm_results: list[tuple[RelationshipType, float]] = []
        if self.use_llm:
            if self._llm_classifier is None:
                self._llm_classifier = _llm_based.create_relationship_classifier()
            if self._llm_classifier is None:
                logger.warning("Failed to create LLM classifier, skipping LLM-based classification")
            else:
                llm_results = _llm_based.classify_llm_based(
                    self._llm_classifier,
                    source=source,
                    target=target,
                    source_type=source_type,
                    target_type=target_type,
                    raw_relation_text=raw_relation_text,
                    context=context,
                )

        # De-duplicate by RelationshipType, keeping the higher confidence.
        seen: dict[RelationshipType, float] = {}
        for rel_type, conf in rule_results + llm_results:
            if rel_type not in seen or conf > seen[rel_type]:
                seen[rel_type] = conf

        return sorted(seen.items(), key=lambda x: x[1], reverse=True)[:5]


__all__ = ["RelationshipTypeClassifier"]
