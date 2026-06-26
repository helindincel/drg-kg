"""Default confidence strategy — deterministic placeholder.

The project has not yet wired a calibrated scorer (LLM self-rating /
ensemble disagreement / labelled-data calibration), so this module ships a
*reasonable* heuristic placeholder. The goals are:

1. **Never block the pipeline.** The default strategy must produce scores
   for any extraction output, even sparse ones.
2. **Be easy to override.** Swap the strategy out by passing a different
   :class:`drg.confidence.ConfidenceStrategy` to the builder/extractor.
3. **Stay explainable.** The signal breakdown on every
   :class:`ConfidenceScore` makes it obvious why a value is what it is.

Heuristic summary
-----------------
Entities:
    - Base score 0.6 (LLM-extracted but otherwise unverified).
    - +0.15 if the entity type is in the supplied schema.
    - +0.10 if the entity name appears in the source text (case-insensitive
      whole-word match).
    - +0.05 if the entity name is multi-word (heuristic for higher
      specificity / lower confusability).
    - Capped to 1.0; bounded below by 0.05 to avoid zeroing out legitimate
      entities lacking signals.

Relations:
    - If ``enriched_relations`` already carries a numeric ``confidence``
      (e.g. a future LLM self-rating), respect it verbatim.
    - Otherwise base score 0.5.
    - +0.20 if the relation name is in the schema for ``(s_type, o_type)``.
    - +0.10 if both endpoints have non-empty types.
    - +0.05 if a deterministic temporal cue (``temporal``) was attached.
    - −0.30 if the relation was flagged ``is_negated`` (negated relations
      are kept low-confidence rather than dropped — callers can filter).
    - Bounded to ``[0.05, 1.0]``.

These coefficients are intentionally simple and **not calibrated against
labelled data**; they're a placeholder until a proper scorer lands. See
``docs/confidence_scoring.md`` for the upgrade roadmap.
"""

from __future__ import annotations

import re
from typing import Any

from ..schema import DRGSchema, EnhancedDRGSchema
from ..utils.logging import get_logger
from ._strategy import ConfidenceStrategy, EntityScoreMap, RelationScoreMap
from ._types import ConfidenceScore, clamp_confidence

__all__ = ["DefaultConfidenceStrategy"]

logger = get_logger(__name__)


# Bounds applied after summing signals — keeps values out of the degenerate
# 0.0/1.0 corners which calibrated scorers rarely produce.
_MIN_SCORE = 0.05
_MAX_SCORE = 1.0


def _entity_in_text(text: str, name: str) -> bool:
    """Whole-word case-insensitive presence check.

    Mirrors the ``(?<!\\w)…(?!\\w)`` boundary used elsewhere in the project
    (``_heuristics``, ``builders.extract_evidence_snippet``) to keep entity
    matching consistent across the codebase.
    """
    if not text or not name:
        return False
    pattern = rf"(?i)(?<!\w){re.escape(name)}(?!\w)"
    return re.search(pattern, text) is not None


def _schema_entity_types(
    schema: DRGSchema | EnhancedDRGSchema | None,
) -> set[str]:
    """Return the set of entity type names declared in ``schema`` (if any)."""
    if schema is None:
        return set()
    if isinstance(schema, EnhancedDRGSchema):
        return {et.name for et in schema.entity_types}
    return {e.name for e in getattr(schema, "entities", [])}


def _schema_allows_relation(
    schema: DRGSchema | EnhancedDRGSchema | None,
    rel_name: str,
    s_type: str | None,
    o_type: str | None,
) -> bool:
    """Best-effort schema-validity check (forward direction only).

    Tolerant of partial signals: returns ``False`` when types are missing or
    the schema doesn't expose a lookup. Callers treat this as a
    confidence-boosting hint rather than a hard filter.
    """
    if schema is None or not s_type or not o_type:
        return False
    if isinstance(schema, EnhancedDRGSchema):
        try:
            return schema.is_valid_relation(rel_name, s_type, o_type)
        except Exception:
            logger.debug(
                "schema relation check failed for rel=%r src=%r dst=%r",
                rel_name,
                s_type,
                o_type,
                exc_info=True,
            )
            return False
    rels = getattr(schema, "relations", [])
    return any(
        getattr(r, "name", None) == rel_name
        and getattr(r, "src", None) == s_type
        and getattr(r, "dst", None) == o_type
        for r in rels
    )


class DefaultConfidenceStrategy(ConfidenceStrategy):
    """Deterministic, schema-aware heuristic confidence scorer.

    See module docstring for the heuristic specification. Public surface:

        strategy = DefaultConfidenceStrategy()
        ent_scores = strategy.score_entities(entities, context={"schema": schema, "source_text": text})
        rel_scores = strategy.score_relations(triples, enriched_relations=..., context={"schema": schema})
    """

    name = "default"

    # Coefficients exposed as class attributes so subclasses / tests can
    # tweak them without monkeypatching the methods.
    BASE_ENTITY_SCORE = 0.6
    BOOST_TYPE_IN_SCHEMA = 0.15
    BOOST_NAME_IN_TEXT = 0.10
    BOOST_MULTI_WORD = 0.05

    BASE_RELATION_SCORE = 0.5
    BOOST_SCHEMA_VALID = 0.20
    BOOST_BOTH_TYPED = 0.10
    BOOST_TEMPORAL = 0.05
    PENALTY_NEGATED = 0.30

    def score_entities(
        self,
        entities: list[tuple[str, str]],
        *,
        context: dict[str, Any] | None = None,
    ) -> EntityScoreMap:
        ctx = context or {}
        schema = ctx.get("schema")
        source_text = ctx.get("source_text") or ""
        schema_types = _schema_entity_types(schema)

        scores: EntityScoreMap = {}
        for name, etype in entities:
            if not name:
                continue
            signals: dict[str, float] = {"base": self.BASE_ENTITY_SCORE}
            score = self.BASE_ENTITY_SCORE

            if etype and etype in schema_types:
                score += self.BOOST_TYPE_IN_SCHEMA
                signals["type_in_schema"] = self.BOOST_TYPE_IN_SCHEMA

            if source_text and _entity_in_text(source_text, name):
                score += self.BOOST_NAME_IN_TEXT
                signals["name_in_text"] = self.BOOST_NAME_IN_TEXT

            if len(name.split()) >= 2:
                score += self.BOOST_MULTI_WORD
                signals["multi_word"] = self.BOOST_MULTI_WORD

            score = max(_MIN_SCORE, min(_MAX_SCORE, score))
            # ``clamp_confidence`` re-applies bounds defensively — cheap and
            # ensures NaN guards apply even if subclasses change the maths.
            scores[name] = ConfidenceScore(
                value=clamp_confidence(score),
                signals=signals,
                method=self.name,
            )
        return scores

    def score_relations(
        self,
        relations: list[tuple[str, str, str]],
        *,
        enriched_relations: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> RelationScoreMap:
        ctx = context or {}
        schema = ctx.get("schema")
        # ``entity_types`` lets us short-circuit the schema-validity boost
        # when types are unknown, which is the common case in legacy paths.
        entity_types: dict[str, str] = ctx.get("entity_types") or {}

        # Build a quick lookup from triple -> enriched dict (preserves the
        # existing "enriched_relations" contract of `extract_typed`).
        enriched_by_triple: dict[tuple[str, str, str], dict[str, Any]] = {}
        for er in enriched_relations or []:
            rel = er.get("relation")
            if isinstance(rel, tuple) and len(rel) == 3:
                enriched_by_triple[rel] = er

        scores: RelationScoreMap = {}
        for triple in relations:
            if not (isinstance(triple, tuple) and len(triple) == 3):
                continue
            s, r, o = triple
            er = enriched_by_triple.get(triple, {})

            # If a numeric confidence was provided upstream (LLM self-rating
            # in the future), honour it without re-scoring — strategies
            # compose top-down rather than overwriting earlier signals.
            upstream_conf = er.get("confidence")
            if isinstance(upstream_conf, (int, float)):
                scores[triple] = ConfidenceScore(
                    value=clamp_confidence(float(upstream_conf)),
                    signals={"upstream": float(upstream_conf)},
                    method=f"{self.name}+upstream",
                )
                continue

            signals: dict[str, float] = {"base": self.BASE_RELATION_SCORE}
            score = self.BASE_RELATION_SCORE

            s_type = entity_types.get(s)
            o_type = entity_types.get(o)
            if s_type and o_type:
                score += self.BOOST_BOTH_TYPED
                signals["both_typed"] = self.BOOST_BOTH_TYPED
                if _schema_allows_relation(schema, r, s_type, o_type):
                    score += self.BOOST_SCHEMA_VALID
                    signals["schema_valid"] = self.BOOST_SCHEMA_VALID

            if er.get("temporal"):
                score += self.BOOST_TEMPORAL
                signals["temporal"] = self.BOOST_TEMPORAL

            if er.get("is_negated"):
                score -= self.PENALTY_NEGATED
                signals["negated"] = -self.PENALTY_NEGATED

            score = max(_MIN_SCORE, min(_MAX_SCORE, score))
            scores[triple] = ConfidenceScore(
                value=clamp_confidence(score),
                signals=signals,
                method=self.name,
            )
        return scores
