"""Confidence framework — strategy contract.

Strategies turn extraction signals into per-entity / per-relationship
confidence scores. Implementations are intentionally pluggable so the
project can evolve from the placeholder heuristics in
:mod:`drg.confidence._default` to richer scorers (LLM self-rated, ensemble,
calibrated probability, etc.) without changing call sites.

The contract is deliberately minimal:

- Inputs are plain Python primitives (``list[tuple[...]]`` / ``list[dict]``)
  so strategies stay decoupled from the graph data model.
- Outputs are mappings keyed by entity name / triple. Callers (the builder
  and the extractor) apply the resulting scores onto ``KGNode`` /
  ``KGEdge`` instances, which is where confidence ultimately lives.

This separation lets the strategy layer be pure & easily unit-testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ._types import ConfidenceScore

__all__ = ["ConfidenceStrategy", "EntityScoreMap", "RelationScoreMap"]


# Keyed by entity name. Multiple entities sharing a name collapse onto the
# same key — the same convention `KGNode` follows internally.
EntityScoreMap = dict[str, ConfidenceScore]
# Keyed by ``(source, relation, target)`` tuple — matches how the
# extraction pipeline (``extract_typed`` / ``extract_from_chunks``) emits
# triples and how ``EnrichedRelationship`` instances are addressed.
RelationScoreMap = dict[tuple[str, str, str], ConfidenceScore]


class ConfidenceStrategy(ABC):
    """Pluggable scorer for entities and relationships.

    Implementations should be deterministic and side-effect-free given the
    same inputs. Heavy / IO-bound strategies (LLM critics, embedding
    similarity, etc.) are allowed but should respect the project's strict
    mode + throttle conventions just like the rest of the pipeline.
    """

    name: str = "abstract"

    @abstractmethod
    def score_entities(
        self,
        entities: list[tuple[str, str]],
        *,
        context: dict[str, Any] | None = None,
    ) -> EntityScoreMap:
        """Return ``entity_name -> ConfidenceScore`` for the given entities.

        ``context`` is an open-ended dict the caller may use to pass
        ancillary signals (source text, schema, chunk metadata, …). New
        signals can be added without changing the contract.
        """

    @abstractmethod
    def score_relations(
        self,
        relations: list[tuple[str, str, str]],
        *,
        enriched_relations: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> RelationScoreMap:
        """Return ``(s, r, o) -> ConfidenceScore`` for the given triples.

        ``enriched_relations`` mirrors the optional structure produced by
        :class:`drg.extract.KGExtractor` (``[{"relation": (s, r, o),
        "confidence": float | None, "is_negated": bool, ...}, ...]``) and is
        the primary integration point for strategies that want to leverage
        existing post-extraction heuristics.
        """
