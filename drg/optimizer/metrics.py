"""Metrics for DSPy optimizer: entity and relation extraction quality.

Both metrics are compatible with the DSPy ``Metric`` callable signature:
``metric(example, prediction, trace=None) -> float``

Scoring
-------
* :class:`EntityExtractionMetric` — token-level F1 on entity names + types
* :class:`RelationExtractionMetric` — exact-match F1 on (source, relation, target) triples
* :func:`weighted_f1_metric` — 60 % entity + 40 % relation (default composition)
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EntityExtractionMetric",
    "RelationExtractionMetric",
    "weighted_f1_metric",
]


def _normalise(s: str) -> str:
    return s.strip().lower()


def _entity_keys(items: list[Any]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for item in items:
        if isinstance(item, dict):
            name = _normalise(item.get("name", item.get("id", "")))
            typ = _normalise(item.get("type", ""))
            out.add((name, typ))
        elif hasattr(item, "name"):
            out.add((_normalise(item.name), _normalise(getattr(item, "type", "") or "")))
    return out


def _relation_keys(items: list[Any]) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            out.add((_normalise(str(item[0])), _normalise(str(item[1])), _normalise(str(item[2]))))
        elif isinstance(item, dict):
            src = _normalise(item.get("source", item.get("subject", item.get("src", ""))))
            rel = _normalise(
                item.get("relation", item.get("type", item.get("relationship_type", "")))
            )
            tgt = _normalise(item.get("target", item.get("object", item.get("dst", ""))))
            out.add((src, rel, tgt))
        elif hasattr(item, "source"):
            out.add(
                (
                    _normalise(getattr(item, "source", "")),
                    _normalise(getattr(item, "relation", getattr(item, "relationship_type", ""))),
                    _normalise(getattr(item, "target", "")),
                )
            )
    return out


def _f1(gold: set, pred: set) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class EntityExtractionMetric:
    """DSPy-compatible metric for entity extraction quality.

    Computes F1 on (name, type) pairs between gold and predicted entities.

    Parameters
    ----------
    entity_weight:
        Weight of this metric in combined scores (unused internally, for
        documentation purposes).
    """

    def __init__(self, entity_weight: float = 0.6) -> None:
        self.entity_weight = entity_weight

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: Any = None,
    ) -> float:
        gold_raw = (
            getattr(example, "expected_entities", None)
            or getattr(example, "gold_entities", None)
            or []
        )
        pred_raw = (
            getattr(prediction, "entities", None) or getattr(prediction, "entity_list", None) or []
        )
        gold = _entity_keys(list(gold_raw))
        pred = _entity_keys(list(pred_raw))
        return _f1(gold, pred)


class RelationExtractionMetric:
    """DSPy-compatible metric for relation extraction quality.

    Computes F1 on (source, relation, target) triples between gold and
    predicted relations.

    Parameters
    ----------
    relation_weight:
        Weight of this metric in combined scores (documentation only).
    """

    def __init__(self, relation_weight: float = 0.4) -> None:
        self.relation_weight = relation_weight

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: Any = None,
    ) -> float:
        gold_raw = (
            getattr(example, "expected_relations", None)
            or getattr(example, "gold_relations", None)
            or []
        )
        pred_raw = (
            getattr(prediction, "relations", None)
            or getattr(prediction, "relation_list", None)
            or getattr(prediction, "triples", None)
            or []
        )
        gold = _relation_keys(list(gold_raw))
        pred = _relation_keys(list(pred_raw))
        return _f1(gold, pred)


def weighted_f1_metric(
    example: Any,
    prediction: Any,
    trace: Any = None,
    *,
    entity_weight: float = 0.6,
    relation_weight: float = 0.4,
) -> float:
    """Weighted combination: ``entity_weight × entity_F1 + relation_weight × relation_F1``.

    Default weights follow the evaluation framework spec (60 % entity, 40 % relation).
    """
    entity_score = EntityExtractionMetric()(example, prediction, trace)
    relation_score = RelationExtractionMetric()(example, prediction, trace)
    return entity_weight * entity_score + relation_weight * relation_score
