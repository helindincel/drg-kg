"""Evaluation metrics for extraction quality."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractionMetrics:
    """Metrics for entity and relation extraction."""

    # Entity metrics
    entity_precision: float
    entity_recall: float
    entity_f1: float

    # Relation metrics
    relation_precision: float
    relation_recall: float
    relation_f1: float

    # Combined metrics
    precision: float
    recall: float
    f1: float
    accuracy: float

    # Additional details
    total_expected_entities: int
    total_predicted_entities: int
    total_expected_relations: int
    total_predicted_relations: int


def calculate_metrics(
    expected_entities: list[tuple[str, str]],
    predicted_entities: list[tuple[str, str]],
    expected_relations: list[tuple[str, str, str]],
    predicted_relations: list[tuple[str, str, str]],
) -> ExtractionMetrics:
    """Calculate extraction metrics.

    Args:
        expected_entities: Ground truth entities
        predicted_entities: Predicted entities
        expected_relations: Ground truth relations
        predicted_relations: Predicted relations

    Returns:
        ExtractionMetrics object
    """
    # Convert to sets for comparison
    expected_entities_set = set(expected_entities)
    predicted_entities_set = set(predicted_entities)
    expected_relations_set = set(expected_relations)
    predicted_relations_set = set(predicted_relations)

    # Entity metrics
    entity_intersection = expected_entities_set & predicted_entities_set
    entity_precision = (
        len(entity_intersection) / len(predicted_entities_set) if predicted_entities_set else 0.0
    )
    entity_recall = (
        len(entity_intersection) / len(expected_entities_set) if expected_entities_set else 0.0
    )
    entity_f1 = (
        2 * entity_precision * entity_recall / (entity_precision + entity_recall)
        if (entity_precision + entity_recall) > 0
        else 0.0
    )

    # Relation metrics
    relation_intersection = expected_relations_set & predicted_relations_set
    relation_precision = (
        len(relation_intersection) / len(predicted_relations_set)
        if predicted_relations_set
        else 0.0
    )
    relation_recall = (
        len(relation_intersection) / len(expected_relations_set) if expected_relations_set else 0.0
    )
    relation_f1 = (
        2 * relation_precision * relation_recall / (relation_precision + relation_recall)
        if (relation_precision + relation_recall) > 0
        else 0.0
    )

    # Combined metrics (weighted average)
    precision = 0.6 * entity_precision + 0.4 * relation_precision
    recall = 0.6 * entity_recall + 0.4 * relation_recall
    f1 = 0.6 * entity_f1 + 0.4 * relation_f1

    # Accuracy (exact match)
    total_expected = len(expected_entities_set) + len(expected_relations_set)
    total_correct = len(entity_intersection) + len(relation_intersection)
    accuracy = total_correct / total_expected if total_expected > 0 else 0.0

    return ExtractionMetrics(
        entity_precision=entity_precision,
        entity_recall=entity_recall,
        entity_f1=entity_f1,
        relation_precision=relation_precision,
        relation_recall=relation_recall,
        relation_f1=relation_f1,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        total_expected_entities=len(expected_entities_set),
        total_predicted_entities=len(predicted_entities_set),
        total_expected_relations=len(expected_relations_set),
        total_predicted_relations=len(predicted_relations_set),
    )


def compare_metrics(
    before: ExtractionMetrics,
    after: ExtractionMetrics,
) -> dict[str, Any]:
    """Compare two sets of metrics.

    Args:
        before: Metrics before optimization
        after: Metrics after optimization

    Returns:
        Comparison dictionary with absolute and percentage improvements
    """
    return {
        "entity": {
            "precision": {
                "before": before.entity_precision,
                "after": after.entity_precision,
                "improvement": after.entity_precision - before.entity_precision,
                "improvement_percent": (
                    (after.entity_precision - before.entity_precision)
                    / before.entity_precision
                    * 100
                    if before.entity_precision > 0
                    else 0.0
                ),
            },
            "recall": {
                "before": before.entity_recall,
                "after": after.entity_recall,
                "improvement": after.entity_recall - before.entity_recall,
                "improvement_percent": (
                    (after.entity_recall - before.entity_recall) / before.entity_recall * 100
                    if before.entity_recall > 0
                    else 0.0
                ),
            },
            "f1": {
                "before": before.entity_f1,
                "after": after.entity_f1,
                "improvement": after.entity_f1 - before.entity_f1,
                "improvement_percent": (
                    (after.entity_f1 - before.entity_f1) / before.entity_f1 * 100
                    if before.entity_f1 > 0
                    else 0.0
                ),
            },
        },
        "relation": {
            "precision": {
                "before": before.relation_precision,
                "after": after.relation_precision,
                "improvement": after.relation_precision - before.relation_precision,
                "improvement_percent": (
                    (after.relation_precision - before.relation_precision)
                    / before.relation_precision
                    * 100
                    if before.relation_precision > 0
                    else 0.0
                ),
            },
            "recall": {
                "before": before.relation_recall,
                "after": after.relation_recall,
                "improvement": after.relation_recall - before.relation_recall,
                "improvement_percent": (
                    (after.relation_recall - before.relation_recall) / before.relation_recall * 100
                    if before.relation_recall > 0
                    else 0.0
                ),
            },
            "f1": {
                "before": before.relation_f1,
                "after": after.relation_f1,
                "improvement": after.relation_f1 - before.relation_f1,
                "improvement_percent": (
                    (after.relation_f1 - before.relation_f1) / before.relation_f1 * 100
                    if before.relation_f1 > 0
                    else 0.0
                ),
            },
        },
        "combined": {
            "precision": {
                "before": before.precision,
                "after": after.precision,
                "improvement": after.precision - before.precision,
                "improvement_percent": (
                    (after.precision - before.precision) / before.precision * 100
                    if before.precision > 0
                    else 0.0
                ),
            },
            "recall": {
                "before": before.recall,
                "after": after.recall,
                "improvement": after.recall - before.recall,
                "improvement_percent": (
                    (after.recall - before.recall) / before.recall * 100
                    if before.recall > 0
                    else 0.0
                ),
            },
            "f1": {
                "before": before.f1,
                "after": after.f1,
                "improvement": after.f1 - before.f1,
                "improvement_percent": (
                    (after.f1 - before.f1) / before.f1 * 100 if before.f1 > 0 else 0.0
                ),
            },
            "accuracy": {
                "before": before.accuracy,
                "after": after.accuracy,
                "improvement": after.accuracy - before.accuracy,
                "improvement_percent": (
                    (after.accuracy - before.accuracy) / before.accuracy * 100
                    if before.accuracy > 0
                    else 0.0
                ),
            },
        },
    }
