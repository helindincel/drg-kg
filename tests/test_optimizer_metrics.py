"""Unit tests for DSPy optimizer metrics (no LLM required)."""

from __future__ import annotations

from types import SimpleNamespace

from drg.optimizer.metrics import (
    EntityExtractionMetric,
    RelationExtractionMetric,
    weighted_f1_metric,
)


def test_entity_extraction_metric_perfect_match():
    metric = EntityExtractionMetric()
    example = SimpleNamespace(expected_entities=[("Alice", "Person")])
    prediction = SimpleNamespace(entities=[("Alice", "Person")])
    assert metric(example, prediction) == 1.0


def test_entity_extraction_metric_partial_match():
    metric = EntityExtractionMetric()
    example = SimpleNamespace(
        expected_entities=[
            {"name": "Alice", "type": "Person"},
            {"name": "Acme", "type": "Company"},
        ]
    )
    prediction = SimpleNamespace(entities=[{"name": "Alice", "type": "Person"}])
    score = metric(example, prediction)
    assert 0.0 < score < 1.0


def test_relation_extraction_metric_exact_triples():
    metric = RelationExtractionMetric()
    example = SimpleNamespace(expected_relations=[("Alice", "works_at", "Acme")])
    prediction = SimpleNamespace(relations=[("Alice", "works_at", "Acme")])
    assert metric(example, prediction) == 1.0


def test_relation_extraction_metric_dict_shape():
    metric = RelationExtractionMetric()
    example = SimpleNamespace(
        expected_relations=[{"source": "Alice", "relation": "works_at", "target": "Acme"}]
    )
    prediction = SimpleNamespace(
        relations=[{"source": "Alice", "relationship_type": "works_at", "target": "Acme"}]
    )
    assert metric(example, prediction) == 1.0


def test_weighted_f1_metric_combines_entity_and_relation_scores():
    example = SimpleNamespace(
        expected_entities=[("Alice", "Person")],
        expected_relations=[("Alice", "works_at", "Acme")],
    )
    prediction = SimpleNamespace(
        entities=[("Alice", "Person")],
        relations=[("Alice", "works_at", "Acme")],
    )
    assert weighted_f1_metric(example, prediction) == 1.0
