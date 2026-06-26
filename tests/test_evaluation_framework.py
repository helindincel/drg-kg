from __future__ import annotations

import json

import pytest

from drg.evaluation import (
    BenchmarkDataset,
    BenchmarkRunner,
    EvaluationReport,
    MetricResult,
    PipelinePrediction,
    compare_reports,
    evaluate_graph_quality,
    load_prediction_artifact,
)
from drg.graph.diff import diff_graph_data


def test_runner_scores_alias_aware_relations_events_evidence_and_calibration():
    dataset = BenchmarkDataset(
        name="alias-event",
        text="Microsoft acquired GitHub in 2018.",
        gold_entities=[
            {
                "name": "Microsoft",
                "type": "Organization",
                "aliases": ["MSFT", "MS"],
                "span": [0, 9],
            },
            {"name": "GitHub", "type": "Organization"},
        ],
        gold_relations=[
            {"source": "Microsoft", "relationship_type": "ACQUIRED", "target": "GitHub"}
        ],
        gold_events=[
            {
                "event_type": "Acquisition",
                "participants": {"acquirer": ["Microsoft"], "acquired": ["GitHub"]},
                "timestamp": "2018",
            }
        ],
        gold_evidence=[{"fact_id": "rel:1", "snippet": "Microsoft acquired GitHub in 2018."}],
        documents=[{"id": "doc-1", "text": "Microsoft acquired GitHub in 2018."}],
        metadata={"domain": "business", "difficulty": "smoke"},
    )
    prediction = PipelinePrediction(
        entities=[
            {
                "name": "MS",
                "type": "Organization",
                "aliases": ["Microsoft"],
                "confidence": 0.9,
            },
            {"name": "GitHub", "type": "Organization", "confidence": 0.8},
        ],
        relations=[
            {
                "source": "MS",
                "relationship_type": "ACQUIRED",
                "target": "GitHub",
                "confidence": 0.85,
            }
        ],
        events=[
            {
                "event_type": "Acquisition",
                "participants": {"acquirer": ["MS"], "acquired": ["GitHub"]},
                "timestamp": "2018",
                "confidence": 0.7,
            }
        ],
        evidence=[{"fact_id": "rel:1", "snippet": "Microsoft acquired GitHub in 2018."}],
    )

    report = BenchmarkRunner(run_id="candidate").evaluate([dataset], predictions=[prediction])

    assert report.entity_metrics.f1 == pytest.approx(1.0)
    assert report.relation_metrics.f1 == pytest.approx(1.0)
    assert report.event_metrics.f1 == pytest.approx(1.0)
    assert report.evidence_metrics.f1 == pytest.approx(1.0)
    assert report.weighted_f1 == pytest.approx(1.0)
    assert report.calibration.sample_count == 4
    assert report.per_dataset[0]["multi_document"]["document_count"] == 1


def test_prediction_artifact_dict_is_matched_by_dataset_name(tmp_path):
    datasets = [
        BenchmarkDataset(
            name="alpha",
            text="Alpha Corp ships a product.",
            gold_entities=[{"name": "Alpha Corp", "type": "Company"}],
        ),
        BenchmarkDataset(
            name="beta",
            text="Beta Person wrote a memo.",
            gold_entities=[{"name": "Beta Person", "type": "Person"}],
        ),
    ]
    artifact = {
        "predictions": {
            "beta": {"entities": [{"name": "Beta Person", "type": "Person"}]},
            "alpha": {"entities": [{"name": "Alpha Corp", "type": "Company"}]},
        }
    }
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    predictions, _metadata = load_prediction_artifact(path)
    report = BenchmarkRunner(run_id="candidate").evaluate(datasets, predictions=predictions)

    assert [row["dataset"] for row in report.per_dataset] == ["alpha", "beta"]
    assert report.entity_metrics.f1 == pytest.approx(1.0)


def test_duplicate_relation_predictions_count_as_false_positives():
    dataset = BenchmarkDataset(
        name="duplicate-relation",
        text="Apple acquired Beats.",
        gold_entities=[
            {"name": "Apple", "type": "Company"},
            {"name": "Beats", "type": "Company"},
        ],
        gold_relations=[{"source": "Apple", "relationship_type": "ACQUIRED", "target": "Beats"}],
    )
    prediction = PipelinePrediction(
        entities=[
            {"name": "Apple", "type": "Company"},
            {"name": "Beats", "type": "Company"},
        ],
        relations=[
            {"source": "Apple", "relationship_type": "ACQUIRED", "target": "Beats"},
            {"source": "Apple", "relationship_type": "ACQUIRED", "target": "Beats"},
        ],
    )

    report = BenchmarkRunner(run_id="candidate").evaluate([dataset], predictions=[prediction])

    assert report.relation_metrics.true_positives == 1
    assert report.relation_metrics.false_positives == 1
    assert report.relation_metrics.false_negatives == 0
    assert report.relation_metrics.f1 == pytest.approx(2 / 3)


def test_relations_events_and_reasoning_require_matched_entities_when_entities_are_labelled():
    dataset = BenchmarkDataset(
        name="relation-without-entity",
        text="Microsoft acquired GitHub in 2018.",
        gold_entities=[
            {"name": "Microsoft", "type": "Organization"},
            {"name": "GitHub", "type": "Organization"},
        ],
        gold_relations=[
            {"source": "Microsoft", "relationship_type": "ACQUIRED", "target": "GitHub"}
        ],
        gold_events=[
            {
                "event_type": "Acquisition",
                "participants": {"acquirer": ["Microsoft"], "acquired": ["GitHub"]},
                "timestamp": "2018",
            }
        ],
        gold_inferred_relations=[
            {"source": "Microsoft", "relationship_type": "CONNECTED_TO", "target": "GitHub"}
        ],
    )
    prediction = PipelinePrediction(
        entities=[],
        relations=[{"source": "Microsoft", "relationship_type": "ACQUIRED", "target": "GitHub"}],
        events=[
            {
                "event_type": "Acquisition",
                "participants": {"acquirer": ["Microsoft"], "acquired": ["GitHub"]},
                "timestamp": "2018",
            }
        ],
        inferred_relations=[
            {"source": "Microsoft", "relationship_type": "CONNECTED_TO", "target": "GitHub"}
        ],
    )

    report = BenchmarkRunner(run_id="candidate").evaluate([dataset], predictions=[prediction])

    assert report.relation_metrics.true_positives == 0
    assert report.relation_metrics.false_positives == 1
    assert report.relation_metrics.false_negatives == 1
    assert report.event_metrics.true_positives == 0
    assert report.event_metrics.false_positives == 1
    assert report.event_metrics.false_negatives == 1
    assert report.reasoning_metrics.true_positives == 0
    assert report.reasoning_metrics.false_positives == 1
    assert report.reasoning_metrics.false_negatives == 1


def test_compare_reports_supports_metric_and_dataset_thresholds():
    baseline = EvaluationReport(
        run_id="baseline",
        entity_metrics=MetricResult(1.0, 1.0, 0.9),
        relation_metrics=MetricResult(1.0, 1.0, 0.8),
        event_metrics=MetricResult(1.0, 1.0, 0.7),
        evidence_metrics=MetricResult(1.0, 1.0, 0.6),
        weighted_f1=0.82,
        dataset_count=1,
        per_dataset=[{"dataset": "business", "entity_f1": 0.9, "relation_f1": 0.8}],
    )
    candidate = EvaluationReport(
        run_id="candidate",
        entity_metrics=MetricResult(1.0, 1.0, 0.86),
        relation_metrics=MetricResult(1.0, 1.0, 0.8),
        event_metrics=MetricResult(1.0, 1.0, 0.7),
        evidence_metrics=MetricResult(1.0, 1.0, 0.6),
        weighted_f1=0.8,
        dataset_count=1,
        per_dataset=[{"dataset": "business", "entity_f1": 0.86, "relation_f1": 0.8}],
    )

    comparison = compare_reports(
        baseline,
        candidate,
        regression_threshold=0.05,
        metric_thresholds={"entity_f1": 0.02},
        dataset_thresholds={"business": 0.02},
    )

    assert comparison.overall_regressed is True
    assert next(delta for delta in comparison.deltas if delta.metric == "entity_f1").regressed
    assert comparison.dataset_deltas[0]["regressed"] is True


def test_graph_diff_reports_semantic_confidence_provenance_and_evidence_changes():
    old = {
        "nodes": [
            {
                "id": "a",
                "type": "Company",
                "confidence": 0.6,
                "metadata": {"provenance": {"document_id": "doc-1"}, "evidence": "old"},
            }
        ],
        "edges": [
            {
                "source": "a",
                "relationship_type": "WORKS_WITH",
                "target": "b",
                "confidence": 0.4,
                "metadata": {"provenance": {"document_id": "doc-1"}, "evidence": "old edge"},
            }
        ],
        "clusters": [],
    }
    new = {
        "nodes": [
            {
                "id": "a",
                "type": "Organization",
                "confidence": 0.9,
                "metadata": {"provenance": {"document_id": "doc-2"}, "evidence": "new"},
            }
        ],
        "edges": [
            {
                "source": "a",
                "relationship_type": "WORKS_WITH",
                "target": "b",
                "confidence": 0.8,
                "metadata": {"provenance": {"document_id": "doc-2"}, "evidence": "new edge"},
            }
        ],
        "clusters": [],
    }

    diff = diff_graph_data(old, new)

    assert diff.changed
    assert diff.node_semantic_changes == [
        {"node": "a", "fields": ["type", "confidence", "provenance", "evidence"]}
    ]
    assert diff.edge_semantic_changes == [
        {
            "edge": ("a", "WORKS_WITH", "b"),
            "fields": ["confidence", "provenance", "evidence"],
        }
    ]


def test_graph_quality_evaluator_reacts_to_controlled_bad_graph():
    schema = {
        "entity_types": [
            {"name": "Company"},
            {"name": "Product"},
        ],
        "relation_groups": [
            {
                "name": "Business Relations",
                "relations": [
                    {
                        "name": "PRODUCES",
                        "source": "Company",
                        "target": "Product",
                    }
                ],
            }
        ],
    }
    clean_graph = {
        "nodes": [
            {
                "id": "Apple",
                "type": "Company",
                "confidence": 0.92,
                "metadata": {
                    "evidence": "Apple produces the iPhone.",
                    "provenance": {"document_id": "doc-1", "source_span": [0, 26]},
                },
            },
            {
                "id": "iPhone",
                "type": "Product",
                "confidence": 0.88,
                "metadata": {
                    "evidence": "Apple produces the iPhone.",
                    "provenance": {"document_id": "doc-1", "source_span": [0, 26]},
                },
            },
        ],
        "edges": [
            {
                "source": "Apple",
                "relationship_type": "PRODUCES",
                "target": "iPhone",
                "relationship_detail": "Apple produces the iPhone.",
                "confidence": 0.9,
                "metadata": {
                    "evidence": "Apple produces the iPhone.",
                    "provenance": {"document_id": "doc-1", "source_span": [0, 26]},
                },
            }
        ],
        "clusters": [],
    }
    bad_graph = {
        "nodes": [
            {
                "id": "Apple",
                "type": "Company",
                "confidence": 0,
                "metadata": {},
            },
            {
                "id": "Apple",
                "type": "Company",
                "confidence": 0.4,
                "metadata": {
                    "evidence": "Duplicate Apple node.",
                    "provenance": {"document_id": "doc-1"},
                },
            },
            {
                "id": "Mystery",
                "type": "UnknownType",
                "confidence": 0.5,
                "metadata": {
                    "evidence": "Mystery appears.",
                    "provenance": {"document_id": "doc-1"},
                },
            },
            {
                "id": "iPhone",
                "type": "Product",
                "confidence": 0.8,
                "metadata": {
                    "evidence": "Apple produces the iPhone.",
                    "provenance": {"document_id": "doc-1"},
                },
            },
        ],
        "edges": [
            {
                "source": "Apple",
                "relationship_type": "FOUNDED_BY",
                "target": "iPhone",
                "relationship_detail": "Invalid relation for the schema.",
                "confidence": 0,
                "metadata": {},
            }
        ],
        "clusters": [],
    }

    clean_report = evaluate_graph_quality(clean_graph, schema=schema)
    bad_report = evaluate_graph_quality(bad_graph, schema=schema)
    bad_payload = bad_report.to_dict()
    findings = "\n".join(bad_report.findings)
    recommendations = "\n".join(bad_report.recommendations).lower()

    assert clean_report.overall_score == pytest.approx(1.0)
    assert bad_report.overall_score < clean_report.overall_score
    assert bad_report.overall_score < 0.5
    assert "duplicate_node_id" in findings
    assert "unknown_entity_type" in findings
    assert "invalid_relation" in findings
    assert "zero_confidence" in findings
    assert "missing_evidence" in findings
    assert "missing_provenance" in findings
    assert "deduplicate" in recommendations
    assert "ontology" in recommendations
    assert "evidence" in recommendations
    assert "provenance" in recommendations
    assert bad_payload["checks"]["zero_confidence"] == 2
