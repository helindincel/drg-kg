from __future__ import annotations

from drg.evaluation import OntologyEvaluationResult, OntologyMetricEvaluator, evaluate_ontology
from drg.schema import EnhancedDRGSchema, EntityType, Relation, RelationGroup


def _sample_schema() -> EnhancedDRGSchema:
    return EnhancedDRGSchema(
        entity_types=[
            EntityType(
                name="Research Method",
                description="A reusable approach used to investigate a question.",
                examples=["survey"],
                properties={"scope": "The setting where the method applies."},
            ),
            EntityType(
                name="Finding",
                description="A conclusion or observed result produced by analysis.",
                examples=["higher retention"],
                properties={"confidence": "How strongly the finding is supported."},
            ),
            EntityType(
                name="Research Activity",
                description="An event-like activity with participants and timing.",
                examples=["field study"],
            ),
        ],
        relation_groups=[
            RelationGroup(
                name="Research Relations",
                description="Reusable research graph relations.",
                relations=[
                    Relation(
                        name="produces",
                        src="Research Method",
                        dst="Finding",
                        description="Connects a method to a finding it produces.",
                        detail="The survey produced a retention finding.",
                        properties={"timeframe": "The period during which the relation holds."},
                    ),
                    Relation(
                        name="occurs_during",
                        src="Research Activity",
                        dst="Research Method",
                        description="Links an activity to the method used during it.",
                        detail="The field study used the survey method.",
                    ),
                ],
            )
        ],
    )


def test_evaluate_ontology_returns_complete_report():
    report = evaluate_ontology(_sample_schema())

    assert 0.0 <= report.overall_score <= 1.0
    assert set(report.per_metric_scores) == {
        "semantic_completeness",
        "semantic_consistency",
        "reusability",
        "relation_quality",
        "property_modeling_quality",
        "event_modeling_quality",
        "extraction_suitability",
    }
    payload = report.to_dict()
    assert payload["overall_score"] == round(report.overall_score, 4)
    assert payload["metrics"][0]["score"] == report.per_metric_scores["semantic_completeness"]


def test_semantic_consistency_flags_invalid_relation_endpoints_from_dict():
    report = evaluate_ontology(
        {
            "entity_types": [{"name": "Person", "description": "A human actor in text."}],
            "relation_groups": [
                {
                    "name": "Relations",
                    "relations": [
                        {
                            "name": "works_at",
                            "source": "Person",
                            "target": "Missing Organization",
                            "description": "Employment between person and organization.",
                        }
                    ],
                }
            ],
        }
    )

    consistency = next(
        result for result in report.metrics if result.metric == "semantic_consistency"
    )
    assert consistency.score < 1.0
    assert any("invalid" in finding for finding in consistency.findings)
    assert any("source and target" in recommendation for recommendation in report.recommendations)


def test_custom_metric_can_be_added_without_architecture_changes():
    class ConstantMetric:
        name = "constant"
        weight = 2.0

        def evaluate(self, _context):
            return OntologyEvaluationResult(
                metric=self.name,
                score=0.25,
                explanation="Custom metric result.",
                recommendations=("Custom recommendation.",),
                weight=self.weight,
            )

    metric: OntologyMetricEvaluator = ConstantMetric()
    report = evaluate_ontology(_sample_schema(), evaluators=[metric])

    assert report.overall_score == 0.25
    assert report.per_metric_scores == {"constant": 0.25}
    assert report.recommendations == ("Custom recommendation.",)
