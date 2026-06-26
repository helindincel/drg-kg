"""Type definitions for the evaluation framework.

Core data model::

    BenchmarkDataset   — a single text + gold-standard annotations
    PipelinePrediction — the output of a pipeline run over one dataset
    EvaluationReport   — aggregated metrics for a full benchmark run
    BenchmarkSuite     — named collection of datasets
    RegressionComparison — diff between two EvaluationReports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BenchmarkDataset",
    "BenchmarkSuite",
    "CalibrationReport",
    "EvaluationReport",
    "MetricResult",
    "PipelinePrediction",
    "RegressionComparison",
]


# ---------------------------------------------------------------------------
# Input: gold-standard benchmark data
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkDataset:
    """A single labelled benchmark example.

    Attributes:
        name: Unique dataset identifier.
        text: The input text to be processed by the pipeline.
        gold_entities: List of expected entities (dicts with ``name`` / ``type``).
        gold_relations: List of expected relation triples (dicts with ``source``,
            ``target``, ``type``).
        gold_events: List of expected events.
        gold_inferred_relations: Relations expected after reasoning.
        gold_communities: Expected cluster/community memberships.
        metadata: Free-form tags (``domain``, ``task``, ``source``, etc.).
    """

    name: str
    text: str
    gold_entities: list[dict[str, Any]] = field(default_factory=list)
    gold_relations: list[dict[str, Any]] = field(default_factory=list)
    gold_events: list[dict[str, Any]] = field(default_factory=list)
    gold_inferred_relations: list[dict[str, Any]] = field(default_factory=list)
    gold_communities: list[dict[str, Any]] | dict[str, Any] = field(default_factory=list)
    gold_evidence: list[dict[str, Any]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "gold_entities": self.gold_entities,
            "gold_relations": self.gold_relations,
            "gold_events": self.gold_events,
            "gold_inferred_relations": self.gold_inferred_relations,
            "gold_communities": self.gold_communities,
            "gold_evidence": self.gold_evidence,
            "documents": self.documents,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkDataset:
        return cls(
            name=data["name"],
            text=data.get("text", ""),
            gold_entities=data.get("gold_entities", []),
            gold_relations=data.get("gold_relations", []),
            gold_events=data.get("gold_events", []),
            gold_inferred_relations=data.get("gold_inferred_relations", []),
            gold_communities=data.get("gold_communities", []),
            gold_evidence=data.get("gold_evidence", data.get("gold_supporting_evidence", [])),
            documents=data.get("documents", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BenchmarkSuite:
    """Named collection of :class:`BenchmarkDataset` instances.

    Attributes:
        name: Suite identifier.
        datasets: Ordered list of benchmark examples.
        adapters: Optional list of adapter names this suite targets.
        metadata: Free-form suite-level metadata.
    """

    name: str
    datasets: list[BenchmarkDataset] = field(default_factory=list)
    adapters: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "datasets": [d.to_dict() for d in self.datasets],
            "adapters": self.adapters,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSuite:
        return cls(
            name=data["name"],
            datasets=[BenchmarkDataset.from_dict(d) for d in data.get("datasets", [])],
            adapters=data.get("adapters", []),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Output: pipeline predictions
# ---------------------------------------------------------------------------


@dataclass
class PipelinePrediction:
    """The output of one pipeline run over a single :class:`BenchmarkDataset`.

    Attributes:
        entities: Extracted entity dicts (``name`` / ``type`` at minimum).
        relations: Extracted relation dicts (``source``, ``target``, ``type``).
        events: Extracted event dicts.
        inferred_relations: Relations after reasoning (if ``--infer`` was used).
        communities: Detected cluster/community memberships.
        metadata: Pipeline metadata (model name, runtime, etc.).
    """

    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    inferred_relations: list[dict[str, Any]] = field(default_factory=list)
    communities: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": self.entities,
            "relations": self.relations,
            "events": self.events,
            "inferred_relations": self.inferred_relations,
            "communities": self.communities,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelinePrediction:
        return cls(
            entities=data.get("entities", []),
            relations=data.get("relations", []),
            events=data.get("events", []),
            inferred_relations=data.get("inferred_relations", []),
            communities=data.get("communities", []),
            evidence=data.get("evidence", data.get("supporting_evidence", [])),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Results: per-category metrics
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Precision / recall / F1 for one evaluation category."""

    precision: float
    recall: float
    f1: float
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }
        if self.details:
            data["details"] = self.details
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResult:
        return cls(
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            f1=data.get("f1", 0.0),
            true_positives=data.get("true_positives", 0),
            false_positives=data.get("false_positives", 0),
            false_negatives=data.get("false_negatives", 0),
            details=data.get("details", {}),
        )


@dataclass
class CalibrationReport:
    """Confidence calibration quality for scored predictions."""

    sample_count: int = 0
    expected_calibration_error: float = 0.0
    brier_score: float = 0.0
    bins: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "bins": self.bins,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CalibrationReport:
        data = data or {}
        return cls(
            sample_count=data.get("sample_count", 0),
            expected_calibration_error=data.get("expected_calibration_error", 0.0),
            brier_score=data.get("brier_score", 0.0),
            bins=data.get("bins", []),
        )


@dataclass
class EvaluationReport:
    """Aggregated evaluation results for a benchmark run.

    Attributes:
        run_id: Unique identifier for this run.
        entity_metrics: Entity extraction F1 metrics.
        relation_metrics: Relation extraction F1 metrics.
        event_metrics: Event extraction F1 metrics.
        reasoning_metrics: Post-reasoning relation F1 metrics.
        weighted_f1: Weighted average F1 (45 % entity, 35 % relation, 20 % event).
        dataset_count: Number of datasets evaluated.
        metadata: Run metadata (model, timestamp, etc.).
        per_dataset: Per-dataset metric breakdowns.
        performance: Optional timing metrics (seconds per call, etc.).
    """

    run_id: str
    entity_metrics: MetricResult = field(default_factory=lambda: MetricResult(0, 0, 0))
    relation_metrics: MetricResult = field(default_factory=lambda: MetricResult(0, 0, 0))
    event_metrics: MetricResult = field(default_factory=lambda: MetricResult(0, 0, 0))
    reasoning_metrics: MetricResult = field(default_factory=lambda: MetricResult(0, 0, 0))
    evidence_metrics: MetricResult = field(default_factory=lambda: MetricResult(0, 0, 0))
    calibration: CalibrationReport = field(default_factory=CalibrationReport)
    weighted_f1: float = 0.0
    dataset_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    per_dataset: list[dict[str, Any]] = field(default_factory=list)
    performance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "aggregate": {
                "extraction_f1": round(self.weighted_f1, 4),
                "entity_f1": round(self.entity_metrics.f1, 4),
                "relation_f1": round(self.relation_metrics.f1, 4),
                "event_f1": round(self.event_metrics.f1, 4),
                "evidence_f1": round(self.evidence_metrics.f1, 4),
            },
            "entity_metrics": self.entity_metrics.to_dict(),
            "relation_metrics": self.relation_metrics.to_dict(),
            "event_metrics": self.event_metrics.to_dict(),
            "reasoning_metrics": self.reasoning_metrics.to_dict(),
            "evidence_metrics": self.evidence_metrics.to_dict(),
            "calibration": self.calibration.to_dict(),
            "weighted_f1": round(self.weighted_f1, 4),
            "dataset_count": self.dataset_count,
            "metadata": self.metadata,
            "per_dataset": self.per_dataset,
            "performance": self.performance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationReport:
        return cls(
            run_id=data["run_id"],
            entity_metrics=MetricResult.from_dict(data.get("entity_metrics", {})),
            relation_metrics=MetricResult.from_dict(data.get("relation_metrics", {})),
            event_metrics=MetricResult.from_dict(data.get("event_metrics", {})),
            reasoning_metrics=MetricResult.from_dict(data.get("reasoning_metrics", {})),
            evidence_metrics=MetricResult.from_dict(data.get("evidence_metrics", {})),
            calibration=CalibrationReport.from_dict(data.get("calibration", {})),
            weighted_f1=data.get("weighted_f1", 0.0),
            dataset_count=data.get("dataset_count", 0),
            metadata=data.get("metadata", {}),
            per_dataset=data.get("per_dataset", []),
            performance=data.get("performance", {}),
        )


# ---------------------------------------------------------------------------
# Regression comparison
# ---------------------------------------------------------------------------


@dataclass
class MetricDelta:
    """Absolute and relative change for a single metric value."""

    metric: str
    baseline: float
    candidate: float
    delta: float
    regressed: bool
    threshold: float = 0.0
    status: str = "unchanged"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": round(self.baseline, 4),
            "candidate": round(self.candidate, 4),
            "delta": round(self.delta, 4),
            "regressed": self.regressed,
            "threshold": round(self.threshold, 4),
            "status": self.status,
        }


@dataclass
class RegressionComparison:
    """Diff between a baseline and a candidate :class:`EvaluationReport`."""

    baseline_run_id: str
    candidate_run_id: str
    regression_threshold: float
    overall_regressed: bool
    deltas: list[MetricDelta] = field(default_factory=list)
    dataset_deltas: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "regression_threshold": self.regression_threshold,
            "overall_regressed": self.overall_regressed,
            "deltas": [d.to_dict() for d in self.deltas],
            "dataset_deltas": self.dataset_deltas,
            "metadata": self.metadata,
        }
