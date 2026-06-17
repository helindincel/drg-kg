"""Data models for benchmark datasets and evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GoldEntity:
    """Gold-standard entity annotation."""

    name: str
    type: str | None = None

    def key(self, *, typed: bool = True) -> tuple[str, str | None] | str:
        name = _norm(self.name)
        return (name, _norm(self.type)) if typed else name

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.type is not None:
            out["type"] = self.type
        return out

    @classmethod
    def from_any(cls, value: Any) -> "GoldEntity":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(name=str(value["name"]), type=value.get("type"))
        if isinstance(value, (list, tuple)) and value:
            return cls(name=str(value[0]), type=str(value[1]) if len(value) > 1 else None)
        return cls(name=str(value))


@dataclass(frozen=True)
class GoldRelation:
    """Gold-standard relationship annotation."""

    source: str
    relationship_type: str
    target: str

    def key(self) -> tuple[str, str, str]:
        return (_norm(self.source), _norm(self.relationship_type), _norm(self.target))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "relationship_type": self.relationship_type,
            "target": self.target,
        }

    @classmethod
    def from_any(cls, value: Any) -> "GoldRelation":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                source=str(value["source"]),
                relationship_type=str(value.get("relationship_type") or value.get("relation")),
                target=str(value["target"]),
            )
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return cls(source=str(value[0]), relationship_type=str(value[1]), target=str(value[2]))
        raise ValueError(f"Invalid relation annotation: {value!r}")


@dataclass(frozen=True)
class GoldEvent:
    """Gold-standard event annotation."""

    id: str | None = None
    event_type: str = ""
    participants: dict[str, list[str]] = field(default_factory=dict)
    timestamp: str | None = None
    location: str | None = None

    def key(self) -> tuple[Any, ...]:
        participant_key = tuple(
            sorted(
                (_norm(role), tuple(sorted(_norm(x) for x in values)))
                for role, values in self.participants.items()
            )
        )
        return (
            _norm(self.event_type),
            participant_key,
            _norm(self.timestamp),
            _norm(self.location),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "event_type": self.event_type,
            "participants": {k: list(v) for k, v in self.participants.items()},
        }
        if self.id is not None:
            out["id"] = self.id
        if self.timestamp is not None:
            out["timestamp"] = self.timestamp
        if self.location is not None:
            out["location"] = self.location
        return out

    @classmethod
    def from_any(cls, value: Any) -> "GoldEvent":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"Invalid event annotation: {value!r}")
        participants: dict[str, list[str]] = {}
        raw = value.get("participants") or {}
        if isinstance(raw, dict):
            for role, vals in raw.items():
                if isinstance(vals, list):
                    participants[str(role)] = [str(v) for v in vals]
                elif vals is not None:
                    participants[str(role)] = [str(vals)]
        ts = value.get("timestamp")
        if isinstance(ts, dict):
            ts = ts.get("start") or ts.get("value")
        return cls(
            id=value.get("id"),
            event_type=str(value.get("event_type") or value.get("type") or ""),
            participants=participants,
            timestamp=str(ts) if ts is not None else None,
            location=str(value["location"]) if value.get("location") is not None else None,
        )


@dataclass(frozen=True)
class QueryCase:
    """Gold-standard query/retrieval expectation."""

    query: str
    relevant_entities: list[str] = field(default_factory=list)
    relevant_relations: list[GoldRelation] = field(default_factory=list)
    relevant_chunks: list[str] = field(default_factory=list)
    expected_answer_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "relevant_entities": list(self.relevant_entities),
            "relevant_relations": [r.to_dict() for r in self.relevant_relations],
            "relevant_chunks": list(self.relevant_chunks),
            "expected_answer_entities": list(self.expected_answer_entities),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryCase":
        return cls(
            query=str(data["query"]),
            relevant_entities=[str(x) for x in data.get("relevant_entities", [])],
            relevant_relations=[
                GoldRelation.from_any(x) for x in data.get("relevant_relations", [])
            ],
            relevant_chunks=[str(x) for x in data.get("relevant_chunks", [])],
            expected_answer_entities=[
                str(x) for x in data.get("expected_answer_entities", [])
            ],
        )


@dataclass(frozen=True)
class BenchmarkDataset:
    """A single reproducible benchmark dataset."""

    name: str
    text: str = ""
    gold_entities: list[GoldEntity] = field(default_factory=list)
    gold_relations: list[GoldRelation] = field(default_factory=list)
    gold_events: list[GoldEvent] = field(default_factory=list)
    gold_inferred_relations: list[GoldRelation] = field(default_factory=list)
    query_cases: list[QueryCase] = field(default_factory=list)
    gold_communities: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "gold_entities": [e.to_dict() for e in self.gold_entities],
            "gold_relations": [r.to_dict() for r in self.gold_relations],
            "gold_events": [e.to_dict() for e in self.gold_events],
            "gold_inferred_relations": [r.to_dict() for r in self.gold_inferred_relations],
            "query_cases": [q.to_dict() for q in self.query_cases],
            "gold_communities": dict(self.gold_communities),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkDataset":
        return cls(
            name=str(data["name"]),
            text=str(data.get("text", "")),
            gold_entities=[GoldEntity.from_any(x) for x in data.get("gold_entities", [])],
            gold_relations=[GoldRelation.from_any(x) for x in data.get("gold_relations", [])],
            gold_events=[GoldEvent.from_any(x) for x in data.get("gold_events", [])],
            gold_inferred_relations=[
                GoldRelation.from_any(x) for x in data.get("gold_inferred_relations", [])
            ],
            query_cases=[QueryCase.from_dict(x) for x in data.get("query_cases", [])],
            gold_communities={str(k): str(v) for k, v in (data.get("gold_communities") or {}).items()},
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class PipelinePrediction:
    """Predictions emitted by any extraction/graph/retrieval pipeline."""

    entities: list[tuple[str, str | None]] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)
    kg: Any | None = None
    query_results: dict[str, list[str]] = field(default_factory=dict)
    query_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    hybrid_results: dict[str, list[str]] = field(default_factory=dict)
    inferred_relations: list[tuple[str, str, str]] = field(default_factory=list)
    resolved_clusters: dict[str, str] = field(default_factory=dict)
    communities: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricResult:
    """A metric value with optional numerator/denominator context."""

    value: float
    numerator: float | None = None
    denominator: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value}
        if self.numerator is not None:
            out["numerator"] = self.numerator
        if self.denominator is not None:
            out["denominator"] = self.denominator
        return out


@dataclass
class ComponentEvaluation:
    """Metrics and diagnostics for one system component."""

    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": dict(self.metrics),
            "counts": dict(self.counts),
            "failures": list(self.failures),
        }


@dataclass
class DatasetEvaluation:
    """Evaluation result for one benchmark dataset."""

    dataset_name: str
    components: dict[str, ComponentEvaluation]
    overall: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "overall": dict(self.overall),
            "metadata": dict(self.metadata),
        }


@dataclass
class EvaluationReport:
    """A full benchmark run report."""

    run_id: str
    datasets: list[DatasetEvaluation]
    aggregate: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "datasets": [d.to_dict() for d in self.datasets],
            "aggregate": dict(self.aggregate),
            "metadata": dict(self.metadata),
        }


@dataclass
class RegressionComparison:
    """Metric-by-metric comparison between two reports."""

    baseline_run_id: str
    candidate_run_id: str
    deltas: dict[str, dict[str, float]]
    regressions: list[dict[str, Any]]
    improvements: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "deltas": self.deltas,
            "regressions": self.regressions,
            "improvements": self.improvements,
        }


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())
