"""Benchmark runners and regression comparison utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from ._metrics import (
    community_pair_quality,
    entity_prf,
    entity_resolution_pair_metrics,
    event_prf,
    graph_metrics,
    relation_prf,
)
from ._performance import measure_call, summarize_performance
from ._types import (
    BenchmarkDataset,
    ComponentEvaluation,
    DatasetEvaluation,
    EvaluationReport,
    PipelinePrediction,
    RegressionComparison,
)

PredictionRunner = Callable[[BenchmarkDataset], PipelinePrediction]

__all__ = [
    "BenchmarkRunner",
    "compare_reports",
]


class BenchmarkRunner:
    """Evaluate predictions against one or more benchmark datasets."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        measure_performance: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id or _default_run_id()
        self.measure_performance = measure_performance
        self.metadata = dict(metadata or {})

    def evaluate(
        self,
        datasets: Iterable[BenchmarkDataset],
        predictions: dict[str, PipelinePrediction] | None = None,
        runner: PredictionRunner | None = None,
    ) -> EvaluationReport:
        """Evaluate all datasets.

        Pass either ``predictions`` keyed by dataset name or a ``runner``
        callable that executes a pipeline for each dataset.
        """
        if predictions is None and runner is None:
            raise ValueError("Provide predictions or runner")

        dataset_results: list[DatasetEvaluation] = []
        performance_results: list[dict[str, Any]] = []
        for dataset in datasets:
            performance: dict[str, Any] | None = None
            if runner is not None:
                if self.measure_performance:
                    prediction, performance = measure_call(lambda dataset=dataset: runner(dataset))
                else:
                    prediction = runner(dataset)
            else:
                assert predictions is not None
                prediction = predictions.get(dataset.name)
                if prediction is None:
                    raise KeyError(f"No prediction supplied for dataset {dataset.name!r}")
            if performance is not None:
                _add_prediction_throughput(performance, dataset, prediction)
                performance_results.append(performance)
            dataset_results.append(
                self.evaluate_dataset(dataset, prediction, performance=performance)
            )

        aggregate = _aggregate(dataset_results)
        metadata = dict(self.metadata)
        if performance_results:
            metadata["performance"] = summarize_performance(performance_results)
        return EvaluationReport(
            run_id=self.run_id,
            datasets=dataset_results,
            aggregate=aggregate,
            metadata=metadata,
        )

    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        prediction: PipelinePrediction,
        *,
        performance: dict[str, Any] | None = None,
    ) -> DatasetEvaluation:
        components: dict[str, ComponentEvaluation] = {}

        entity_metrics = entity_prf(dataset.gold_entities, prediction.entities, typed=True)
        components["entity_extraction"] = _component(
            "entity_extraction",
            entity_metrics,
            expected=int(entity_metrics["expected"]),
            predicted=int(entity_metrics["predicted"]),
        )
        _add_entity_diagnostics(components["entity_extraction"], dataset, prediction)

        relation_metrics = relation_prf(dataset.gold_relations, prediction.relations)
        components["relationship_extraction"] = _component(
            "relationship_extraction",
            relation_metrics,
            expected=int(relation_metrics["expected"]),
            predicted=int(relation_metrics["predicted"]),
        )
        _add_relation_diagnostics(components["relationship_extraction"], dataset, prediction)

        event_metrics = event_prf(dataset.gold_events, prediction.events)
        components["event_extraction"] = _component(
            "event_extraction",
            event_metrics,
            expected=int(event_metrics["expected"]),
            predicted=int(event_metrics["predicted"]),
        )
        _add_event_diagnostics(components["event_extraction"], dataset, prediction)

        graph_eval = graph_metrics(
            gold_entities=dataset.gold_entities,
            gold_relations=dataset.gold_relations,
            predicted_entities=prediction.entities,
            predicted_relations=prediction.relations,
        )
        components["graph_construction"] = _component("graph_construction", graph_eval)

        inference_metrics = relation_prf(
            dataset.gold_inferred_relations,
            prediction.inferred_relations,
        )
        components["query_reasoning"] = _component(
            "query_reasoning",
            {
                "inference_precision": inference_metrics["precision"],
                "inference_recall": inference_metrics["recall"],
                "inference_f1": inference_metrics["f1"],
            },
            expected=int(inference_metrics["expected"]),
            predicted=int(inference_metrics["predicted"]),
        )

        if dataset.gold_communities or prediction.resolved_clusters:
            er_metrics = entity_resolution_pair_metrics(
                dataset.gold_communities,
                prediction.resolved_clusters,
            )
        else:
            er_metrics = {
                "pairwise_precision": 0.0,
                "pairwise_recall": 0.0,
                "pairwise_f1": 0.0,
            }
        components["entity_resolution"] = _component("entity_resolution", er_metrics)

        if dataset.gold_communities or prediction.communities:
            comm_metrics = community_pair_quality(dataset.gold_communities, prediction.communities)
        else:
            comm_metrics = {
                "community_pair_precision": 0.0,
                "community_pair_recall": 0.0,
                "community_pair_f1": 0.0,
            }
        components["community_quality"] = _component("community_quality", comm_metrics)

        if performance:
            components["runtime_performance"] = _component(
                "runtime_performance",
                {k: v for k, v in performance.items() if isinstance(v, int | float)},
            )

        overall = {
            "extraction_f1": _mean(
                [
                    entity_metrics["f1"],
                    relation_metrics["f1"],
                    event_metrics["f1"] if dataset.gold_events else None,
                ]
            ),
            "graph_quality": _mean(
                [
                    graph_eval["entity_coverage"],
                    graph_eval["relation_coverage"],
                    1.0 - graph_eval["orphan_node_rate"],
                ]
            ),
            "reasoning_f1": inference_metrics["f1"],
        }
        overall["overall_score"] = _mean(
            [
                overall["extraction_f1"],
                overall["graph_quality"],
                overall["reasoning_f1"] if dataset.gold_inferred_relations else None,
            ]
        )

        _add_failures(components)
        metadata = dict(dataset.metadata)
        if prediction.metadata:
            metadata["prediction"] = dict(prediction.metadata)
        if performance:
            metadata["performance"] = dict(performance)
        return DatasetEvaluation(
            dataset_name=dataset.name,
            components=components,
            overall=overall,
            metadata=metadata,
        )

def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    regression_threshold: float = 0.01,
) -> RegressionComparison:
    """Compare aggregate and per-dataset overall metrics."""
    deltas: dict[str, dict[str, float]] = {"aggregate": {}}
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    for metric, before in baseline.aggregate.items():
        after = candidate.aggregate.get(metric, 0.0)
        delta = after - before
        deltas["aggregate"][metric] = delta
        _classify_delta(
            regressions,
            improvements,
            scope="aggregate",
            metric=metric,
            before=before,
            after=after,
            delta=delta,
            threshold=regression_threshold,
        )

    baseline_by_name = {d.dataset_name: d for d in baseline.datasets}
    for cand_ds in candidate.datasets:
        base_ds = baseline_by_name.get(cand_ds.dataset_name)
        if base_ds is None:
            continue
        scope = f"dataset:{cand_ds.dataset_name}"
        deltas[scope] = {}
        for metric, before in base_ds.overall.items():
            after = cand_ds.overall.get(metric, 0.0)
            delta = after - before
            deltas[scope][metric] = delta
            _classify_delta(
                regressions,
                improvements,
                scope=scope,
                metric=metric,
                before=before,
                after=after,
                delta=delta,
                threshold=regression_threshold,
            )

    return RegressionComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        deltas=deltas,
        regressions=regressions,
        improvements=improvements,
    )


def _component(
    name: str,
    metrics: dict[str, float],
    *,
    expected: int | None = None,
    predicted: int | None = None,
) -> ComponentEvaluation:
    counts = {}
    if expected is not None:
        counts["expected"] = expected
    if predicted is not None:
        counts["predicted"] = predicted
    return ComponentEvaluation(
        name=name,
        metrics={k: float(v) for k, v in metrics.items()},
        counts=counts,
    )


def _add_prediction_throughput(
    performance: dict[str, Any],
    dataset: BenchmarkDataset,
    prediction: PipelinePrediction,
) -> None:
    elapsed = float(performance.get("wall_time_seconds", 0.0))
    if elapsed <= 0:
        return
    chunk_count = int(dataset.metadata.get("chunk_count") or 0)
    if chunk_count:
        performance["chunks_per_second"] = chunk_count / elapsed
    performance["characters_per_second"] = len(dataset.text) / elapsed
    performance["entities_per_second"] = len(prediction.entities) / elapsed
    performance["relations_per_second"] = len(prediction.relations) / elapsed
    performance["predicted_entities"] = float(len(prediction.entities))
    performance["predicted_relations"] = float(len(prediction.relations))


def _add_entity_diagnostics(
    component: ComponentEvaluation,
    dataset: BenchmarkDataset,
    prediction: PipelinePrediction,
) -> None:
    expected = {(_norm(e.name), _norm(e.type)): e for e in dataset.gold_entities}
    predicted = {(_norm(name), _norm(etype)): (name, etype) for name, etype in prediction.entities}
    expected_names = {_norm(e.name): e for e in dataset.gold_entities}
    predicted_names = {_norm(name): etype for name, etype in prediction.entities}

    for key, entity in sorted(expected.items()):
        if key not in predicted:
            name_key = _norm(entity.name)
            kind = "wrong_type" if name_key in predicted_names else "missing_entity"
            component.failures.append(
                {
                    "type": kind,
                    "expected": entity.to_dict(),
                    "predicted_type": predicted_names.get(name_key),
                    "severity": "error",
                    "description": f"Entity {entity.name!r} was not matched with the expected type.",
                }
            )
    for key, (name, etype) in sorted(predicted.items()):
        if key not in expected and _norm(name) not in expected_names:
            component.failures.append(
                {
                    "type": "hallucinated_entity",
                    "predicted": {"name": name, "type": etype},
                    "severity": "warning",
                    "description": f"Predicted entity {name!r} is not in the gold set.",
                }
            )


def _add_relation_diagnostics(
    component: ComponentEvaluation,
    dataset: BenchmarkDataset,
    prediction: PipelinePrediction,
) -> None:
    expected = {r.key(): r for r in dataset.gold_relations}
    predicted = {(_norm(s), _norm(r), _norm(t)): (s, r, t) for s, r, t in prediction.relations}
    predicted_pairs = {(_norm(s), _norm(t)) for s, _r, t in prediction.relations}

    for key, relation in sorted(expected.items()):
        if key not in predicted:
            pair = (_norm(relation.source), _norm(relation.target))
            kind = "wrong_relation_type" if pair in predicted_pairs else "missing_relation"
            component.failures.append(
                {
                    "type": kind,
                    "expected": relation.to_dict(),
                    "severity": "error",
                    "description": (
                        f"Relation {relation.source!r} -[{relation.relationship_type}]-> "
                        f"{relation.target!r} was not matched."
                    ),
                }
            )
    for key, (source, rel_type, target) in sorted(predicted.items()):
        if key not in expected:
            component.failures.append(
                {
                    "type": "hallucinated_edge",
                    "predicted": {
                        "source": source,
                        "relationship_type": rel_type,
                        "target": target,
                    },
                    "severity": "warning",
                    "description": (
                        f"Predicted edge {source!r} -[{rel_type}]-> {target!r} "
                        "is not in the gold set."
                    ),
                }
            )


def _add_event_diagnostics(
    component: ComponentEvaluation,
    dataset: BenchmarkDataset,
    prediction: PipelinePrediction,
) -> None:
    expected_count = len(dataset.gold_events)
    predicted_count = len(prediction.events)
    if expected_count and not predicted_count:
        component.failures.append(
            {
                "type": "missing_event",
                "expected": expected_count,
                "severity": "error",
                "description": "Gold events exist, but the pipeline returned no events.",
            }
        )
    elif predicted_count > expected_count:
        component.failures.append(
            {
                "type": "extra_event",
                "expected": expected_count,
                "predicted": predicted_count,
                "severity": "warning",
                "description": "The pipeline returned more events than the gold annotations.",
            }
        )


def _aggregate(results: list[DatasetEvaluation]) -> dict[str, float]:
    keys = sorted({k for result in results for k in result.overall})
    return {key: _mean([r.overall.get(key) for r in results]) for key in keys}


def _mean(values: Iterable[float | None]) -> float:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _add_failures(components: dict[str, ComponentEvaluation]) -> None:
    for component in components.values():
        for metric, value in component.metrics.items():
            if metric.endswith("f1") and value < 0.5:
                component.failures.append(
                    {
                        "metric": metric,
                        "value": value,
                        "severity": "warning",
                        "description": f"{component.name}.{metric} is below 0.50",
                    }
                )


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _classify_delta(
    regressions: list[dict[str, Any]],
    improvements: list[dict[str, Any]],
    *,
    scope: str,
    metric: str,
    before: float,
    after: float,
    delta: float,
    threshold: float,
) -> None:
    item = {
        "scope": scope,
        "metric": metric,
        "before": before,
        "after": after,
        "delta": delta,
    }
    if delta <= -threshold:
        regressions.append(item)
    elif delta >= threshold:
        improvements.append(item)


def _default_run_id() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
