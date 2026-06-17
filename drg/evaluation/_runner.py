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
    retrieval_metrics,
)
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
        retrieval_k: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id or _default_run_id()
        self.retrieval_k = retrieval_k
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
        for dataset in datasets:
            if runner is not None:
                prediction = runner(dataset)
            else:
                assert predictions is not None
                prediction = predictions.get(dataset.name)
                if prediction is None:
                    raise KeyError(f"No prediction supplied for dataset {dataset.name!r}")
            dataset_results.append(self.evaluate_dataset(dataset, prediction))

        aggregate = _aggregate(dataset_results)
        return EvaluationReport(
            run_id=self.run_id,
            datasets=dataset_results,
            aggregate=aggregate,
            metadata=dict(self.metadata),
        )

    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        prediction: PipelinePrediction,
    ) -> DatasetEvaluation:
        components: dict[str, ComponentEvaluation] = {}

        entity_metrics = entity_prf(dataset.gold_entities, prediction.entities, typed=True)
        components["entity_extraction"] = _component(
            "entity_extraction",
            entity_metrics,
            expected=int(entity_metrics["expected"]),
            predicted=int(entity_metrics["predicted"]),
        )

        relation_metrics = relation_prf(dataset.gold_relations, prediction.relations)
        components["relationship_extraction"] = _component(
            "relationship_extraction",
            relation_metrics,
            expected=int(relation_metrics["expected"]),
            predicted=int(relation_metrics["predicted"]),
        )

        event_metrics = event_prf(dataset.gold_events, prediction.events)
        components["event_extraction"] = _component(
            "event_extraction",
            event_metrics,
            expected=int(event_metrics["expected"]),
            predicted=int(event_metrics["predicted"]),
        )

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

        retrieval_eval = self._evaluate_query_cases(
            dataset,
            prediction.query_results,
            prediction.query_scores,
        )
        components["retrieval"] = _component("retrieval", retrieval_eval)

        hybrid_eval = self._evaluate_query_cases(
            dataset,
            prediction.hybrid_results,
            prediction.query_scores,
        )
        components["hybrid_retrieval"] = _component("hybrid_retrieval", hybrid_eval)

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
            "retrieval_ndcg": retrieval_eval.get("ndcg", 0.0),
            "hybrid_ndcg": hybrid_eval.get("ndcg", 0.0),
        }
        overall["overall_score"] = _mean(
            [
                overall["extraction_f1"],
                overall["graph_quality"],
                overall["reasoning_f1"] if dataset.gold_inferred_relations else None,
                overall["retrieval_ndcg"] if dataset.query_cases else None,
                overall["hybrid_ndcg"] if dataset.query_cases else None,
            ]
        )

        _add_failures(components)
        return DatasetEvaluation(
            dataset_name=dataset.name,
            components=components,
            overall=overall,
            metadata=dict(dataset.metadata),
        )

    def _evaluate_query_cases(
        self,
        dataset: BenchmarkDataset,
        results_by_query: dict[str, list[str]],
        scores_by_query: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        if not dataset.query_cases:
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
                "hits_at_k": 0.0,
            }
        per_case: list[dict[str, float]] = []
        for case in dataset.query_cases:
            relevant = set(case.relevant_entities) | set(case.expected_answer_entities)
            relevant |= {r.source for r in case.relevant_relations}
            relevant |= {r.target for r in case.relevant_relations}
            relevant |= set(case.relevant_chunks)
            ranked = results_by_query.get(case.query, [])
            scores = scores_by_query.get(case.query, {})
            per_case.append(
                retrieval_metrics(
                    ranked,
                    relevant,
                    k=self.retrieval_k,
                    scores=scores,
                )
            )
        return {
            key: _mean([m[key] for m in per_case])
            for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg", "hits_at_k")
        }


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
