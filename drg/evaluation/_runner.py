"""BenchmarkRunner — evaluates pipeline predictions against gold-standard data.

Metrics computed
----------------
* Entity extraction: precision, recall, F1 (matched by normalised name + type)
* Relation extraction: precision, recall, F1 (matched by source + type + target)
* Event extraction: precision, recall, F1 (matched by type + trigger)
* Post-reasoning relations: same as relation metrics but for inferred edges
* Weighted F1: 0.6 × entity_F1 + 0.4 × relation_F1
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from ._types import (
    BenchmarkDataset,
    EvaluationReport,
    MetricResult,
    PipelinePrediction,
)

__all__ = ["BenchmarkRunner"]


def _normalise(s: str) -> str:
    return s.strip().lower()


def _entity_key(e: Any) -> tuple[str, str]:
    if isinstance(e, (list, tuple)):
        name = str(e[0]) if len(e) > 0 else ""
        etype = str(e[1]) if len(e) > 1 else ""
        return (_normalise(name), _normalise(etype))
    return (_normalise(e.get("name", e.get("id", ""))), _normalise(e.get("type", "")))


def _relation_key(r: Any) -> tuple[str, str, str]:
    if isinstance(r, (list, tuple)):
        src = str(r[0]) if len(r) > 0 else ""
        rel = str(r[1]) if len(r) > 1 else ""
        tgt = str(r[2]) if len(r) > 2 else ""
        return (_normalise(src), _normalise(rel), _normalise(tgt))
    return (
        _normalise(r.get("source", r.get("subject", ""))),
        _normalise(r.get("type", r.get("relation", r.get("relationship_type", "")))),
        _normalise(r.get("target", r.get("object", ""))),
    )


def _event_key(ev: dict[str, Any]) -> tuple[str, str]:
    return (
        _normalise(ev.get("type", ev.get("event_type", ""))),
        _normalise(ev.get("trigger", ev.get("mention", ""))),
    )


def _prf(tp: int, fp: int, fn: int) -> MetricResult:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return MetricResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def _score_sets(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
) -> MetricResult:
    gold_keys = {key_fn(g) for g in gold}
    pred_keys = {key_fn(p) for p in pred}
    tp = len(gold_keys & pred_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)
    return _prf(tp, fp, fn)


def _average_metrics(results: list[MetricResult]) -> MetricResult:
    if not results:
        return MetricResult(0.0, 0.0, 0.0)
    n = len(results)
    return MetricResult(
        precision=sum(r.precision for r in results) / n,
        recall=sum(r.recall for r in results) / n,
        f1=sum(r.f1 for r in results) / n,
        true_positives=sum(r.true_positives for r in results),
        false_positives=sum(r.false_positives for r in results),
        false_negatives=sum(r.false_negatives for r in results),
    )


class BenchmarkRunner:
    """Evaluate a pipeline against one or more :class:`~drg.evaluation.BenchmarkDataset`.

    Parameters
    ----------
    run_id:
        Unique identifier for this evaluation run.  Auto-generated when not
        provided.
    measure_performance:
        When ``True``, record wall-clock time per dataset.
    metadata:
        Arbitrary metadata to embed in the :class:`~drg.evaluation.EvaluationReport`.
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        measure_performance: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.measure_performance = measure_performance
        self.metadata: dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        datasets: list[BenchmarkDataset],
        *,
        predictions: list[PipelinePrediction] | None = None,
        runner: Callable[[BenchmarkDataset], PipelinePrediction] | None = None,
    ) -> EvaluationReport:
        """Run evaluation and return an :class:`~drg.evaluation.EvaluationReport`.

        Exactly one of *predictions* or *runner* must be provided.

        Parameters
        ----------
        datasets:
            The benchmark datasets to evaluate against.
        predictions:
            Pre-computed predictions (one per dataset, in order).
        runner:
            Callable that takes a :class:`~drg.evaluation.BenchmarkDataset` and
            returns a :class:`~drg.evaluation.PipelinePrediction`.
        """
        if predictions is not None and runner is not None:
            raise ValueError("Provide either 'predictions' or 'runner', not both.")
        if predictions is None and runner is None:
            raise ValueError("One of 'predictions' or 'runner' must be provided.")

        if predictions is not None:
            if len(predictions) != len(datasets):
                raise ValueError(
                    f"predictions length ({len(predictions)}) != "
                    f"datasets length ({len(datasets)})"
                )
            preds = predictions
        else:
            assert runner is not None
            preds = []
            for ds in datasets:
                preds.append(runner(ds))

        entity_metrics_list: list[MetricResult] = []
        relation_metrics_list: list[MetricResult] = []
        event_metrics_list: list[MetricResult] = []
        reasoning_metrics_list: list[MetricResult] = []
        per_dataset: list[dict[str, Any]] = []
        perf_times: list[float] = []

        for ds, pred in zip(datasets, preds):
            t0 = time.monotonic()

            ent_m = _score_sets(ds.gold_entities, pred.entities, _entity_key)
            rel_m = _score_sets(ds.gold_relations, pred.relations, _relation_key)
            evt_m = _score_sets(ds.gold_events, pred.events, _event_key)
            rsn_m = _score_sets(
                ds.gold_inferred_relations,
                pred.inferred_relations,
                _relation_key,
            )

            elapsed = time.monotonic() - t0
            if self.measure_performance:
                perf_times.append(elapsed)

            entity_metrics_list.append(ent_m)
            relation_metrics_list.append(rel_m)
            event_metrics_list.append(evt_m)
            reasoning_metrics_list.append(rsn_m)

            per_dataset.append(
                {
                    "dataset": ds.name,
                    "entity_f1": round(ent_m.f1, 4),
                    "relation_f1": round(rel_m.f1, 4),
                    "event_f1": round(evt_m.f1, 4),
                    "reasoning_f1": round(rsn_m.f1, 4),
                    **({"elapsed_seconds": round(elapsed, 3)} if self.measure_performance else {}),
                }
            )

        avg_entity = _average_metrics(entity_metrics_list)
        avg_relation = _average_metrics(relation_metrics_list)
        avg_event = _average_metrics(event_metrics_list)
        avg_reasoning = _average_metrics(reasoning_metrics_list)
        weighted_f1 = 0.6 * avg_entity.f1 + 0.4 * avg_relation.f1

        performance: dict[str, Any] = {}
        if self.measure_performance and perf_times:
            performance = {
                "mean_seconds": round(sum(perf_times) / len(perf_times), 3),
                "total_seconds": round(sum(perf_times), 3),
                "dataset_count": len(perf_times),
            }

        return EvaluationReport(
            run_id=self.run_id,
            entity_metrics=avg_entity,
            relation_metrics=avg_relation,
            event_metrics=avg_event,
            reasoning_metrics=avg_reasoning,
            weighted_f1=round(weighted_f1, 4),
            dataset_count=len(datasets),
            metadata=self.metadata,
            per_dataset=per_dataset,
            performance=performance,
        )
