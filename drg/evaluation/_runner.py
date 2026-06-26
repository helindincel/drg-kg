"""BenchmarkRunner — evaluates pipeline predictions against gold-standard data.

Metrics computed
----------------
* Entity extraction: precision, recall, F1 (alias/canonical-aware name + type)
* Relation extraction: precision, recall, F1 (alias-aware source + type + target)
* Event extraction: precision, recall, F1 (type + roles + temporal fields)
* Post-reasoning relations: same as relation metrics but for inferred edges
* Evidence quality: source-span/snippet precision, recall, F1 when gold evidence exists
* Confidence calibration: ECE and Brier score when predictions carry confidence
* Weighted F1: 0.45 × entity_F1 + 0.35 × relation_F1 + 0.20 × event_F1
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from collections.abc import Callable
from typing import Any

from ._types import (
    BenchmarkDataset,
    CalibrationReport,
    EvaluationReport,
    MetricResult,
    PipelinePrediction,
)

__all__ = ["BenchmarkRunner"]


def _normalise(s: str) -> str:
    return s.strip().lower()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return {str(i): item for i, item in enumerate(value)}
    return {}


def _first_str(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            return value
    return ""


def _entity_name(e: Any) -> str:
    if isinstance(e, (list, tuple)):
        return str(e[0]) if len(e) > 0 else ""
    data = _as_dict(e)
    return _first_str(data, "canonical_id", "canonical_name", "name", "id", "text", "mention")


def _entity_type(e: Any) -> str:
    if isinstance(e, (list, tuple)):
        return str(e[1]) if len(e) > 1 else ""
    return _first_str(_as_dict(e), "type", "entity_type", "label")


def _aliases(e: Any) -> set[str]:
    data = _as_dict(e)
    names = {_entity_name(e)}
    for key in ("alias", "aliases", "mentions", "surface_forms"):
        raw = data.get(key)
        if isinstance(raw, str):
            names.add(raw)
        elif isinstance(raw, list):
            names.update(str(item) for item in raw if item)
    return {_normalise(name) for name in names if name}


def _span(e: Any) -> tuple[int, int] | None:
    data = _as_dict(e)
    raw = data.get("span", data.get("source_span"))
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and all(isinstance(v, int) for v in raw):
        return (raw[0], raw[1])
    start = data.get("start")
    end = data.get("end")
    if isinstance(start, int) and isinstance(end, int):
        return (start, end)
    return None


def _span_iou(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    if a is None or b is None:
        return 0.0
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    intersection = max(0, right - left)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def _entity_key(e: Any) -> tuple[str, str]:
    return (_normalise(_entity_name(e)), _normalise(_entity_type(e)))


def _relation_key(r: Any, aliases: dict[str, str] | None = None) -> tuple[str, str, str]:
    if isinstance(r, (list, tuple)):
        src = str(r[0]) if len(r) > 0 else ""
        rel = str(r[1]) if len(r) > 1 else ""
        tgt = str(r[2]) if len(r) > 2 else ""
    else:
        data = _as_dict(r)
        src = _first_str(data, "source", "subject", "src")
        rel = _first_str(data, "type", "relation", "relationship_type", "predicate")
        tgt = _first_str(data, "target", "object", "dst")
    src_key = _normalise(src)
    tgt_key = _normalise(tgt)
    aliases = aliases or {}
    return (aliases.get(src_key, src_key), _normalise(rel), aliases.get(tgt_key, tgt_key))


def _event_participants(
    ev: dict[str, Any], aliases: dict[str, str] | None = None
) -> tuple[Any, ...]:
    aliases = aliases or {}
    raw = ev.get("participants", ev.get("arguments", ev.get("roles", {})))
    if not isinstance(raw, dict):
        return ()
    parts: list[tuple[str, tuple[str, ...]]] = []
    for role, values in raw.items():
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        normalized = tuple(
            sorted(aliases.get(_normalise(str(value)), _normalise(str(value))) for value in values)
        )
        parts.append((_normalise(str(role)), normalized))
    return tuple(sorted(parts))


def _event_key(ev: dict[str, Any], aliases: dict[str, str] | None = None) -> tuple[Any, ...]:
    return (
        _normalise(ev.get("type", ev.get("event_type", ""))),
        _normalise(ev.get("trigger", ev.get("mention", ""))),
        _normalise(str(ev.get("timestamp", ev.get("time", ev.get("date", ""))) or "")),
        _event_participants(ev, aliases),
    )


def _prf(tp: int, fp: int, fn: int) -> MetricResult:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return MetricResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def _score_sets(
    gold: list[Any],
    pred: list[Any],
    key_fn: Callable[[Any], Any],
    pred_key_fn: Callable[[Any], Any] | None = None,
) -> MetricResult:
    gold_keys = Counter(key_fn(g) for g in gold)
    pred_keys = Counter((pred_key_fn or key_fn)(p) for p in pred)
    tp = sum(min(gold_keys[key], pred_keys[key]) for key in gold_keys)
    fp = sum((pred_keys - gold_keys).values())
    fn = sum((gold_keys - pred_keys).values())
    metric = _prf(tp, fp, fn)
    metric.details.update(
        {
            "false_positive_keys": sorted(map(str, (pred_keys - gold_keys).elements())),
            "false_negative_keys": sorted(map(str, (gold_keys - pred_keys).elements())),
        }
    )
    return metric


def _build_alias_lookup(*entity_sets: list[Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for entities in entity_sets:
        for entity in entities:
            canonical = _normalise(_entity_name(entity))
            if not canonical:
                continue
            for alias in _aliases(entity):
                lookup.setdefault(alias, canonical)
            lookup.setdefault(canonical, canonical)
    return lookup


def _score_entities(
    gold: list[Any], pred: list[Any]
) -> tuple[MetricResult, dict[str, str], set[str]]:
    unmatched_pred = set(range(len(pred)))
    matches: list[dict[str, Any]] = []
    matched_pred_aliases: set[str] = set()

    for gold_item in gold:
        gold_type = _normalise(_entity_type(gold_item))
        gold_aliases = _aliases(gold_item)
        best_idx: int | None = None
        best_reason = ""
        best_score = 0.0

        for idx in list(unmatched_pred):
            pred_item = pred[idx]
            pred_type = _normalise(_entity_type(pred_item))
            if gold_type and pred_type and gold_type != pred_type:
                continue
            score = 0.0
            reason = ""
            if gold_aliases & _aliases(pred_item):
                score = 1.0
                reason = "alias_or_canonical"
            else:
                span_iou = _span_iou(_span(gold_item), _span(pred_item))
                if span_iou >= 0.5:
                    score = span_iou
                    reason = "span_overlap"
            if score > best_score:
                best_idx = idx
                best_reason = reason
                best_score = score

        if best_idx is not None:
            unmatched_pred.remove(best_idx)
            matched_pred_aliases.update(_aliases(pred[best_idx]))
            matches.append(
                {
                    "gold": _entity_name(gold_item),
                    "prediction": _entity_name(pred[best_idx]),
                    "reason": best_reason,
                }
            )

    metric = _prf(len(matches), len(unmatched_pred), len(gold) - len(matches))
    metric.details.update(
        {
            "matching": "alias-aware canonical/type match with optional span overlap",
            "matches": matches,
        }
    )
    alias_lookup = _build_alias_lookup(gold, pred)
    matched_canonical_names = {alias_lookup.get(alias, alias) for alias in matched_pred_aliases}
    return metric, alias_lookup, matched_canonical_names


def _gated_relation_key(
    item: Any,
    aliases: dict[str, str],
    matched_entity_names: set[str] | None,
) -> tuple[Any, ...]:
    key = _relation_key(item, aliases)
    if matched_entity_names is None:
        return key
    if key[0] in matched_entity_names and key[2] in matched_entity_names:
        return key
    return ("__unmatched_entity__", *key)


def _event_mentions(item: Any, aliases: dict[str, str]) -> set[str]:
    data = _as_dict(item)
    raw = data.get("participants", data.get("arguments", data.get("roles", {})))
    if not isinstance(raw, dict):
        return set()
    mentions: set[str] = set()
    for values in raw.values():
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            normalized = _normalise(str(value))
            mentions.add(aliases.get(normalized, normalized))
    return mentions


def _gated_event_key(
    item: dict[str, Any],
    aliases: dict[str, str],
    matched_entity_names: set[str] | None,
) -> tuple[Any, ...]:
    key = _event_key(item, aliases)
    if matched_entity_names is None:
        return key
    mentions = _event_mentions(item, aliases)
    if not mentions or mentions <= matched_entity_names:
        return key
    return ("__unmatched_entity__", *key)


def _score_evidence(gold: list[Any], pred: list[Any]) -> MetricResult:
    if not gold and not pred:
        return MetricResult(0.0, 0.0, 0.0)

    def evidence_key(item: Any) -> tuple[str, str]:
        data = _as_dict(item)
        owner = _first_str(data, "fact_id", "relation_id", "entity_id", "event_id", "id")
        snippet = _first_str(data, "snippet", "text", "evidence")
        span = _span(data)
        span_key = f"{span[0]}:{span[1]}" if span else _normalise(snippet)
        return (_normalise(owner), span_key)

    return _score_sets(gold, pred, evidence_key)


def _confidence_from(item: Any) -> float | None:
    data = _as_dict(item)
    raw = data.get("confidence")
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    raw_score = data.get("score")
    if isinstance(raw_score, dict) and isinstance(raw_score.get("value"), (int, float)):
        return max(0.0, min(1.0, float(raw_score["value"])))
    return None


def _collect_calibration_samples(
    gold_keys: set[Any],
    predictions: list[Any],
    key_fn: Callable[[Any], Any],
) -> list[tuple[float, bool]]:
    samples: list[tuple[float, bool]] = []
    for item in predictions:
        confidence = _confidence_from(item)
        if confidence is None:
            continue
        samples.append((confidence, key_fn(item) in gold_keys))
    return samples


def _prediction_dataset_name(prediction: PipelinePrediction) -> str | None:
    raw = prediction.metadata.get("dataset_name", prediction.metadata.get("dataset"))
    return str(raw) if raw else None


def _align_predictions_to_datasets(
    datasets: list[BenchmarkDataset],
    predictions: list[PipelinePrediction],
) -> list[PipelinePrediction]:
    names = [_prediction_dataset_name(prediction) for prediction in predictions]
    if not any(names):
        return predictions
    if not all(names):
        raise ValueError("Predictions must either all include dataset metadata or none of them.")

    prediction_by_name: dict[str, PipelinePrediction] = {}
    for name, prediction in zip(names, predictions, strict=True):
        assert name is not None
        if name in prediction_by_name:
            raise ValueError(f"Duplicate prediction for dataset {name!r}")
        prediction_by_name[name] = prediction

    dataset_names = [dataset.name for dataset in datasets]
    missing = sorted(set(dataset_names) - set(prediction_by_name))
    extra = sorted(set(prediction_by_name) - set(dataset_names))
    if missing or extra:
        raise ValueError(
            "Prediction dataset names do not match benchmark datasets: "
            f"missing={missing}, extra={extra}"
        )
    return [prediction_by_name[dataset.name] for dataset in datasets]


def _calibration_report(samples: list[tuple[float, bool]], bins: int = 10) -> CalibrationReport:
    if not samples:
        return CalibrationReport()
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in samples:
        idx = min(bins - 1, int(confidence * bins))
        buckets[idx].append((confidence, correct))

    brier = sum((confidence - float(correct)) ** 2 for confidence, correct in samples) / len(
        samples
    )
    ece = 0.0
    rendered_bins: list[dict[str, Any]] = []
    for idx, bucket in enumerate(buckets):
        if not bucket:
            continue
        accuracy = sum(1 for _, correct in bucket if correct) / len(bucket)
        avg_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        weight = len(bucket) / len(samples)
        ece += weight * abs(accuracy - avg_confidence)
        rendered_bins.append(
            {
                "range": [round(idx / bins, 2), round((idx + 1) / bins, 2)],
                "count": len(bucket),
                "accuracy": round(accuracy, 4),
                "avg_confidence": round(avg_confidence, 4),
            }
        )

    return CalibrationReport(
        sample_count=len(samples),
        expected_calibration_error=ece,
        brier_score=brier,
        bins=rendered_bins,
    )


def _error_summary(metric: MetricResult, *, category: str) -> dict[str, Any]:
    errors: list[str] = []
    if metric.false_positives:
        errors.append(f"hallucinated_{category}")
    if metric.false_negatives:
        errors.append(f"missing_{category}")
    return {
        "category": category,
        "false_positives": metric.false_positives,
        "false_negatives": metric.false_negatives,
        "error_types": errors,
    }


def _weighted_extraction_f1(
    entity: MetricResult,
    relation: MetricResult,
    event: MetricResult,
) -> float:
    weighted_parts: list[tuple[float, float]] = []
    for weight, metric in ((0.45, entity), (0.35, relation), (0.20, event)):
        if metric.true_positives or metric.false_positives or metric.false_negatives:
            weighted_parts.append((weight, metric.f1))
    if not weighted_parts:
        return 0.0
    total_weight = sum(weight for weight, _ in weighted_parts)
    return sum(weight * f1 for weight, f1 in weighted_parts) / total_weight


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
        details={"averaging": "macro", "dataset_count": n},
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
                    f"predictions length ({len(predictions)}) != datasets length ({len(datasets)})"
                )
            preds = _align_predictions_to_datasets(datasets, predictions)
        else:
            assert runner is not None
            preds = []
            for ds in datasets:
                preds.append(runner(ds))

        entity_metrics_list: list[MetricResult] = []
        relation_metrics_list: list[MetricResult] = []
        event_metrics_list: list[MetricResult] = []
        reasoning_metrics_list: list[MetricResult] = []
        evidence_metrics_list: list[MetricResult] = []
        calibration_samples: list[tuple[float, bool]] = []
        per_dataset: list[dict[str, Any]] = []
        perf_times: list[float] = []

        for ds, pred in zip(datasets, preds, strict=True):
            t0 = time.monotonic()

            ent_m, aliases, matched_entity_names = _score_entities(ds.gold_entities, pred.entities)
            entity_gate = matched_entity_names if ds.gold_entities else None

            def rel_key(item: Any, alias_lookup: dict[str, str] = aliases) -> tuple[str, str, str]:
                return _relation_key(item, alias_lookup)

            def evt_key(
                item: dict[str, Any], alias_lookup: dict[str, str] = aliases
            ) -> tuple[Any, ...]:
                return _event_key(item, alias_lookup)

            def pred_rel_key(
                item: Any,
                alias_lookup: dict[str, str] = aliases,
                matched_names: set[str] | None = entity_gate,
            ) -> tuple[Any, ...]:
                return _gated_relation_key(item, alias_lookup, matched_names)

            def pred_evt_key(
                item: dict[str, Any],
                alias_lookup: dict[str, str] = aliases,
                matched_names: set[str] | None = entity_gate,
            ) -> tuple[Any, ...]:
                return _gated_event_key(item, alias_lookup, matched_names)

            rel_m = _score_sets(ds.gold_relations, pred.relations, rel_key, pred_rel_key)
            evt_m = _score_sets(ds.gold_events, pred.events, evt_key, pred_evt_key)
            rsn_m = _score_sets(
                ds.gold_inferred_relations,
                pred.inferred_relations,
                rel_key,
                pred_rel_key,
            )
            evd_m = _score_evidence(ds.gold_evidence, pred.evidence)

            gold_entity_keys = {_entity_key(item) for item in ds.gold_entities}
            gold_relation_keys = {rel_key(item) for item in ds.gold_relations}
            gold_event_keys = {evt_key(item) for item in ds.gold_events}
            calibration_samples.extend(
                _collect_calibration_samples(gold_entity_keys, pred.entities, _entity_key)
            )
            calibration_samples.extend(
                _collect_calibration_samples(gold_relation_keys, pred.relations, pred_rel_key)
            )
            calibration_samples.extend(
                _collect_calibration_samples(gold_event_keys, pred.events, pred_evt_key)
            )

            elapsed = time.monotonic() - t0
            if self.measure_performance:
                perf_times.append(elapsed)

            entity_metrics_list.append(ent_m)
            relation_metrics_list.append(rel_m)
            event_metrics_list.append(evt_m)
            reasoning_metrics_list.append(rsn_m)
            evidence_metrics_list.append(evd_m)

            per_dataset.append(
                {
                    "dataset": ds.name,
                    "domain": ds.metadata.get("domain"),
                    "difficulty": ds.metadata.get("difficulty"),
                    "entity_f1": round(ent_m.f1, 4),
                    "relation_f1": round(rel_m.f1, 4),
                    "event_f1": round(evt_m.f1, 4),
                    "reasoning_f1": round(rsn_m.f1, 4),
                    "evidence_f1": round(evd_m.f1, 4),
                    "error_analysis": [
                        _error_summary(ent_m, category="entity"),
                        _error_summary(rel_m, category="relation"),
                        _error_summary(evt_m, category="event"),
                    ],
                    "multi_document": {
                        "document_count": len(ds.documents) or int(bool(ds.text)),
                        "source_documents": ds.metadata.get("source_documents", []),
                    },
                    **({"elapsed_seconds": round(elapsed, 3)} if self.measure_performance else {}),
                }
            )

        avg_entity = _average_metrics(entity_metrics_list)
        avg_relation = _average_metrics(relation_metrics_list)
        avg_event = _average_metrics(event_metrics_list)
        avg_reasoning = _average_metrics(reasoning_metrics_list)
        avg_evidence = _average_metrics(evidence_metrics_list)
        weighted_f1 = _weighted_extraction_f1(avg_entity, avg_relation, avg_event)

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
            evidence_metrics=avg_evidence,
            calibration=_calibration_report(calibration_samples),
            weighted_f1=round(weighted_f1, 4),
            dataset_count=len(datasets),
            metadata=self.metadata,
            per_dataset=per_dataset,
            performance=performance,
        )
