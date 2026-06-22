"""Metric implementations for extraction, graph, and reasoning."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from ._types import GoldEntity, GoldEvent, GoldRelation

__all__ = [
    "community_pair_quality",
    "entity_prf",
    "entity_resolution_pair_metrics",
    "event_prf",
    "graph_metrics",
    "precision_recall_f1",
    "relation_prf",
]


def precision_recall_f1(expected: set[Any], predicted: set[Any]) -> dict[str, float]:
    """Precision/recall/F1 for exact-match sets."""
    correct = expected & predicted
    precision = len(correct) / len(predicted) if predicted else 0.0
    recall = len(correct) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": float(len(correct)),
        "expected": float(len(expected)),
        "predicted": float(len(predicted)),
    }


def entity_prf(
    expected: list[GoldEntity],
    predicted: list[tuple[str, str | None]],
    *,
    typed: bool = True,
) -> dict[str, float]:
    exp = {e.key(typed=typed) for e in expected}
    pred = {(norm(name), norm(etype)) if typed else norm(name) for name, etype in predicted}
    return precision_recall_f1(exp, pred)


def relation_prf(
    expected: list[GoldRelation],
    predicted: list[tuple[str, str, str]],
) -> dict[str, float]:
    exp = {r.key() for r in expected}
    pred = {(norm(s), norm(r), norm(t)) for s, r, t in predicted}
    return precision_recall_f1(exp, pred)


def event_prf(expected: list[GoldEvent], predicted: list[Any]) -> dict[str, float]:
    """Event precision/recall/F1 using canonical event payload keys."""
    exp = {e.key() for e in expected}
    pred = {_event_key(e) for e in predicted}
    return precision_recall_f1(exp, pred)


def graph_metrics(
    *,
    gold_entities: list[GoldEntity],
    gold_relations: list[GoldRelation],
    predicted_entities: list[tuple[str, str | None]],
    predicted_relations: list[tuple[str, str, str]],
) -> dict[str, float]:
    """Graph construction coverage and shape metrics."""
    gold_node_keys = {e.key(typed=False) for e in gold_entities}
    pred_node_keys = {norm(name) for name, _ in predicted_entities}
    gold_edge_keys = {r.key() for r in gold_relations}
    pred_edge_keys = {(norm(s), norm(r), norm(t)) for s, r, t in predicted_relations}

    node_count = len(pred_node_keys)
    edge_count = len(pred_edge_keys)
    possible_directed = node_count * max(0, node_count - 1)
    density = edge_count / possible_directed if possible_directed else 0.0

    endpoints = {s for s, _r, _t in pred_edge_keys} | {t for _s, _r, t in pred_edge_keys}
    orphan_nodes = pred_node_keys - endpoints if edge_count else pred_node_keys

    return {
        "entity_coverage": (
            len(gold_node_keys & pred_node_keys) / len(gold_node_keys) if gold_node_keys else 0.0
        ),
        "relation_coverage": (
            len(gold_edge_keys & pred_edge_keys) / len(gold_edge_keys) if gold_edge_keys else 0.0
        ),
        "graph_density": density,
        "orphan_node_rate": len(orphan_nodes) / node_count if node_count else 0.0,
        "node_count": float(node_count),
        "edge_count": float(edge_count),
    }


def entity_resolution_pair_metrics(
    gold_clusters: dict[str, str],
    predicted_clusters: dict[str, str],
) -> dict[str, float]:
    """Pairwise precision/recall/F1 for entity-resolution clusters."""
    gold_pairs = _cluster_pairs(gold_clusters)
    pred_pairs = _cluster_pairs(predicted_clusters)
    out = precision_recall_f1(gold_pairs, pred_pairs)
    return {
        "pairwise_precision": out["precision"],
        "pairwise_recall": out["recall"],
        "pairwise_f1": out["f1"],
    }


def community_pair_quality(
    gold_communities: dict[str, str],
    predicted_communities: dict[str, str],
) -> dict[str, float]:
    """Community quality via pairwise co-membership."""
    metrics = entity_resolution_pair_metrics(gold_communities, predicted_communities)
    return {
        "community_pair_precision": metrics["pairwise_precision"],
        "community_pair_recall": metrics["pairwise_recall"],
        "community_pair_f1": metrics["pairwise_f1"],
    }


def _cluster_pairs(mapping: dict[str, str]) -> set[tuple[str, str]]:
    by_cluster: dict[str, list[str]] = {}
    for entity, cluster in mapping.items():
        by_cluster.setdefault(norm(cluster), []).append(norm(entity))
    pairs: set[tuple[str, str]] = set()
    for members in by_cluster.values():
        for a, b in combinations(sorted(set(members)), 2):
            pairs.add((a, b))
    return pairs


def _event_key(event: Any) -> tuple[Any, ...]:
    if isinstance(event, GoldEvent):
        return event.key()
    if isinstance(event, dict):
        return GoldEvent.from_any(event).key()
    event_type = getattr(event, "event_type", "") or ""
    participants_raw = getattr(event, "participants", {}) or {}
    participants = {
        str(role): [str(v) for v in values] for role, values in participants_raw.items()
    }
    ts = getattr(event, "timestamp", None)
    timestamp = None
    if ts is not None:
        timestamp = getattr(ts, "start", None) or getattr(ts, "value", None) or str(ts)
    location = getattr(event, "location", None)
    return GoldEvent(
        event_type=str(event_type),
        participants=participants,
        timestamp=str(timestamp) if timestamp is not None else None,
        location=str(location) if location is not None else None,
    ).key()


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())
