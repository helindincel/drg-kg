"""Evaluation framework for extraction, graph, reasoning, and retrieval quality."""

from __future__ import annotations

from ._metrics import (
    community_pair_quality,
    entity_prf,
    entity_resolution_pair_metrics,
    event_prf,
    graph_metrics,
    precision_recall_f1,
    relation_prf,
    retrieval_metrics,
)
from ._reporting import (
    load_benchmark_dataset,
    load_benchmark_datasets,
    render_markdown_report,
    render_regression_markdown,
    save_json_report,
    save_markdown_report,
)
from ._runner import BenchmarkRunner, compare_reports
from ._types import (
    BenchmarkDataset,
    ComponentEvaluation,
    DatasetEvaluation,
    EvaluationReport,
    GoldEntity,
    GoldEvent,
    GoldRelation,
    PipelinePrediction,
    QueryCase,
    RegressionComparison,
)

__all__ = [
    "BenchmarkDataset",
    "BenchmarkRunner",
    "ComponentEvaluation",
    "DatasetEvaluation",
    "EvaluationReport",
    "GoldEntity",
    "GoldEvent",
    "GoldRelation",
    "PipelinePrediction",
    "QueryCase",
    "RegressionComparison",
    "community_pair_quality",
    "compare_reports",
    "entity_prf",
    "entity_resolution_pair_metrics",
    "event_prf",
    "graph_metrics",
    "load_benchmark_dataset",
    "load_benchmark_datasets",
    "precision_recall_f1",
    "relation_prf",
    "render_markdown_report",
    "render_regression_markdown",
    "retrieval_metrics",
    "save_json_report",
    "save_markdown_report",
]
