"""Evaluation framework for extraction, graph, and reasoning quality."""

from __future__ import annotations

from ._adapters import load_prediction_artifact
from ._metrics import (
    community_pair_quality,
    entity_prf,
    entity_resolution_pair_metrics,
    event_prf,
    graph_metrics,
    precision_recall_f1,
    relation_prf,
)
from ._reporting import (
    load_benchmark_dataset,
    load_benchmark_datasets,
    load_evaluation_report,
    render_markdown_report,
    render_regression_markdown,
    save_json_report,
    save_markdown_report,
)
from ._runner import BenchmarkRunner, compare_reports
from ._suite import (
    BenchmarkSuite,
    default_benchmark_suite_path,
    load_benchmark_suite,
    load_official_benchmark_suite,
)
from ._types import (
    BenchmarkDataset,
    ComponentEvaluation,
    DatasetEvaluation,
    EvaluationReport,
    GoldEntity,
    GoldEvent,
    GoldRelation,
    PipelinePrediction,
    RegressionComparison,
)

__all__ = [
    "BenchmarkDataset",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "ComponentEvaluation",
    "DatasetEvaluation",
    "EvaluationReport",
    "GoldEntity",
    "GoldEvent",
    "GoldRelation",
    "PipelinePrediction",
    "RegressionComparison",
    "community_pair_quality",
    "compare_reports",
    "default_benchmark_suite_path",
    "entity_prf",
    "entity_resolution_pair_metrics",
    "event_prf",
    "graph_metrics",
    "load_benchmark_dataset",
    "load_benchmark_datasets",
    "load_benchmark_suite",
    "load_evaluation_report",
    "load_official_benchmark_suite",
    "load_prediction_artifact",
    "precision_recall_f1",
    "relation_prf",
    "render_markdown_report",
    "render_regression_markdown",
    "save_json_report",
    "save_markdown_report",
]
