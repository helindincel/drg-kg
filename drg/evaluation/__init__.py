"""Evaluation framework for DRG knowledge graph pipelines.

This package provides tools to benchmark and compare pipeline outputs against
gold-standard annotated datasets.

Quick start::

    from drg.evaluation import BenchmarkRunner, PipelinePrediction, load_official_benchmark_suite

    suite = load_official_benchmark_suite()
    runner = BenchmarkRunner(run_id="my-run-001", measure_performance=True)

    def my_pipeline(ds):
        entities, relations = extract_typed(ds.text, schema)
        return PipelinePrediction(entities=entities, relations=relations)

    report = runner.evaluate(suite.datasets, runner=my_pipeline)
    print(f"Weighted F1: {report.weighted_f1:.4f}")

CLI::

    drg eval run my_datasets.json --output report.json
    drg eval compare --baseline baseline.json --candidate new.json
    drg eval list
"""

from ._compare import compare_reports, render_markdown_report, render_regression_markdown
from ._io import (
    load_benchmark_datasets,
    load_benchmark_suite,
    load_evaluation_report,
    load_official_benchmark_suite,
    load_prediction_artifact,
    save_json_report,
    save_markdown_report,
)
from ._runner import BenchmarkRunner
from ._types import (
    BenchmarkDataset,
    BenchmarkSuite,
    CalibrationReport,
    EvaluationReport,
    MetricResult,
    PipelinePrediction,
    RegressionComparison,
)
from .graph import GraphEvaluationReport, evaluate_graph_quality
from .ontology import (
    DEFAULT_ONTOLOGY_EVALUATORS,
    OntologyEvaluationReport,
    OntologyEvaluationResult,
    OntologyMetricEvaluator,
    evaluate_ontology,
)

__all__ = [
    "DEFAULT_ONTOLOGY_EVALUATORS",
    "BenchmarkDataset",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "CalibrationReport",
    "EvaluationReport",
    "GraphEvaluationReport",
    "MetricResult",
    "OntologyEvaluationReport",
    "OntologyEvaluationResult",
    "OntologyMetricEvaluator",
    "PipelinePrediction",
    "RegressionComparison",
    "compare_reports",
    "evaluate_graph_quality",
    "evaluate_ontology",
    "load_benchmark_datasets",
    "load_benchmark_suite",
    "load_evaluation_report",
    "load_official_benchmark_suite",
    "load_prediction_artifact",
    "render_markdown_report",
    "render_regression_markdown",
    "save_json_report",
    "save_markdown_report",
]
