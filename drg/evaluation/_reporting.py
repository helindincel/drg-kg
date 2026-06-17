"""Loading and report rendering for evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._types import BenchmarkDataset, EvaluationReport, RegressionComparison

__all__ = [
    "load_benchmark_dataset",
    "load_benchmark_datasets",
    "render_markdown_report",
    "render_regression_markdown",
    "save_json_report",
    "save_markdown_report",
]


def load_benchmark_dataset(path: str | Path) -> BenchmarkDataset:
    """Load one benchmark dataset from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkDataset.from_dict(data)


def load_benchmark_datasets(path: str | Path) -> list[BenchmarkDataset]:
    """Load a dataset or a list of datasets from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [BenchmarkDataset.from_dict(item) for item in data]
    if isinstance(data, dict) and "datasets" in data:
        return [BenchmarkDataset.from_dict(item) for item in data["datasets"]]
    return [BenchmarkDataset.from_dict(data)]


def save_json_report(report: EvaluationReport, path: str | Path) -> None:
    """Write report JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def save_markdown_report(report: EvaluationReport, path: str | Path) -> None:
    """Write report Markdown."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: EvaluationReport) -> str:
    """Render a human-readable evaluation report."""
    lines: list[str] = []
    lines.append(f"# Evaluation Report: {report.run_id}")
    lines.append("")
    if report.metadata:
        lines.append("## Metadata")
        for key, value in sorted(report.metadata.items()):
            lines.append(f"- **{key}**: {_fmt_value(value)}")
        lines.append("")

    lines.append("## Aggregate Metrics")
    lines.extend(_metric_table(report.aggregate))
    lines.append("")

    for dataset in report.datasets:
        lines.append(f"## Dataset: {dataset.dataset_name}")
        lines.append("")
        lines.append("### Overall")
        lines.extend(_metric_table(dataset.overall))
        lines.append("")
        lines.append("### Components")
        lines.append("| Component | Metric | Value |")
        lines.append("|---|---:|---:|")
        for component in dataset.components.values():
            for metric, value in sorted(component.metrics.items()):
                lines.append(f"| {component.name} | {metric} | {value:.4f} |")
        lines.append("")
        failures = [
            failure
            for component in dataset.components.values()
            for failure in component.failures
        ]
        if failures:
            lines.append("### Failure Cases")
            for failure in failures:
                lines.append(
                    f"- `{failure.get('metric')}` = {failure.get('value'):.4f}: "
                    f"{failure.get('description')}"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_regression_markdown(comparison: RegressionComparison) -> str:
    """Render old-vs-new regression comparison."""
    lines = [
        f"# Regression Comparison: {comparison.baseline_run_id} -> {comparison.candidate_run_id}",
        "",
        "## Regressions",
    ]
    if comparison.regressions:
        for item in comparison.regressions:
            lines.append(
                f"- {item['scope']} `{item['metric']}`: "
                f"{item['before']:.4f} -> {item['after']:.4f} ({item['delta']:.4f})"
            )
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Improvements")
    if comparison.improvements:
        for item in comparison.improvements:
            lines.append(
                f"- {item['scope']} `{item['metric']}`: "
                f"{item['before']:.4f} -> {item['after']:.4f} (+{item['delta']:.4f})"
            )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _metric_table(metrics: dict[str, float]) -> list[str]:
    lines = ["| Metric | Value |", "|---|---:|"]
    for key, value in sorted(metrics.items()):
        lines.append(f"| {key} | {value:.4f} |")
    return lines


def _fmt_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)
