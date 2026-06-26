"""Comparison and rendering helpers for evaluation reports."""

from __future__ import annotations

from typing import Any

from ._types import (
    EvaluationReport,
    MetricDelta,
    RegressionComparison,
)

__all__ = [
    "compare_reports",
    "render_markdown_report",
    "render_regression_markdown",
]


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    regression_threshold: float = 0.02,
    metric_thresholds: dict[str, float] | None = None,
    dataset_thresholds: dict[str, float] | None = None,
) -> RegressionComparison:
    """Compare two :class:`~drg.evaluation.EvaluationReport` instances.

    A regression is declared when a candidate metric drops more than
    *regression_threshold* below the baseline value.

    Parameters
    ----------
    baseline:
        The reference report (e.g. a previously published run).
    candidate:
        The new report to evaluate against the baseline.
    regression_threshold:
        Maximum allowed drop in F1 before a regression is flagged.
    """
    checks: list[tuple[str, float, float]] = [
        ("entity_f1", baseline.entity_metrics.f1, candidate.entity_metrics.f1),
        ("relation_f1", baseline.relation_metrics.f1, candidate.relation_metrics.f1),
        ("event_f1", baseline.event_metrics.f1, candidate.event_metrics.f1),
        ("reasoning_f1", baseline.reasoning_metrics.f1, candidate.reasoning_metrics.f1),
        ("evidence_f1", baseline.evidence_metrics.f1, candidate.evidence_metrics.f1),
        ("weighted_f1", baseline.weighted_f1, candidate.weighted_f1),
    ]

    deltas: list[MetricDelta] = []
    any_regressed = False
    thresholds = metric_thresholds or {}

    for name, base_val, cand_val in checks:
        delta = cand_val - base_val
        threshold = thresholds.get(name, regression_threshold)
        regressed = delta < -threshold
        if regressed:
            any_regressed = True
        if regressed:
            status = "regressed"
        elif delta > threshold:
            status = "improved"
        else:
            status = "unchanged"
        deltas.append(
            MetricDelta(
                metric=name,
                baseline=base_val,
                candidate=cand_val,
                delta=delta,
                regressed=regressed,
                threshold=threshold,
                status=status,
            )
        )

    dataset_deltas = _compare_dataset_rows(
        baseline.per_dataset,
        candidate.per_dataset,
        default_threshold=regression_threshold,
        dataset_thresholds=dataset_thresholds or {},
    )
    if any(row.get("regressed") for row in dataset_deltas):
        any_regressed = True

    return RegressionComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        regression_threshold=regression_threshold,
        overall_regressed=any_regressed,
        deltas=deltas,
        dataset_deltas=dataset_deltas,
        metadata={
            "baseline_dataset_count": baseline.dataset_count,
            "candidate_dataset_count": candidate.dataset_count,
            "metric_thresholds": thresholds,
            "dataset_thresholds": dataset_thresholds or {},
        },
    )


def _compare_dataset_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    default_threshold: float,
    dataset_thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    candidate_by_name = {str(row.get("dataset")): row for row in candidate_rows}
    rows: list[dict[str, Any]] = []
    for base in baseline_rows:
        name = str(base.get("dataset"))
        cand = candidate_by_name.get(name)
        if cand is None:
            rows.append({"dataset": name, "status": "missing_candidate", "regressed": True})
            continue
        threshold = dataset_thresholds.get(name, default_threshold)
        metric_deltas: dict[str, float] = {}
        regressed = False
        for metric in ("entity_f1", "relation_f1", "event_f1", "reasoning_f1", "evidence_f1"):
            base_val = float(base.get(metric, 0.0) or 0.0)
            cand_val = float(cand.get(metric, 0.0) or 0.0)
            delta = cand_val - base_val
            metric_deltas[metric] = round(delta, 4)
            if delta < -threshold:
                regressed = True
        rows.append(
            {
                "dataset": name,
                "domain": base.get("domain"),
                "difficulty": base.get("difficulty"),
                "threshold": threshold,
                "regressed": regressed,
                "status": "regressed" if regressed else "ok",
                "metric_deltas": metric_deltas,
            }
        )
    return rows


def render_markdown_report(report: EvaluationReport) -> str:
    """Render an :class:`~drg.evaluation.EvaluationReport` as a Markdown string."""
    lines: list[str] = []
    meta: dict[str, Any] = report.metadata or {}
    model = meta.get("model", "unknown")
    lines.append(f"# Evaluation Report — `{report.run_id}`\n")
    lines.append(f"**Model:** {model}  ")
    lines.append(f"**Datasets evaluated:** {report.dataset_count}  ")
    lines.append(f"**Weighted F1:** {report.weighted_f1:.4f}\n")

    lines.append("## Metrics\n")
    lines.append("| Category | Precision | Recall | F1 |")
    lines.append("|:---|---:|---:|---:|")
    for label, m in [
        ("Entity", report.entity_metrics),
        ("Relation", report.relation_metrics),
        ("Event", report.event_metrics),
        ("Reasoning", report.reasoning_metrics),
        ("Evidence", report.evidence_metrics),
    ]:
        lines.append(f"| {label} | {m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} |")
    lines.append("")

    if report.calibration.sample_count:
        lines.append("## Confidence Calibration\n")
        lines.append(f"* **Samples:** {report.calibration.sample_count}")
        lines.append(f"* **ECE:** {report.calibration.expected_calibration_error:.4f}")
        lines.append(f"* **Brier score:** {report.calibration.brier_score:.4f}")
        lines.append("")

    if report.per_dataset:
        lines.append("## Per-dataset Results\n")
        lines.append(
            "| Dataset | Entity F1 | Relation F1 | Event F1 | Reasoning F1 | Evidence F1 |"
        )
        lines.append("|:---|---:|---:|---:|---:|---:|")
        for row in report.per_dataset:
            lines.append(
                f"| {row['dataset']} "
                f"| {row.get('entity_f1', 0):.4f} "
                f"| {row.get('relation_f1', 0):.4f} "
                f"| {row.get('event_f1', 0):.4f} "
                f"| {row.get('reasoning_f1', 0):.4f} "
                f"| {row.get('evidence_f1', 0):.4f} |"
            )
        lines.append("")

    if report.performance:
        lines.append("## Performance\n")
        for key, val in report.performance.items():
            lines.append(f"* **{key}**: {val}")
        lines.append("")

    return "\n".join(lines)


def render_regression_markdown(comparison: RegressionComparison) -> str:
    """Render a :class:`~drg.evaluation.RegressionComparison` as Markdown."""
    lines: list[str] = []
    status = "REGRESSION DETECTED" if comparison.overall_regressed else "OK"
    lines.append(f"# Regression Comparison — {status}\n")
    lines.append(f"**Baseline:** `{comparison.baseline_run_id}`  ")
    lines.append(f"**Candidate:** `{comparison.candidate_run_id}`  ")
    lines.append(f"**Threshold:** {comparison.regression_threshold}\n")

    lines.append("## Metric Deltas\n")
    lines.append("| Metric | Baseline | Candidate | Delta | Status |")
    lines.append("|:---|---:|---:|---:|:---|")
    for d in comparison.deltas:
        status = "REGRESSION" if d.regressed else d.status.upper()
        lines.append(
            f"| {d.metric} | {d.baseline:.4f} | {d.candidate:.4f} | {d.delta:+.4f} | {status} |"
        )
    lines.append("")

    if comparison.dataset_deltas:
        lines.append("## Dataset Deltas\n")
        lines.append("| Dataset | Status | Entity Δ | Relation Δ | Event Δ | Evidence Δ |")
        lines.append("|:---|:---|---:|---:|---:|---:|")
        for row in comparison.dataset_deltas:
            deltas = row.get("metric_deltas", {})
            lines.append(
                f"| {row.get('dataset')} "
                f"| {row.get('status')} "
                f"| {deltas.get('entity_f1', 0):+.4f} "
                f"| {deltas.get('relation_f1', 0):+.4f} "
                f"| {deltas.get('event_f1', 0):+.4f} "
                f"| {deltas.get('evidence_f1', 0):+.4f} |"
            )
        lines.append("")
    return "\n".join(lines)
