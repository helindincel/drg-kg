#!/usr/bin/env python3
"""Evaluation framework example — no LLM required.

Demonstrates how to use :class:`drg.evaluation.BenchmarkRunner` to evaluate
pipeline predictions against gold-standard benchmark datasets.

Covers:
- Loading the built-in benchmark suite
- Creating pre-computed predictions (no live LLM call)
- Running evaluation and inspecting metrics
- Comparing two runs for regressions
- Saving and loading reports

Run::

    python examples/evaluation_framework_example.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drg.evaluation import (
    BenchmarkRunner,
    PipelinePrediction,
    compare_reports,
    load_evaluation_report,
    load_official_benchmark_suite,
    render_markdown_report,
    render_regression_markdown,
    save_json_report,
    save_markdown_report,
)


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Simulate a "good" pipeline with near-perfect predictions
# ---------------------------------------------------------------------------


def _make_good_predictions(suite) -> list[PipelinePrediction]:
    """Return predictions that closely match the gold standard (high F1)."""
    preds = []
    for ds in suite.datasets:
        # Inject one false positive to avoid perfect 1.0 (more realistic)
        entities = list(ds.gold_entities) + [{"name": "Spurious Entity", "type": "Unknown"}]
        relations = list(ds.gold_relations)
        preds.append(
            PipelinePrediction(
                entities=entities,
                relations=relations,
                metadata={"source": "simulated_good"},
            )
        )
    return preds


def _make_poor_predictions(suite) -> list[PipelinePrediction]:
    """Return predictions with low recall (only first gold entity/relation)."""
    preds = []
    for ds in suite.datasets:
        entities = ds.gold_entities[:1] if ds.gold_entities else []
        relations = ds.gold_relations[:1] if ds.gold_relations else []
        preds.append(
            PipelinePrediction(
                entities=entities,
                relations=relations,
                metadata={"source": "simulated_poor"},
            )
        )
    return preds


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load the official benchmark suite
    # ------------------------------------------------------------------
    _section("Official benchmark suite")
    suite = load_official_benchmark_suite()
    print(f"  Suite name  : {suite.name}")
    print(f"  Datasets    : {len(suite.datasets)}")
    for ds in suite.datasets:
        task = ds.metadata.get("task") or ds.metadata.get("domain") or "general"
        print(f"    - {ds.name} ({task}): {len(ds.gold_entities)} gold entities, "
              f"{len(ds.gold_relations)} gold relations")

    # ------------------------------------------------------------------
    # 2. Evaluate "good" predictions
    # ------------------------------------------------------------------
    _section("Evaluating good predictions")
    runner_good = BenchmarkRunner(
        run_id="demo-good-v1",
        measure_performance=True,
        metadata={"model": "simulated", "version": "good"},
    )
    good_preds = _make_good_predictions(suite)
    report_good = runner_good.evaluate(suite.datasets, predictions=good_preds)

    print(f"  Run ID            : {report_good.run_id}")
    print(f"  Datasets evaluated: {report_good.dataset_count}")
    print(f"  Entity F1         : {report_good.entity_metrics.f1:.4f}")
    print(f"  Relation F1       : {report_good.relation_metrics.f1:.4f}")
    print(f"  Weighted F1       : {report_good.weighted_f1:.4f}")
    if report_good.performance:
        print(f"  Mean latency      : {report_good.performance.get('mean_seconds', 'n/a')} s")

    # ------------------------------------------------------------------
    # 3. Evaluate "poor" predictions
    # ------------------------------------------------------------------
    _section("Evaluating poor predictions")
    runner_poor = BenchmarkRunner(
        run_id="demo-poor-v1",
        metadata={"model": "simulated", "version": "poor"},
    )
    poor_preds = _make_poor_predictions(suite)
    report_poor = runner_poor.evaluate(suite.datasets, predictions=poor_preds)

    print(f"  Entity F1   : {report_poor.entity_metrics.f1:.4f}")
    print(f"  Relation F1 : {report_poor.relation_metrics.f1:.4f}")
    print(f"  Weighted F1 : {report_poor.weighted_f1:.4f}")

    # ------------------------------------------------------------------
    # 4. Per-dataset breakdown
    # ------------------------------------------------------------------
    _section("Per-dataset results (good run)")
    for row in report_good.per_dataset:
        print(
            f"  {row['dataset']:30s}  entity_f1={row['entity_f1']:.4f}  "
            f"relation_f1={row['relation_f1']:.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Markdown rendering
    # ------------------------------------------------------------------
    _section("Markdown report (good run, first 600 chars)")
    md = render_markdown_report(report_good)
    print(md[:600])

    # ------------------------------------------------------------------
    # 6. Regression comparison
    # ------------------------------------------------------------------
    _section("Regression comparison: good (baseline) vs. poor (candidate)")
    comparison = compare_reports(report_good, report_poor, regression_threshold=0.05)
    print(f"  Overall regressed: {comparison.overall_regressed}")
    for delta in comparison.deltas:
        status = "REGRESSION" if delta.regressed else "OK"
        print(f"  {delta.metric:20s}  {delta.baseline:.4f} → {delta.candidate:.4f}  "
              f"(Δ{delta.delta:+.4f})  {status}")

    print()
    print(render_regression_markdown(comparison))

    # ------------------------------------------------------------------
    # 7. Save and load
    # ------------------------------------------------------------------
    _section("Save/load round-trip")
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "report.json"
        md_path = Path(tmpdir) / "report.md"

        save_json_report(report_good, json_path)
        save_markdown_report(report_good, md_path)
        print(f"  Saved JSON report  : {json_path}")
        print(f"  Saved Markdown     : {md_path}")

        # Load it back
        loaded = load_evaluation_report(json_path)
        assert loaded.run_id == report_good.run_id
        assert abs(loaded.weighted_f1 - report_good.weighted_f1) < 1e-6
        print(f"  Round-trip passed  : run_id={loaded.run_id}, weighted_f1={loaded.weighted_f1}")

    # ------------------------------------------------------------------
    # 8. Runner callable API (alternative to pre-computed predictions)
    # ------------------------------------------------------------------
    _section("Runner callable API")

    def mock_pipeline(ds):
        # Simulate a pipeline that returns all gold entities/relations
        return PipelinePrediction(
            entities=ds.gold_entities,
            relations=ds.gold_relations,
            metadata={"source": "mock_pipeline"},
        )

    report_callable = BenchmarkRunner(run_id="demo-callable").evaluate(
        suite.datasets, runner=mock_pipeline
    )
    print(f"  Perfect mock F1   : {report_callable.weighted_f1:.4f} (expect 1.0000)")


if __name__ == "__main__":
    main()
