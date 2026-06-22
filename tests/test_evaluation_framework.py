from __future__ import annotations

from drg.evaluation import (
    BenchmarkRunner,
    GoldEntity,
    GoldEvent,
    GoldRelation,
    PipelinePrediction,
    compare_reports,
    graph_metrics,
    load_benchmark_dataset,
    load_benchmark_suite,
    load_evaluation_report,
    load_official_benchmark_suite,
    load_prediction_artifact,
    precision_recall_f1,
    render_markdown_report,
    save_json_report,
)


def _prediction() -> PipelinePrediction:
    return PipelinePrediction(
        entities=[
            ("Sam Altman", "Person"),
            ("OpenAI", "Company"),
            ("Microsoft", "Company"),
            ("GitHub", "Company"),
            ("GitHub Acquisition", "Event:Acquisition"),
        ],
        relations=[
            ("Sam Altman", "WORKED_WITH", "OpenAI"),
            ("Microsoft", "INVESTED_IN", "OpenAI"),
            ("Microsoft", "ACQUIRED", "GitHub"),
        ],
        events=[
            {
                "event_type": "Acquisition",
                "participants": {"acquirer": ["Microsoft"], "acquired": ["GitHub"]},
                "timestamp": "2018",
            }
        ],
        inferred_relations=[("Sam Altman", "CONNECTED_TO", "Microsoft")],
        resolved_clusters={
            "Sam Altman": "ai",
            "OpenAI": "ai",
            "Microsoft": "ai",
            "GitHub": "developer_tools",
        },
        communities={
            "Sam Altman": "ai",
            "OpenAI": "ai",
            "Microsoft": "ai",
            "GitHub": "developer_tools",
        },
    )


def test_precision_recall_f1_exact_sets():
    metrics = precision_recall_f1({"a", "b"}, {"b", "c"})
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_graph_metrics_coverage_density_orphans():
    metrics = graph_metrics(
        gold_entities=[
            GoldEntity("A", "Thing"),
            GoldEntity("B", "Thing"),
            GoldEntity("C", "Thing"),
        ],
        gold_relations=[GoldRelation("A", "LINKS", "B")],
        predicted_entities=[("A", "Thing"), ("B", "Thing"), ("D", "Thing")],
        predicted_relations=[("A", "LINKS", "B")],
    )
    assert metrics["entity_coverage"] == 2 / 3
    assert metrics["relation_coverage"] == 1.0
    assert metrics["orphan_node_rate"] == 1 / 3


def test_benchmark_runner_full_report():
    dataset = load_benchmark_dataset("examples/benchmarks/synthetic_kg_benchmark.json")
    report = BenchmarkRunner(run_id="test_run").evaluate(
        [dataset],
        predictions={dataset.name: _prediction()},
    )

    assert report.aggregate["extraction_f1"] == 1.0
    removed_metric = "query" + "_ndcg"
    assert removed_metric not in report.aggregate
    ds = report.datasets[0]
    assert ds.components["event_extraction"].metrics["f1"] == 1.0
    assert ds.components["query_reasoning"].metrics["inference_f1"] == 1.0


def test_benchmark_runner_can_measure_runtime_performance():
    dataset = load_benchmark_dataset("examples/benchmarks/synthetic_kg_benchmark.json")
    report = BenchmarkRunner(run_id="perf", measure_performance=True).evaluate(
        [dataset],
        runner=lambda _dataset: _prediction(),
    )

    assert "performance" in report.metadata
    assert report.metadata["performance"]["total_wall_time_seconds"] >= 0.0
    runtime = report.datasets[0].components["runtime_performance"].metrics
    assert runtime["wall_time_seconds"] >= 0.0
    assert runtime["entities_per_second"] >= 0.0


def test_official_benchmark_suite_loads_dataset_catalog():
    suite = load_official_benchmark_suite()
    assert suite.name == "drg_official_minimal_suite"
    assert {dataset.name for dataset in suite.datasets} >= {
        "synthetic_openai_microsoft",
        "corporate_acquisition_benchmark",
    }
    assert "external-baseline" in suite.adapters


def test_benchmark_suite_manifest_supports_runner_callable(tmp_path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        """
{
  "name": "inline_suite",
  "datasets": [
    {
      "name": "tiny",
      "text": "Apple acquired Beats.",
      "gold_entities": [
        {"name": "Apple", "type": "Company"},
        {"name": "Beats", "type": "Company"}
      ],
      "gold_relations": [
        {"source": "Apple", "relationship_type": "ACQUIRED", "target": "Beats"}
      ]
    }
  ],
  "adapters": ["drg"]
}
""",
        encoding="utf-8",
    )
    suite = load_benchmark_suite(suite_path)

    def runner(dataset):
        assert dataset.name == "tiny"
        return PipelinePrediction(
            entities=[("Apple", "Company"), ("Beats", "Company")],
            relations=[("Apple", "ACQUIRED", "Beats")],
        )

    report = BenchmarkRunner(run_id="suite_smoke").evaluate(suite.datasets, runner=runner)
    assert report.aggregate["extraction_f1"] == 1.0


def test_event_key_matching_supports_gold_event_dicts():
    dataset = load_benchmark_dataset("examples/benchmarks/synthetic_kg_benchmark.json")
    assert isinstance(dataset.gold_events[0], GoldEvent)
    report = BenchmarkRunner(run_id="events").evaluate(
        [dataset],
        predictions={dataset.name: _prediction()},
    )
    assert report.datasets[0].components["event_extraction"].metrics["correct"] == 1.0


def test_compare_reports_flags_regression_and_improvement():
    dataset = load_benchmark_dataset("examples/benchmarks/synthetic_kg_benchmark.json")
    good = BenchmarkRunner(run_id="good").evaluate(
        [dataset],
        predictions={dataset.name: _prediction()},
    )
    weak_prediction = _prediction()
    weak_prediction.entities = weak_prediction.entities[:-2]
    weak = BenchmarkRunner(run_id="weak").evaluate(
        [dataset],
        predictions={dataset.name: weak_prediction},
    )

    comparison = compare_reports(good, weak, regression_threshold=0.01)
    assert comparison.regressions
    assert any(item["metric"] == "overall_score" for item in comparison.regressions)


def test_evaluation_report_json_round_trip(tmp_path):
    dataset = load_benchmark_dataset("examples/benchmarks/synthetic_kg_benchmark.json")
    report = BenchmarkRunner(run_id="roundtrip").evaluate(
        [dataset],
        predictions={dataset.name: _prediction()},
    )
    report_path = tmp_path / "report.json"

    save_json_report(report, report_path)
    loaded = load_evaluation_report(report_path)

    assert loaded.run_id == "roundtrip"
    assert loaded.aggregate == report.aggregate
    assert loaded.datasets[0].components["entity_extraction"].metrics["f1"] == 1.0


def test_prediction_artifact_loader_supports_external_adapter_shape(tmp_path):
    artifact = tmp_path / "predictions.json"
    artifact.write_text(
        """
{
  "adapter": "external-baseline",
  "model": "baseline",
  "predictions": {
    "tiny": {
      "entities": [{"name": "Apple", "type": "Company"}],
      "relations": [["Apple", "ACQUIRED", "Beats"]]
    }
  }
}
""",
        encoding="utf-8",
    )

    predictions, metadata = load_prediction_artifact(artifact)

    assert metadata["adapter"] == "external-baseline"
    assert predictions["tiny"].entities == [("Apple", "Company")]
    assert predictions["tiny"].relations == [("Apple", "ACQUIRED", "Beats")]


def test_render_markdown_report_contains_components():
    dataset = load_benchmark_dataset("examples/benchmarks/synthetic_kg_benchmark.json")
    report = BenchmarkRunner(run_id="markdown").evaluate(
        [dataset],
        predictions={dataset.name: _prediction()},
    )
    markdown = render_markdown_report(report)
    assert "# Evaluation Report: markdown" in markdown
    assert "entity_extraction" in markdown
    assert "graph_construction" in markdown
