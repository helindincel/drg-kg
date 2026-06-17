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
    precision_recall_f1,
    render_markdown_report,
    retrieval_metrics,
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
        query_results={
            "Who has worked with Sam Altman?": ["Sam Altman", "OpenAI", "Microsoft"],
            "What acquisitions involve Microsoft?": ["Microsoft", "GitHub"],
        },
        hybrid_results={
            "Who has worked with Sam Altman?": ["OpenAI", "Sam Altman"],
            "What acquisitions involve Microsoft?": [
                "Microsoft",
                "GitHub",
                "doc_github_chunk_000",
            ],
        },
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


def test_retrieval_metrics_precision_recall_mrr_ndcg():
    metrics = retrieval_metrics(["a", "x", "b"], {"a", "b"}, k=3)
    assert metrics["precision_at_k"] == 2 / 3
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert 0.0 < metrics["ndcg"] <= 1.0


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
    report = BenchmarkRunner(run_id="test_run", retrieval_k=3).evaluate(
        [dataset],
        predictions={dataset.name: _prediction()},
    )

    assert report.aggregate["extraction_f1"] == 1.0
    assert report.aggregate["hybrid_ndcg"] >= report.aggregate["retrieval_ndcg"]
    ds = report.datasets[0]
    assert ds.components["event_extraction"].metrics["f1"] == 1.0
    assert ds.components["query_reasoning"].metrics["inference_f1"] == 1.0


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
    good = BenchmarkRunner(run_id="good", retrieval_k=3).evaluate(
        [dataset],
        predictions={dataset.name: _prediction()},
    )
    weak_prediction = _prediction()
    weak_prediction.entities = weak_prediction.entities[:-2]
    weak = BenchmarkRunner(run_id="weak", retrieval_k=3).evaluate(
        [dataset],
        predictions={dataset.name: weak_prediction},
    )

    comparison = compare_reports(good, weak, regression_threshold=0.01)
    assert comparison.regressions
    assert any(item["metric"] == "overall_score" for item in comparison.regressions)


def test_render_markdown_report_contains_components():
    dataset = load_benchmark_dataset("examples/benchmarks/synthetic_kg_benchmark.json")
    report = BenchmarkRunner(run_id="markdown").evaluate(
        [dataset],
        predictions={dataset.name: _prediction()},
    )
    markdown = render_markdown_report(report)
    assert "# Evaluation Report: markdown" in markdown
    assert "entity_extraction" in markdown
    assert "hybrid_retrieval" in markdown
