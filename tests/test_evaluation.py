from drg.evaluation import (
    BenchmarkDataset,
    BenchmarkRunner,
    GoldEntity,
    GoldRelation,
    PipelinePrediction,
    retrieval_metrics,
)


def test_retrieval_metrics():
    ranked = ["A", "B", "C", "D"]
    relevant = {"B", "D", "E"}
    metrics = retrieval_metrics(ranked, relevant, k=4)

    # B is at index 1 (rank 2), D is at index 3 (rank 4)
    # Correct: 2, Total relevant in top 4: 2
    assert metrics["precision_at_k"] == 0.5  # 2/4
    assert metrics["recall_at_k"] == 2 / 3
    assert metrics["hits_at_k"] == 1.0
    assert metrics["mrr"] == 0.5  # 1/2 (first hit at rank 2)


def test_benchmark_runner_basic():
    ds = BenchmarkDataset(
        name="test_ds",
        text="Apple is in Cupertino.",
        gold_entities=[GoldEntity("Apple", "Company"), GoldEntity("Cupertino", "City")],
        gold_relations=[GoldRelation("Apple", "LOCATED_IN", "Cupertino")],
    )

    pred = PipelinePrediction(
        entities=[("Apple", "Company"), ("Cupertino", "City")],
        relations=[("Apple", "LOCATED_IN", "Cupertino")],
    )

    runner = BenchmarkRunner(run_id="test_run")
    report = runner.evaluate([ds], predictions={ds.name: pred})

    assert report.aggregate["extraction_f1"] == 1.0
    assert report.datasets[0].components["entity_extraction"].metrics["f1"] == 1.0
    assert report.datasets[0].components["relationship_extraction"].metrics["f1"] == 1.0


def test_benchmark_runner_partial():
    ds = BenchmarkDataset(
        name="test_ds",
        text="Apple is in Cupertino.",
        gold_entities=[GoldEntity("Apple", "Company"), GoldEntity("Cupertino", "City")],
        gold_relations=[GoldRelation("Apple", "LOCATED_IN", "Cupertino")],
    )

    # Missing Cupertino
    pred = PipelinePrediction(entities=[("Apple", "Company")], relations=[])

    runner = BenchmarkRunner(run_id="test_run")
    report = runner.evaluate([ds], predictions={ds.name: pred})

    # Entity P=1, R=0.5, F1=0.666
    ent_metrics = report.datasets[0].components["entity_extraction"].metrics
    assert ent_metrics["precision"] == 1.0
    assert ent_metrics["recall"] == 0.5

    # Relation R=0
    rel_metrics = report.datasets[0].components["relationship_extraction"].metrics
    assert rel_metrics["recall"] == 0.0
