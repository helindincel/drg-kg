from drg.evaluation import (
    BenchmarkDataset,
    BenchmarkRunner,
    GoldEntity,
    GoldRelation,
    PipelinePrediction,
)


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
