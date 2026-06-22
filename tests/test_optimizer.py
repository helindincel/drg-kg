from __future__ import annotations

from types import SimpleNamespace

from drg.optimizer.optimizer import DRGOptimizer, OptimizerConfig, OptimizerType
from drg.schema import DRGSchema, Entity, Relation


class _FakeExample:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inputs = ()

    def with_inputs(self, *names):
        self.inputs = names
        return self


def test_bootstrap_optimizer_compile_path_uses_supported_dspy_kwargs(monkeypatch):
    import drg.optimizer.optimizer as opt_mod

    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")],
    )
    base_extractor = object()
    optimized_extractor = object()
    created = []

    class FakeBootstrap:
        def __init__(self, metric, max_bootstrapped_demos, max_labeled_demos):
            self.metric = metric
            self.max_bootstrapped_demos = max_bootstrapped_demos
            self.max_labeled_demos = max_labeled_demos
            self.compile_calls = []
            created.append(self)

        def compile(self, student, trainset, valset=None):
            self.compile_calls.append({"student": student, "trainset": trainset, "valset": valset})
            return optimized_extractor

    fake_dspy = SimpleNamespace(
        BootstrapFewShot=FakeBootstrap,
        Example=_FakeExample,
    )
    monkeypatch.setattr(opt_mod, "dspy", fake_dspy)
    monkeypatch.setattr(opt_mod, "KGExtractor", lambda _schema, lm=None: base_extractor)

    optimizer = DRGOptimizer(
        schema=schema,
        config=OptimizerConfig(optimizer_type=OptimizerType.BOOTSTRAP_FEW_SHOT),
        training_examples=[
            {
                "text": "Apple produces iPhone.",
                "expected_entities": [("Apple", "Company"), ("iPhone", "Product")],
                "expected_relations": [("Apple", "produces", "iPhone")],
            }
        ],
    )

    validation_examples = [
        {
            "text": "Google produces Android.",
            "expected_entities": [("Google", "Company"), ("Android", "Product")],
            "expected_relations": [("Google", "produces", "Android")],
        }
    ]
    result = optimizer.optimize(validation_examples=validation_examples)

    assert result is optimized_extractor
    assert optimizer.optimized_extractor is optimized_extractor
    assert len(created) == 1
    assert created[0].metric.__self__ is optimizer
    assert created[0].metric.__func__ is optimizer._default_metric.__func__
    assert created[0].compile_calls[0]["student"] is base_extractor
    trainset = created[0].compile_calls[0]["trainset"]
    assert trainset[0].kwargs["text"] == "Apple produces iPhone."
    assert trainset[0].inputs == ("text",)
    valset = created[0].compile_calls[0]["valset"]
    assert valset[0].kwargs["text"] == "Google produces Android."
    assert optimizer.last_compile_config["trainset"] is trainset
    assert optimizer.last_compile_config["valset"] is valset
    assert optimizer.last_compile_config["metric"].__self__ is optimizer


def test_optimizer_requires_training_examples(monkeypatch):
    import pytest
    import drg.optimizer.optimizer as opt_mod

    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")],
    )
    monkeypatch.setattr(opt_mod, "KGExtractor", lambda _schema, lm=None: object())

    optimizer = DRGOptimizer(schema=schema, training_examples=[])

    with pytest.raises(ValueError, match="training example"):
        optimizer.optimize()


def test_optimizer_rejects_non_compiling_optimizer(monkeypatch):
    import pytest
    import drg.optimizer.optimizer as opt_mod

    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")],
    )

    class FakeBootstrap:
        def __init__(self, **_kwargs):
            pass

    fake_dspy = SimpleNamespace(
        BootstrapFewShot=FakeBootstrap,
        Example=_FakeExample,
    )
    monkeypatch.setattr(opt_mod, "dspy", fake_dspy)
    monkeypatch.setattr(opt_mod, "KGExtractor", lambda _schema, lm=None: object())

    optimizer = DRGOptimizer(
        schema=schema,
        training_examples=[
            {
                "text": "Apple produces iPhone.",
                "expected_entities": [("Apple", "Company"), ("iPhone", "Product")],
                "expected_relations": [("Apple", "produces", "iPhone")],
            }
        ],
    )

    with pytest.raises(RuntimeError, match="compile"):
        optimizer.optimize()
