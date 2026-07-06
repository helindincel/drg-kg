from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from drg.confidence import CalibratedConfidenceStrategy, CalibrationPoint
from drg.errors import LLMConfigError
from drg.extract import extract_from_chunks
from drg.optimizer import KGOptimizerConfig, optimize_extractor
from drg.schema import DRGSchema, Entity, Relation


@pytest.fixture()
def company_schema() -> DRGSchema:
    return DRGSchema(
        entities=[Entity("Company"), Entity("Person"), Entity("Product")],
        relations=[
            Relation("founded_by", "Company", "Person"),
            Relation("produces", "Company", "Product"),
        ],
    )


def _result(entities=None, relations=None, enriched=None):
    return SimpleNamespace(
        entities=entities or [],
        relations=relations or [],
        enriched_relations=enriched,
    )


def test_extract_from_chunks_requires_lm_in_strict_mode(company_schema, monkeypatch):
    monkeypatch.setenv("DRG_REQUIRE_LM", "1")

    with pytest.raises(LLMConfigError, match="No DSPy LM"):
        extract_from_chunks([{"text": "Apple produces iPhone."}], company_schema)


def test_chunked_extraction_filters_negation_and_low_confidence(company_schema, monkeypatch):
    monkeypatch.setenv("DRG_WINDOWED_RELATION_EXTRACTION", "never")
    extractor = Mock(return_value=_result(entities=[("Apple", "Company"), ("iPhone", "Product")]))
    extractor.extract_document_relations = Mock(
        return_value=_result(
            entities=[("Apple", "Company"), ("iPhone", "Product")],
            relations=[
                ("Apple", "produces", "iPhone"),
                ("Apple", "produces", "iPhone Legacy"),
                ("Apple", "produces", "iPhone Mini"),
            ],
            enriched=[
                {
                    "relation": ("Apple", "produces", "iPhone"),
                    "confidence": 0.91,
                    "is_negated": False,
                    "metadata": {},
                },
                {
                    "relation": ("Apple", "produces", "iPhone Legacy"),
                    "confidence": 0.95,
                    "is_negated": True,
                    "metadata": {},
                },
                {
                    "relation": ("Apple", "produces", "iPhone Mini"),
                    "confidence": 0.2,
                    "is_negated": False,
                    "metadata": {},
                },
            ],
        )
    )

    with (
        patch("drg.extract._get_extractor", return_value=extractor),
        patch("drg.extract.resolve_entities_and_relations", None),
        patch("drg.extract.dspy") as mock_dspy,
    ):
        mock_dspy.settings.lm = Mock()
        entities, triples, enriched = extract_from_chunks(
            [{"text": "Apple produces iPhone but not iPhone Legacy."}],
            company_schema,
            return_enriched=True,
            min_confidence=0.5,
            enable_implicit_relationships=False,
        )

    assert entities == [("Apple", "Company"), ("iPhone", "Product")]
    assert triples == [("Apple", "produces", "iPhone")]
    assert enriched[0]["confidence"] == 0.91


def test_windowed_document_relation_extraction_limits_context(company_schema, monkeypatch):
    monkeypatch.setenv("DRG_WINDOWED_RELATION_EXTRACTION", "always")
    monkeypatch.setenv("DRG_MAX_RELATION_CANDIDATE_PAIRS", "2")
    monkeypatch.setenv("DRG_MAX_RELATION_EVIDENCE_WINDOWS", "2")

    extractor = Mock(
        return_value=_result(
            entities=[("Apple", "Company"), ("Steve Jobs", "Person"), ("iPhone", "Product")]
        )
    )
    extractor.extract_document_relations = Mock(
        return_value=_result(
            relations=[("Apple", "founded_by", "Steve Jobs")],
            enriched=[
                {
                    "relation": ("Apple", "founded_by", "Steve Jobs"),
                    "confidence": 0.88,
                    "is_negated": False,
                    "metadata": {},
                }
            ],
        )
    )

    with (
        patch("drg.extract._get_extractor", return_value=extractor),
        patch("drg.extract.resolve_entities_and_relations", None),
        patch("drg.extract.dspy") as mock_dspy,
    ):
        mock_dspy.settings.lm = Mock()
        _entities, triples, enriched = extract_from_chunks(
            [
                {"text": "Apple is a company."},
                {"text": "Steve Jobs co-founded Apple."},
                {"text": "The iPhone became a flagship product."},
            ],
            company_schema,
            return_enriched=True,
            enable_implicit_relationships=False,
        )

    assert ("Apple", "founded_by", "Steve Jobs") in triples
    assert enriched[0]["metadata"]["source"] == "windowed_document_relation_extraction"
    called_chunks = extractor.extract_document_relations.call_args.kwargs["chunks"]
    assert len(called_chunks) < 3


def test_calibrated_confidence_strategy_maps_scores():
    strategy = CalibratedConfidenceStrategy(
        [
            CalibrationPoint(predicted=0.9, actual=True),
            CalibrationPoint(predicted=0.8, actual=False),
        ],
        bins=5,
    )

    scores = strategy.score_relations(
        [("Apple", "produces", "iPhone")],
        enriched_relations=[
            {
                "relation": ("Apple", "produces", "iPhone"),
                "confidence": 0.85,
            }
        ],
    )

    score = scores[("Apple", "produces", "iPhone")]
    assert score.method == "calibrated"
    assert "raw_confidence" in score.signals


def test_optimizer_compiles_document_relation_stage(monkeypatch):
    compiled = object()
    captured = {}

    class FakeBootstrapFewShot:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def compile(self, program, *, trainset):
            captured["program"] = program
            captured["trainset"] = trainset
            return compiled

    class FakeExample:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.inputs = ()

        def with_inputs(self, *names):
            self.inputs = names
            return self

    fake_dspy = SimpleNamespace(
        BootstrapFewShot=FakeBootstrapFewShot,
        Example=FakeExample,
        Module=object,
    )
    monkeypatch.setitem(__import__("sys").modules, "dspy", fake_dspy)

    extractor = Mock()
    config = KGOptimizerConfig(pipeline_stage="document_relations")
    result = optimize_extractor(
        [
            {
                "chunks": [{"text": "Apple was founded by Steve Jobs."}],
                "entities": [("Apple", "Company"), ("Steve Jobs", "Person")],
                "expected_relations": [("Apple", "founded_by", "Steve Jobs")],
            }
        ],
        config=config,
        extractor=extractor,
    )

    assert result is compiled
    assert captured["program"].pipeline_stage == "document_relations"
    assert captured["trainset"][0].inputs == ("chunks", "entities")
