"""Tests for the confidence scoring framework (``drg.confidence``).

Coverage:
    - ``ConfidenceScore`` value clamping (NaN, out-of-range, frozen semantics).
    - ``DefaultConfidenceStrategy`` entity scoring with/without schema and text.
    - ``DefaultConfidenceStrategy`` relation scoring including upstream
      confidence pass-through, schema validity boost, negation penalty,
      and temporal cue boost.
    - ``KGNode.confidence`` validation, serialisation, JSON round-trip,
      and JSON-LD/enriched-format export.
    - ``build_enhanced_kg`` end-to-end integration with the default
      strategy, with custom strategies, with explicit overrides, and with
      ``confidence_strategy=None`` (legacy mode).
    - Neo4j-style export readiness: confirms confidence survives both the
      ``to_dict`` and ``to_enriched_format`` serialisation paths.

These tests do not require DSPy or any LLM; they exercise the pure
strategy/builder layer directly.
"""

from __future__ import annotations

import json
import math

import pytest

from drg.confidence import (
    ConfidenceScore,
    ConfidenceStrategy,
    DefaultConfidenceStrategy,
    clamp_confidence,
)
from drg.graph.builders import build_enhanced_kg
from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode
from drg.schema import (
    DRGSchema,
    EnhancedDRGSchema,
    Entity,
    EntityType,
    Relation,
    RelationGroup,
)

# ---------------------------------------------------------------------------
# clamp_confidence + ConfidenceScore
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (-0.1, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.2, 1.0),
    ],
)
def test_clamp_confidence_bounds_input(raw: float, expected: float):
    assert clamp_confidence(raw) == expected


def test_clamp_confidence_handles_nan():
    assert clamp_confidence(float("nan")) == 0.0


def test_confidence_score_clamps_value_at_construction():
    sc = ConfidenceScore(value=1.5)
    assert sc.value == 1.0

    sc = ConfidenceScore(value=-0.2)
    assert sc.value == 0.0


def test_confidence_score_to_dict_round_trips_signals_and_method():
    sc = ConfidenceScore(value=0.7, signals={"base": 0.5, "boost": 0.2}, method="default")
    d = sc.to_dict()
    assert d == {"value": 0.7, "signals": {"base": 0.5, "boost": 0.2}, "method": "default"}


def test_confidence_score_is_frozen():
    sc = ConfidenceScore(value=0.5)
    with pytest.raises(Exception):
        sc.value = 0.9  # frozen dataclass forbids reassignment


# ---------------------------------------------------------------------------
# DefaultConfidenceStrategy — entities
# ---------------------------------------------------------------------------


def _enh_schema() -> EnhancedDRGSchema:
    return EnhancedDRGSchema(
        entity_types=[
            EntityType(name="Company", description="c"),
            EntityType(name="Product", description="p"),
        ],
        relation_groups=[
            RelationGroup(
                name="g",
                description="g",
                relations=[Relation("produces", "Company", "Product", description="p")],
            )
        ],
    )


def test_default_strategy_scores_entity_with_no_signals():
    strat = DefaultConfidenceStrategy()
    scores = strat.score_entities([("Apple", "Company")])
    assert "Apple" in scores
    sc = scores["Apple"]
    # base only: 0.6
    assert sc.value == pytest.approx(strat.BASE_ENTITY_SCORE)
    assert sc.method == "default"


def test_default_strategy_boosts_entity_when_type_in_schema():
    strat = DefaultConfidenceStrategy()
    schema = _enh_schema()
    scores = strat.score_entities([("Apple", "Company")], context={"schema": schema})
    sc = scores["Apple"]
    # base + type_in_schema = 0.6 + 0.15
    expected = strat.BASE_ENTITY_SCORE + strat.BOOST_TYPE_IN_SCHEMA
    assert sc.value == pytest.approx(expected)
    assert "type_in_schema" in sc.signals


def test_default_strategy_boosts_when_name_appears_in_text():
    strat = DefaultConfidenceStrategy()
    text = "Apple released the iPhone."
    scores = strat.score_entities([("Apple", "Company")], context={"source_text": text})
    assert "name_in_text" in scores["Apple"].signals


def test_default_strategy_boosts_multi_word_entities():
    strat = DefaultConfidenceStrategy()
    scores = strat.score_entities([("Steve Jobs", "Person")])
    assert "multi_word" in scores["Steve Jobs"].signals


def test_default_strategy_skips_entities_with_empty_name():
    strat = DefaultConfidenceStrategy()
    scores = strat.score_entities([("", "Company"), ("Apple", "Company")])
    assert "" not in scores
    assert "Apple" in scores


# ---------------------------------------------------------------------------
# DefaultConfidenceStrategy — relations
# ---------------------------------------------------------------------------


def test_default_strategy_relation_base_only():
    strat = DefaultConfidenceStrategy()
    scores = strat.score_relations([("Apple", "produces", "iPhone")])
    sc = scores[("Apple", "produces", "iPhone")]
    assert sc.value == pytest.approx(strat.BASE_RELATION_SCORE)


def test_default_strategy_relation_schema_validity_boost():
    strat = DefaultConfidenceStrategy()
    schema = _enh_schema()
    scores = strat.score_relations(
        [("Apple", "produces", "iPhone")],
        context={
            "schema": schema,
            "entity_types": {"Apple": "Company", "iPhone": "Product"},
        },
    )
    sc = scores[("Apple", "produces", "iPhone")]
    # base + both_typed + schema_valid = 0.5 + 0.10 + 0.20 = 0.80
    expected = strat.BASE_RELATION_SCORE + strat.BOOST_BOTH_TYPED + strat.BOOST_SCHEMA_VALID
    assert sc.value == pytest.approx(expected)
    assert "schema_valid" in sc.signals


def test_default_strategy_relation_negation_penalty():
    strat = DefaultConfidenceStrategy()
    enriched = [
        {
            "relation": ("Apple", "produces", "iPhone"),
            "confidence": None,
            "is_negated": True,
            "temporal": None,
        }
    ]
    scores = strat.score_relations(
        [("Apple", "produces", "iPhone")],
        enriched_relations=enriched,
    )
    sc = scores[("Apple", "produces", "iPhone")]
    # base - negated penalty
    expected = strat.BASE_RELATION_SCORE - strat.PENALTY_NEGATED
    assert sc.value == pytest.approx(expected)
    assert sc.signals["negated"] < 0


def test_default_strategy_relation_temporal_boost():
    strat = DefaultConfidenceStrategy()
    enriched = [
        {
            "relation": ("Apple", "produces", "iPhone"),
            "confidence": None,
            "temporal": {"start": "2020-01-01", "end": None},
            "is_negated": False,
        }
    ]
    scores = strat.score_relations(
        [("Apple", "produces", "iPhone")],
        enriched_relations=enriched,
    )
    assert "temporal" in scores[("Apple", "produces", "iPhone")].signals


def test_default_strategy_honours_upstream_confidence():
    strat = DefaultConfidenceStrategy()
    enriched = [
        {
            "relation": ("Apple", "produces", "iPhone"),
            "confidence": 0.92,  # pretend an LLM rated this
            "is_negated": False,
            "temporal": None,
        }
    ]
    scores = strat.score_relations(
        [("Apple", "produces", "iPhone")],
        enriched_relations=enriched,
    )
    sc = scores[("Apple", "produces", "iPhone")]
    assert sc.value == pytest.approx(0.92)
    assert sc.method.endswith("upstream")


def test_default_strategy_skips_malformed_triples():
    strat = DefaultConfidenceStrategy()
    # not a 3-tuple
    scores = strat.score_relations([("only", "two")])  # type: ignore[list-item]
    assert scores == {}


def test_default_strategy_relation_score_never_below_min_floor():
    """With heavy penalties and zero boosts, the score must still be >= 0.05."""
    strat = DefaultConfidenceStrategy()
    enriched = [
        {
            "relation": ("Apple", "produces", "iPhone"),
            "confidence": None,
            "is_negated": True,
            "temporal": None,
        }
    ]
    scores = strat.score_relations(
        [("Apple", "produces", "iPhone")],
        enriched_relations=enriched,
    )
    sc = scores[("Apple", "produces", "iPhone")]
    assert sc.value >= 0.05


# ---------------------------------------------------------------------------
# Strategy contract — custom implementations are honoured
# ---------------------------------------------------------------------------


class _ConstStrategy(ConfidenceStrategy):
    """Trivial strategy that returns a constant score for everything."""

    name = "constant"

    def __init__(self, value: float):
        self._value = value

    def score_entities(self, entities, *, context=None):
        return {
            name: ConfidenceScore(value=self._value, method=self.name)
            for name, _ in entities
            if name
        }

    def score_relations(self, relations, *, enriched_relations=None, context=None):
        return {t: ConfidenceScore(value=self._value, method=self.name) for t in relations}


def test_custom_strategy_can_be_passed_to_builder():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
        confidence_strategy=_ConstStrategy(0.42),
    )
    assert kg.nodes["Apple"].confidence == pytest.approx(0.42)
    assert kg.nodes["iPhone"].confidence == pytest.approx(0.42)
    assert kg.edges[0].confidence == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# KGNode confidence — model-layer guarantees
# ---------------------------------------------------------------------------


def test_kgnode_default_confidence_is_none_for_legacy_compatibility():
    node = KGNode(id="x")
    assert node.confidence is None


@pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
def test_kgnode_accepts_boundary_confidence(ok: float):
    assert KGNode(id="x", confidence=ok).confidence == ok


@pytest.mark.parametrize("bad", [-0.01, 1.01, -1.0, 2.0])
def test_kgnode_rejects_out_of_range_confidence(bad: float):
    with pytest.raises(ValueError, match=r"Confidence score must be between 0\.0 and 1\.0"):
        KGNode(id="x", confidence=bad)


def test_kgnode_to_dict_omits_confidence_when_none():
    """Backward compatibility: bare nodes serialise as before."""
    d = KGNode(id="x").to_dict()
    assert "confidence" not in d


def test_kgnode_to_dict_emits_confidence_when_set():
    d = KGNode(id="x", confidence=0.8).to_dict()
    assert d["confidence"] == 0.8


def test_kgnode_round_trips_confidence_through_dict():
    original = KGNode(id="x", type="T", confidence=0.77)
    restored = KGNode.from_dict(original.to_dict())
    assert restored == original
    assert restored.confidence == 0.77


# ---------------------------------------------------------------------------
# Builder integration — default strategy attaches confidence end-to-end
# ---------------------------------------------------------------------------


def test_builder_default_strategy_attaches_confidence_to_nodes():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
        schema=_enh_schema(),
    )
    assert kg.nodes["Apple"].confidence is not None
    assert 0.0 <= kg.nodes["Apple"].confidence <= 1.0
    assert kg.nodes["iPhone"].confidence is not None


def test_builder_default_strategy_attaches_confidence_to_edges():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
        schema=_enh_schema(),
    )
    edge = kg.edges[0]
    assert edge.confidence is not None
    assert 0.0 <= edge.confidence <= 1.0


def test_builder_legacy_mode_when_confidence_strategy_none():
    """Opt-out: passing ``confidence_strategy=None`` matches the pre-feature
    behaviour. No confidence attached to nodes/edges."""
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
        confidence_strategy=None,
    )
    assert kg.nodes["Apple"].confidence is None
    assert kg.edges[0].confidence is None


def test_builder_explicit_entity_overrides_take_precedence():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company")],
        triples=[],
        entity_confidences={"Apple": 0.99},
    )
    assert kg.nodes["Apple"].confidence == pytest.approx(0.99)


def test_builder_explicit_relation_overrides_take_precedence():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
        relation_confidences={("Apple", "produces", "iPhone"): 0.13},
    )
    assert kg.edges[0].confidence == pytest.approx(0.13)


def test_builder_uses_enriched_relations_when_provided():
    enriched = [
        {
            "relation": ("Apple", "produces", "iPhone"),
            "confidence": 0.88,
            "is_negated": False,
            "temporal": None,
        }
    ]
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
        enriched_relations=enriched,
    )
    # Upstream confidence should be respected verbatim by the default strategy.
    assert kg.edges[0].confidence == pytest.approx(0.88)


def test_builder_works_with_legacy_drg_schema():
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")],
    )
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
        schema=schema,
    )
    # No crash + confidence is attached.
    assert kg.nodes["Apple"].confidence is not None
    assert kg.edges[0].confidence is not None


# ---------------------------------------------------------------------------
# Serialisation — confidence survives all export formats
# ---------------------------------------------------------------------------


def test_confidence_survives_to_json_round_trip():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
    )
    data = json.loads(kg.to_json())
    apple = next(n for n in data["nodes"] if n["id"] == "Apple")
    assert "confidence" in apple
    edge = data["edges"][0]
    assert "confidence" in edge


def test_confidence_survives_to_json_ld():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
    )
    data = json.loads(kg.to_json_ld())
    apple_ld = next(n for n in data["nodes"] if n["identifier"] == "Apple")
    assert "confidence" in apple_ld
    assert 0.0 <= apple_ld["confidence"] <= 1.0


def test_confidence_survives_to_enriched_format():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
    )
    data = json.loads(kg.to_enriched_format())
    apple = next(n for n in data["entities"] if n["id"] == "Apple")
    assert "confidence" in apple
    rel = data["relationships"][0]
    assert "confidence" in rel


def test_legacy_kg_without_confidence_still_serialises_cleanly():
    """A graph constructed manually (no strategy) must still round-trip."""
    kg = EnhancedKG()
    kg.add_node(KGNode(id="a", type="T"))
    kg.add_node(KGNode(id="b", type="T"))
    kg.add_edge(
        KGEdge(
            source="a",
            target="b",
            relationship_type="rel",
            relationship_detail="a rel b",
        )
    )
    data = json.loads(kg.to_json())
    # Confidence must be absent (not null) to stay byte-equivalent to the
    # pre-feature output.
    assert "confidence" not in data["nodes"][0]
    assert "confidence" not in data["edges"][0]


# ---------------------------------------------------------------------------
# Sanity checks on score numerical stability
# ---------------------------------------------------------------------------


def test_default_strategy_never_produces_nan_scores():
    strat = DefaultConfidenceStrategy()
    scores = strat.score_entities([("Apple", "Company"), ("Steve Jobs", "Person")])
    for sc in scores.values():
        assert not math.isnan(sc.value)
        assert 0.0 <= sc.value <= 1.0
