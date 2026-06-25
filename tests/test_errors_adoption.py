"""Regression tests: typed exceptions are raised at the right call sites.

These tests pin the new exception adoption — if a future change accidentally
reverts to ``RuntimeError``/``ValueError`` the suite catches it. We also
verify backward compatibility: the typed exceptions inherit from the
original built-in classes, so legacy callers using broad ``except``
clauses still work.
"""

from __future__ import annotations

from unittest import mock

import pytest

from drg.errors import (
    ConfigError,
    DRGError,
    ExtractionError,
    GraphError,
    LLMConfigError,
    SchemaGenerationError,
)

# ---------------------------------------------------------------------------
# Exception hierarchy: typed exceptions stay compatible with builtins
# ---------------------------------------------------------------------------


def test_llm_config_error_inherits_value_error():
    """LLMConfigError → ConfigError → (DRGError, ValueError).

    ConfigError sits under ``ValueError`` because invalid/missing configuration
    is a bad-input condition, not a runtime fault. Legacy code that caught
    ``ValueError`` keeps working.
    """
    err = LLMConfigError("missing key")
    assert isinstance(err, ConfigError)
    assert isinstance(err, DRGError)
    assert isinstance(err, ValueError)


def test_extraction_error_back_compat():
    """ExtractionError must satisfy ``except RuntimeError`` for legacy callers."""
    err = ExtractionError("bad output")
    assert isinstance(err, DRGError)
    assert isinstance(err, RuntimeError)


def test_schema_generation_error_back_compat():
    err = SchemaGenerationError("empty schema")
    assert isinstance(err, DRGError)
    assert isinstance(err, RuntimeError)


def test_graph_error_back_compat():
    err = GraphError("hub dominance")
    assert isinstance(err, DRGError)
    assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# extract_typed: missing LM in strict mode raises LLMConfigError
# ---------------------------------------------------------------------------


def test_extract_typed_raises_llm_config_error_when_lm_required(monkeypatch):
    """DRG_REQUIRE_LM=1 + no LM ⇒ LLMConfigError, not bare ValueError."""
    from drg.extract import extract_typed
    from drg.schema import EnhancedDRGSchema, EntityType, Relation, RelationGroup

    schema = EnhancedDRGSchema(
        entity_types=[EntityType(name="Person", description="people")],
        relation_groups=[
            RelationGroup(
                name="default",
                description="",
                relations=[
                    Relation(name="knows", src="Person", dst="Person", description="", detail="")
                ],
            )
        ],
    )

    monkeypatch.setenv("DRG_REQUIRE_LM", "1")
    # Force no LM available globally either.
    with mock.patch("drg.extract.dspy.settings") as fake_settings:
        fake_settings.lm = None
        with pytest.raises(LLMConfigError):
            extract_typed("Alice knows Bob", schema)


def test_extract_typed_legacy_value_error_still_caught(monkeypatch):
    """Legacy callers catching ValueError keep working (back-compat sanity)."""
    from drg.extract import extract_typed
    from drg.schema import EnhancedDRGSchema, EntityType, Relation, RelationGroup

    schema = EnhancedDRGSchema(
        entity_types=[EntityType(name="Person", description="people")],
        relation_groups=[
            RelationGroup(
                name="default",
                description="",
                relations=[
                    Relation(name="knows", src="Person", dst="Person", description="", detail="")
                ],
            )
        ],
    )
    monkeypatch.setenv("DRG_REQUIRE_LM", "1")
    with mock.patch("drg.extract.dspy.settings") as fake_settings:
        fake_settings.lm = None
        # ValueError covers our new typed exception via inheritance chain.
        with pytest.raises(ValueError):
            extract_typed("Alice knows Bob", schema)


# ---------------------------------------------------------------------------
# Hub dominance validation: GraphError
# ---------------------------------------------------------------------------


def test_hub_dominance_validation_raises_graph_error(monkeypatch):
    """When the hub-dominance gate is enabled in error mode, we raise GraphError."""
    from drg.extract import _validate_hub_dominance

    monkeypatch.setenv("DRG_VALIDATE_HUB_DOMINANCE", "1")
    monkeypatch.setenv("DRG_HUB_VALIDATION_MODE", "error")
    monkeypatch.setenv("DRG_MAX_HUB_RATIO", "0.30")
    monkeypatch.setenv("DRG_MIN_DIVERSITY_RATIO", "0.50")

    # 50 triples all routed through "Hub" — dominates the graph well past 30%.
    triples = [("Hub", "rel", f"E{i}") for i in range(50)]

    with pytest.raises(GraphError):
        _validate_hub_dominance(triples)


def test_hub_dominance_validation_warn_mode_does_not_raise(monkeypatch):
    """In warn mode the gate logs but never raises — production callers can opt in."""
    from drg.extract import _validate_hub_dominance

    monkeypatch.setenv("DRG_VALIDATE_HUB_DOMINANCE", "1")
    monkeypatch.setenv("DRG_HUB_VALIDATION_MODE", "warn")

    triples = [("Hub", "rel", f"E{i}") for i in range(50)]

    # Must not raise.
    _validate_hub_dominance(triples)


# ---------------------------------------------------------------------------
# Schema generation: SchemaGenerationError on empty/invalid output
# ---------------------------------------------------------------------------


def test_schema_generation_raises_on_empty_schema():
    """Empty parsed schema ⇒ SchemaGenerationError, not bare RuntimeError.

    Mocks the DSPy generator so we don't need a live LM. The fake produces
    an empty JSON object, which should trip the ``empty schema`` guard in
    :func:`drg.extract._schema_gen.generate_schema_from_text`.
    """
    from drg.extract import _schema_gen

    class _FakeResult:
        generated_schema = "{}"

    class _FakeGen:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return _FakeResult()

    # Patch the actual generator factories actually called by the function.
    # ``create=True`` lets us patch attributes the current DSPy install
    # doesn't expose (e.g. TypedPredictor was removed in DSPy ≥2.5).
    with (
        mock.patch.object(_schema_gen.dspy, "Predict", _FakeGen, create=True),
        mock.patch.object(_schema_gen.dspy, "ChainOfThought", _FakeGen, create=True),
        mock.patch("drg.extract._configure_llm_auto"),
    ):
        with pytest.raises(SchemaGenerationError):
            _schema_gen.generate_schema_from_text("dummy text")


def test_schema_generation_error_inherits_runtime_error():
    """Backward compat: legacy callers using ``except RuntimeError`` still trigger."""
    try:
        raise SchemaGenerationError("nope")
    except RuntimeError as e:
        assert isinstance(e, SchemaGenerationError)
    else:  # pragma: no cover - safety
        pytest.fail("SchemaGenerationError must inherit RuntimeError")
