"""Tests for the typed exception hierarchy."""

from __future__ import annotations

import pytest

from drg.errors import (
    ClusteringError,
    ConfigError,
    CoreferenceResolutionError,
    DRGError,
    EmbeddingError,
    EntityResolutionError,
    ExtractionError,
    GraphError,
    LLMConfigError,
    ResolutionError,
    SchemaError,
    SchemaGenerationError,
)


def test_all_errors_descend_from_drgerror():
    """Catching :class:`DRGError` at a boundary must cover every leaf."""
    for cls in (
        ConfigError,
        LLMConfigError,
        SchemaError,
        SchemaGenerationError,
        ExtractionError,
        ResolutionError,
        EntityResolutionError,
        CoreferenceResolutionError,
        EmbeddingError,
        ClusteringError,
        GraphError,
    ):
        assert issubclass(cls, DRGError), cls.__name__


def test_config_errors_are_value_errors_for_back_compat():
    """User code that already catches :class:`ValueError` for config issues
    must keep working after we migrate raises to :class:`ConfigError`."""
    assert issubclass(ConfigError, ValueError)
    assert issubclass(LLMConfigError, ValueError)
    assert issubclass(SchemaError, ValueError)


def test_runtime_errors_inherit_runtimeerror():
    """Same compatibility check for runtime-style failures."""
    for cls in (
        ExtractionError,
        ResolutionError,
        EntityResolutionError,
        CoreferenceResolutionError,
        EmbeddingError,
        ClusteringError,
        GraphError,
    ):
        assert issubclass(cls, RuntimeError), cls.__name__


def test_schema_generation_error_is_both_schema_and_runtime():
    """Multi-inheritance is deliberate; either ``except`` clause must catch it."""
    err = SchemaGenerationError("bad output")
    assert isinstance(err, SchemaError)
    assert isinstance(err, RuntimeError)


def test_resolution_subtypes_are_distinct():
    """Entity and coreference failures share a parent but aren't interchangeable."""
    assert not issubclass(EntityResolutionError, CoreferenceResolutionError)
    assert not issubclass(CoreferenceResolutionError, EntityResolutionError)


def test_drgerror_round_trip_message():
    """Sanity: the base class behaves like a normal Exception."""
    err = DRGError("oops")
    assert str(err) == "oops"
    with pytest.raises(DRGError, match="oops"):
        raise err
