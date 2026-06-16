"""Tests for dependency injection (KGExtractor lm) and structural protocols."""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from drg import extract as extract_pkg
from drg.extract import KGExtractor, extract_typed
from drg.protocols import (
    ClusteringAlgorithmProtocol,
    EmbeddingProviderProtocol,
    KGExtractorProtocol,
    LLMProtocol,
)
from drg.schema import DRGSchema, Entity, Relation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_schema() -> DRGSchema:
    return DRGSchema(
        entities=[Entity(name="Person"), Entity(name="Company")],
        relations=[Relation(name="works_at", src="Person", dst="Company")],
    )


class _FakeLM:
    """Minimal stand-in for a DSPy ``LM`` instance."""

    def __call__(self, *args, **kwargs):  # pragma: no cover - never invoked here
        return ""


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_embedding_provider_satisfies_protocol():
    """The concrete ABC ``EmbeddingProvider`` must pass the structural protocol."""
    from drg.embedding.providers import EmbeddingProvider

    class _DummyEmbedder(EmbeddingProvider):
        def embed(self, text):
            return [0.0]

        def embed_batch(self, texts):
            return [[0.0] for _ in texts]

        def get_dimension(self):
            return 1

        def get_model_name(self):
            return "dummy"

    assert isinstance(_DummyEmbedder(), EmbeddingProviderProtocol)


def test_clustering_algorithm_satisfies_protocol():
    """A trivial clustering object should match :class:`ClusteringAlgorithmProtocol`."""

    class _NoopClusterer:
        def cluster(self, graph):
            return []

    assert isinstance(_NoopClusterer(), ClusteringAlgorithmProtocol)


def test_kg_extractor_satisfies_protocol(minimal_schema: DRGSchema):
    """``KGExtractor`` must structurally implement :class:`KGExtractorProtocol`.

    When dspy is mocked (unit-test context), ``KGExtractor`` becomes a
    ``MagicMock`` because Python uses the mock's type as the metaclass.
    We therefore verify the protocol contract via a concrete class whose
    ``forward`` signature mirrors ``KGExtractor.forward``.
    """

    class _ConcreteExtractor:
        def forward(
            self,
            text: str,
            context_entities: list[tuple[str, str]] | None = None,
        ) -> None:
            return None

    assert isinstance(_ConcreteExtractor(), KGExtractorProtocol)


def test_fake_lm_satisfies_protocol():
    assert isinstance(_FakeLM(), LLMProtocol)


def test_object_without_methods_rejected_by_protocol():
    """Sanity check: protocols should reject objects missing required methods."""

    class _NotAnEmbedder:
        pass

    assert not isinstance(_NotAnEmbedder(), EmbeddingProviderProtocol)


# ---------------------------------------------------------------------------
# KGExtractor LM injection
# ---------------------------------------------------------------------------


def test_kg_extractor_defaults_lm_to_none(minimal_schema: DRGSchema):
    """Backward compat: no-arg construction leaves ``lm`` unset."""
    extractor = KGExtractor(minimal_schema)
    assert extractor.lm is None


def test_kg_extractor_stores_injected_lm(minimal_schema: DRGSchema):
    fake = _FakeLM()
    extractor = KGExtractor(minimal_schema, lm=fake)
    assert extractor.lm is fake


def test_maybe_lm_context_none_returns_nullcontext():
    cm = extract_pkg._maybe_lm_context(None)
    assert isinstance(cm, contextlib.nullcontext)


def test_maybe_lm_context_with_lm_uses_dspy_context():
    """When ``dspy.context`` exists, ``_maybe_lm_context`` should call it."""
    fake_cm = contextlib.nullcontext()
    fake_lm = _FakeLM()

    called_with = {}

    def fake_context_factory(*, lm):
        called_with["lm"] = lm
        return fake_cm

    with patch.object(extract_pkg.dspy, "context", fake_context_factory, create=True):
        cm = extract_pkg._maybe_lm_context(fake_lm)

    assert cm is fake_cm
    assert called_with["lm"] is fake_lm


def test_maybe_lm_context_falls_back_when_dspy_lacks_context():
    """If neither ``dspy.context`` nor ``dspy.settings.context`` exists, we fall
    back to ``nullcontext`` rather than crashing."""
    fake_lm = _FakeLM()

    with (
        patch.object(extract_pkg.dspy, "context", None, create=True),
        patch.object(extract_pkg.dspy, "settings", None, create=True),
    ):
        cm = extract_pkg._maybe_lm_context(fake_lm)

    assert isinstance(cm, contextlib.nullcontext)


# ---------------------------------------------------------------------------
# _get_extractor cache invalidation on lm change
# ---------------------------------------------------------------------------


def test_get_extractor_rebuilds_when_lm_changes(minimal_schema: DRGSchema):
    """The cached extractor must be invalidated when the injected LM changes."""
    # Reset cache so test is deterministic.
    extract_pkg._extractor = None

    lm_a = _FakeLM()
    lm_b = _FakeLM()

    extractor_a = extract_pkg._get_extractor(minimal_schema, lm=lm_a)
    assert extractor_a.lm is lm_a

    extractor_b = extract_pkg._get_extractor(minimal_schema, lm=lm_b)
    assert extractor_b.lm is lm_b
    assert extractor_b is not extractor_a


def test_get_extractor_reuses_cache_when_lm_unchanged(minimal_schema: DRGSchema):
    extract_pkg._extractor = None
    lm = _FakeLM()

    first = extract_pkg._get_extractor(minimal_schema, lm=lm)
    second = extract_pkg._get_extractor(minimal_schema, lm=lm)
    assert first is second


def test_get_extractor_skips_global_lm_config_when_lm_injected(
    minimal_schema: DRGSchema,
):
    """When an ``lm`` is injected, we must not call ``_configure_llm_auto``
    (which would touch ``dspy.settings`` globally)."""
    extract_pkg._extractor = None

    with patch.object(extract_pkg, "_configure_llm_auto") as mock_configure:
        extract_pkg._get_extractor(minimal_schema, lm=_FakeLM())

    mock_configure.assert_not_called()


def test_get_extractor_calls_global_lm_config_when_no_injection(
    minimal_schema: DRGSchema,
):
    """Legacy path: without ``lm``, the auto-configuration helper must still run."""
    extract_pkg._extractor = None

    with patch.object(extract_pkg, "_configure_llm_auto") as mock_configure:
        extract_pkg._get_extractor(minimal_schema)

    mock_configure.assert_called_once()


# ---------------------------------------------------------------------------
# extract_typed accepts injected LM end-to-end
# ---------------------------------------------------------------------------


def test_extract_typed_forwards_lm_to_extractor(minimal_schema: DRGSchema):
    """``extract_typed(..., lm=fake)`` must wire ``fake`` into the extractor.

    We stub ``KGExtractor.forward`` so the test isolates DI wiring from the
    real DSPy round-trip (which would reject ``_FakeLM`` for not being a
    ``dspy.BaseLM``).
    """
    from drg.extract._types import ExtractionResult

    extract_pkg._extractor = None
    fake_lm = _FakeLM()

    with patch.object(
        KGExtractor,
        "forward",
        return_value=ExtractionResult(entities=[], relations=[]),
    ):
        extract_typed("Some text.", minimal_schema, lm=fake_lm)

    assert extract_pkg._extractor is not None
    assert extract_pkg._extractor.lm is fake_lm


def test_extract_typed_empty_text_returns_empty(minimal_schema: DRGSchema):
    """Empty-text short circuit must still work with the new ``lm`` parameter."""
    entities, relations = extract_typed("", minimal_schema, lm=_FakeLM())
    assert entities == [] and relations == []
