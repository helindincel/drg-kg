"""
Pytest configuration and shared fixtures for DRG tests.

This module provides fixtures and configuration that are shared across all tests.
"""

import contextlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# drg/extract/__init__.py imports dspy at module level (so tests can patch it).
# Stub it here — before any drg module is imported — so that the entire test
# suite can collect and run without requiring a live dspy installation.
# Tests that need a real LLM are marked `integration` and skipped in CI.
#
# IMPORTANT: dspy.Module must be a real Python type so that `class KGExtractor(dspy.Module)`
# creates a proper class (not a MagicMock). Using a plain MagicMock as the base makes
# Python treat MagicMock as the metaclass, turning KGExtractor itself into a MagicMock.
_dspy_stub = MagicMock()
_dspy_stub.Module = type(
    "Module",
    (),
    {
        "__init__": lambda self, *args, **kwargs: None,
        # dspy.Module.__call__ delegates to forward() — replicate that behaviour.
        "__call__": lambda self, *args, **kwargs: self.forward(*args, **kwargs),
    },
)
# dspy.settings.lm must be None so mock-mode short-circuit fires when no real LM
# is configured (extract_typed checks `dspy.settings.lm is None`).
_dspy_stub.settings = MagicMock()
_dspy_stub.settings.lm = None
_dspy_stub.context = lambda **_kwargs: contextlib.nullcontext()
_dspy_stub.configure = Mock()
_dspy_stub.LM = Mock()
_dspy_stub.JSONAdapter = type("JSONAdapter", (), {})
sys.modules.setdefault("dspy", _dspy_stub)


@pytest.fixture
def mock_dspy():
    """Fixture to mock DSPy module."""
    with patch("drg.extract.dspy") as mock_dspy:
        mock_predictor_instance = Mock()
        mock_dspy.Predict = Mock(return_value=mock_predictor_instance)
        mock_dspy.Module = type("Module", (), {})
        mock_dspy.Signature = type("Signature", (), {})
        mock_dspy.InputField = Mock()
        mock_dspy.OutputField = Mock()
        mock_dspy.Prediction = Mock
        mock_dspy.Example = Mock
        mock_dspy.configure = Mock()
        mock_dspy.LM = Mock()
        mock_dspy.context = Mock(return_value=contextlib.nullcontext())
        mock_dspy.JSONAdapter = type("JSONAdapter", (), {})

        # Mock Assert/Suggest if available
        if hasattr(mock_dspy, "Assert"):
            mock_dspy.Assert = Mock()
        if hasattr(mock_dspy, "Suggest"):
            mock_dspy.Suggest = Mock()

        yield mock_dspy


@pytest.fixture
def sample_schema():
    """Fixture providing a sample DRGSchema."""
    from drg.schema import DRGSchema, Entity, Relation

    return DRGSchema(
        entities=[
            Entity("Person"),
            Entity("Company"),
            Entity("Product"),
        ],
        relations=[
            Relation("works_at", "Person", "Company"),
            Relation("produces", "Company", "Product"),
        ],
    )


@pytest.fixture
def sample_enhanced_schema():
    """Fixture providing a sample EnhancedDRGSchema."""
    from drg.schema import EnhancedDRGSchema, EntityType, Relation, RelationGroup

    return EnhancedDRGSchema(
        entity_types=[
            EntityType(
                name="Person", description="Individuals", examples=["John", "Jane"], properties={}
            ),
            EntityType(
                name="Company",
                description="Business organizations",
                examples=["Apple", "Tesla"],
                properties={},
            ),
        ],
        relation_groups=[
            RelationGroup(
                name="employment",
                description="Employment relationships",
                relations=[
                    Relation(
                        name="works_at",
                        src="Person",
                        dst="Company",
                        description="Employment relationship",
                        detail="Person works at company",
                    )
                ],
            )
        ],
    )


@pytest.fixture
def sample_text():
    """Fixture providing sample text for extraction."""
    return "John works at Tesla. Tesla produces electric vehicles."


@pytest.fixture
def sample_entities():
    """Fixture providing sample extracted entities."""
    return [
        ("John", "Person"),
        ("Tesla", "Company"),
        ("electric vehicles", "Product"),
    ]


@pytest.fixture
def sample_relations():
    """Fixture providing sample extracted relations."""
    return [
        ("John", "works_at", "Tesla"),
        ("Tesla", "produces", "electric vehicles"),
    ]


@pytest.fixture
def mock_embedding_provider():
    """Fixture providing a mock embedding provider."""
    from drg.embedding.providers import EmbeddingProvider

    class MockEmbeddingProvider(EmbeddingProvider):
        def embed(self, text: str):
            return [0.1] * 384  # Mock embedding vector

        def embed_batch(self, texts: list[str]):
            return [[0.1] * 384 for _ in texts]

        def get_dimension(self):
            return 384

        def get_model_name(self):
            return "mock-embedding-model"

    return MockEmbeddingProvider()


@pytest.fixture(autouse=True)
def disable_lm_config(monkeypatch):
    """Automatically disable LM configuration in tests to avoid API calls."""
    if os.getenv("DRG_RUN_INTEGRATION", "").lower() in {"1", "true", "yes"}:
        # Integration tests may require real LM configuration.
        return

    def mock_configure_lm():
        pass

    monkeypatch.setattr("drg.extract._configure_llm_auto", mock_configure_lm)
    monkeypatch.setattr("drg.config.configure_lm", mock_configure_lm)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration (disabled by default; set DRG_RUN_INTEGRATION=1 to enable)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly enabled."""
    if os.getenv("DRG_RUN_INTEGRATION", "").lower() in {"1", "true", "yes"}:
        return

    skip_integration = pytest.mark.skip(
        reason="Integration tests disabled by default. Set DRG_RUN_INTEGRATION=1 to enable."
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
