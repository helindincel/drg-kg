"""
Pytest configuration and shared fixtures for DRG tests.

This module provides fixtures and configuration that are shared across all tests.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_dspy():
    """Fixture to mock DSPy module."""
    with patch("drg.extract.dspy") as mock_dspy:
        # Mock TypedPredictor
        mock_typed_predictor = Mock()
        mock_dspy.TypedPredictor = Mock(return_value=mock_typed_predictor)
        mock_dspy.Predict = Mock()
        mock_dspy.Module = type("Module", (), {})
        mock_dspy.Signature = type("Signature", (), {})
        mock_dspy.InputField = Mock()
        mock_dspy.OutputField = Mock()
        mock_dspy.Prediction = Mock
        mock_dspy.Example = Mock
        mock_dspy.configure = Mock()

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
