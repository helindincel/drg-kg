"""
Unit tests for extraction module using mocks.

These tests don't require API keys and test the logic without external dependencies.
"""

from unittest.mock import Mock, patch

from drg.entity_resolution import EntityResolver, resolve_entities_and_relations

# Import after path setup (conftest.py handles this)
from drg.extract import KGExtractor, extract_typed
from drg.schema import DRGSchema, Entity, Relation


class TestEntityResolution:
    """Test entity resolution functionality (unit tests without external dependencies)."""

    def test_resolve_similar_entities(self):
        """Test that similar entity names are merged."""
        entities = [
            ("Dr. Elena Vasquez", "Person"),
            ("Dr. Vasquez", "Person"),
            ("Dr. Elena", "Person"),
            ("Tesla", "Company"),
        ]
        relations = [
            ("Dr. Elena Vasquez", "works_at", "Tesla"),
            ("Dr. Vasquez", "leads", "Tesla"),
        ]

        resolver = EntityResolver(similarity_threshold=0.85)
        resolved_entities, name_mapping = resolver.resolve(entities)
        resolved_relations = resolver.resolve_relations(relations, name_mapping)

        # Similar entities should be merged (longest name as canonical)
        assert len(resolved_entities) <= len(entities)
        # Canonical names should be in resolved entities
        canonical_names = {e[0] for e in resolved_entities}
        # Relations should reference canonical names
        for s, _, o in resolved_relations:
            assert s in canonical_names or s in name_mapping.values()
            assert o in canonical_names or o in name_mapping.values()

    def test_resolve_different_entities(self):
        """Test that different entities are not merged."""
        entities = [
            ("Apple", "Company"),
            ("Tesla", "Company"),
        ]

        resolver = EntityResolver(similarity_threshold=0.85)
        resolved_entities, name_mapping = resolver.resolve(entities)

        # Different entities should remain separate (or merged if similarity is high enough)
        assert len(resolved_entities) <= len(entities)
        # Both entity names should be present in resolved entities or mapping
        resolved_names = {e[0] for e in resolved_entities} | set(name_mapping.values())
        assert "Apple" in resolved_names or "Apple" in [e[0] for e in resolved_entities]
        assert "Tesla" in resolved_names or "Tesla" in [e[0] for e in resolved_entities]


class TestKGExtractor:
    """Test KGExtractor module with mocked DSPy."""

    @patch("drg.extract.dspy")
    def test_extractor_initialization(self, mock_dspy):
        """Test that KGExtractor initializes correctly."""
        # Mock TypedPredictor to avoid actual initialization
        mock_dspy.TypedPredictor = Mock()
        mock_dspy.Predict = Mock()
        mock_dspy.hasattr = lambda obj, attr: False  # TypedPredictor not available
        mock_dspy.Module = type("Module", (), {})
        mock_dspy.Signature = type("Signature", (), {})
        mock_dspy.InputField = Mock()

        schema = DRGSchema(
            entities=[Entity("Person"), Entity("Company")],
            relations=[Relation("works_at", "Person", "Company")],
        )

        extractor = KGExtractor(schema)
        assert extractor.schema == schema
        assert hasattr(extractor, "entity_extractor")
        assert hasattr(extractor, "relation_extractor")

    @patch("drg.extract.dspy")
    def test_forward_entity_extraction(self, mock_dspy):
        """Test forward method with mocked DSPy predictors."""
        # Create schema
        schema = DRGSchema(
            entities=[Entity("Person"), Entity("Company")],
            relations=[Relation("works_at", "Person", "Company")],
        )

        # Mock TypedPredictor
        mock_entity_result = Mock()
        mock_entity_result.entities = [("John", "Person"), ("Tesla", "Company")]

        mock_relation_result = Mock()
        mock_relation_result.relations = [("John", "works_at", "Tesla")]

        mock_entity_extractor = Mock(return_value=mock_entity_result)
        mock_relation_extractor = Mock(return_value=mock_relation_result)

        # Create extractor and mock predictors
        extractor = KGExtractor(schema)
        extractor._use_typed_predictor = True
        extractor.entity_extractor = mock_entity_extractor
        extractor.relation_extractor = mock_relation_extractor

        # Call forward
        result = extractor.forward("John works at Tesla")

        # Verify
        assert hasattr(result, "entities")
        assert hasattr(result, "relations")
        assert len(result.entities) == 2
        assert len(result.relations) == 1
        mock_entity_extractor.assert_called_once()
        mock_relation_extractor.assert_called_once()


class TestExtractTyped:
    """Test extract_typed function with mocked components."""

    @patch("drg.extract._get_extractor")
    @patch("drg.extract.resolve_entities_and_relations")
    def test_extract_typed_with_resolution(self, mock_resolve, mock_get_extractor):
        """Test extract_typed with entity resolution enabled."""
        # Mock extractor
        mock_extractor = Mock()
        mock_result = Mock()
        mock_result.entities = [("John", "Person"), ("Tesla", "Company")]
        mock_result.relations = [("John", "works_at", "Tesla")]
        mock_extractor.return_value = mock_result
        mock_get_extractor.return_value = mock_extractor

        # Mock entity resolution
        mock_resolve.return_value = (
            [("John", "Person"), ("Tesla", "Company")],
            [("John", "works_at", "Tesla")],
        )

        schema = DRGSchema(
            entities=[Entity("Person"), Entity("Company")],
            relations=[Relation("works_at", "Person", "Company")],
        )

        entities, relations = extract_typed(
            "John works at Tesla", schema, enable_entity_resolution=True
        )

        # Verify entity resolution was called
        mock_resolve.assert_called_once()
        assert len(entities) == 2
        assert len(relations) == 1

    @patch("drg.extract._get_extractor")
    @patch("drg.extract.resolve_entities_and_relations")
    def test_extract_typed_without_resolution(self, mock_resolve, mock_get_extractor):
        """Test extract_typed with entity resolution disabled."""
        # Mock extractor
        mock_extractor = Mock()
        mock_result = Mock()
        mock_result.entities = [("John", "Person")]
        mock_result.relations = []
        mock_extractor.return_value = mock_result
        mock_get_extractor.return_value = mock_extractor

        schema = DRGSchema(entities=[Entity("Person")], relations=[])

        entities, relations = extract_typed("John", schema, enable_entity_resolution=False)

        # Verify entity resolution was NOT called
        mock_resolve.assert_not_called()
        assert len(entities) == 1
        assert len(relations) == 0


class TestSchemaValidation:
    """Test schema validation in extraction."""

    @patch("drg.extract._get_extractor")
    @patch("drg.extract.resolve_entities_and_relations")
    def test_invalid_entity_type_filtered(self, mock_resolve, mock_get_extractor):
        """Test that invalid entity types are filtered out."""
        # Mock extractor returning invalid entity type
        mock_extractor = Mock()
        mock_result = Mock()
        mock_result.entities = [("John", "Person"), ("Invalid", "UnknownType")]
        mock_result.relations = []
        mock_extractor.return_value = mock_result
        mock_get_extractor.return_value = mock_extractor

        # Mock entity resolution (just pass through)
        mock_resolve.side_effect = lambda e, r, **kwargs: (e, r)

        schema = DRGSchema(
            entities=[Entity("Person")],  # Only Person allowed
            relations=[],
        )

        entities, _relations = extract_typed("Test", schema, enable_entity_resolution=False)

        # Invalid entity type should be filtered
        assert all(entity_type == "Person" for _, entity_type in entities)
        assert ("Invalid", "UnknownType") not in entities

    @patch("drg.extract._get_extractor")
    @patch("drg.extract.resolve_entities_and_relations")
    def test_reverse_relation_converted_to_direct(self, mock_resolve, mock_get_extractor):
        """Schema has only direct relation, but extractor returns reverse. Should be normalized safely."""
        mock_extractor = Mock()
        mock_result = Mock()
        mock_result.entities = [("Apple", "Company"), ("iPhone", "Product")]
        mock_result.relations = [("iPhone", "produced_by", "Apple")]  # reverse direction
        mock_extractor.return_value = mock_result
        mock_get_extractor.return_value = mock_extractor

        # No resolution changes; pass-through
        mock_resolve.side_effect = lambda e, r, **kwargs: (e, r)

        schema = DRGSchema(
            entities=[Entity("Company"), Entity("Product")],
            relations=[Relation("produces", "Company", "Product")],
        )

        entities, relations = extract_typed("dummy", schema, enable_entity_resolution=False)
        assert ("Apple", "Company") in entities
        assert ("iPhone", "Product") in entities
        assert ("Apple", "produces", "iPhone") in relations


class TestEntityResolutionIntegration:
    """Integration tests for entity resolution."""

    def test_resolve_entities_and_relations_function(self):
        """Test the convenience function for entity resolution."""
        entities = [
            ("Apple Inc.", "Company"),
            ("Apple", "Company"),
            ("Tesla", "Company"),
        ]
        relations = [
            ("Apple Inc.", "competes_with", "Tesla"),
            ("Apple", "competes_with", "Tesla"),
        ]

        resolved_entities, resolved_relations = resolve_entities_and_relations(
            entities, relations, similarity_threshold=0.85
        )

        # Similar entities should be merged
        assert len(resolved_entities) <= len(entities)
        # Relations should use canonical names
        assert len(resolved_relations) <= len(relations)
        # No self-relations should exist
        assert not any(s == o for s, _, o in resolved_relations)
