"""
DRG Test Suite - Unit tests using mocks (no API keys required).

These tests use mocks to test the logic without external dependencies.
Integration tests that require API keys should be in a separate file.
"""

from unittest.mock import patch

from drg.extract import extract_triples
from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode
from drg.schema import DRGSchema, Entity, Relation


def _get_test_schema() -> DRGSchema:
    """Test için standart schema oluştur."""
    return DRGSchema(
        entities=[Entity("Company"), Entity("Product"), Entity("Person")],
        relations=[
            Relation("produces", "Company", "Product", description="Şirket ürün üretir"),
            Relation(
                "founded_by", "Company", "Person", description="Şirket kişi tarafından kuruldu"
            ),
            Relation("ceo_of", "Person", "Company", description="Kişi şirketin CEO'sudur"),
        ],
    )


def _get_test_text() -> str:
    """Test için standart metin."""
    return "Apple Inc. was founded by Steve Jobs in 1976. Tim Cook is the current CEO of Apple. Apple produces the iPhone, iPad, and Mac computers."


class TestExtractTriplesWithMock:
    """Mock-based unit tests for extract_triples function."""

    @patch("drg.extract.extract_typed")
    def test_extract_triples_basic(self, mock_extract_typed):
        """Test extract_triples with mocked extract_typed."""
        # Mock extract_typed to return predefined entities and relations
        mock_entities = [
            ("Apple", "Company"),
            ("Steve Jobs", "Person"),
            ("Tim Cook", "Person"),
            ("iPhone", "Product"),
            ("iPad", "Product"),
            ("Mac computers", "Product"),
        ]
        mock_relations = [
            ("Apple", "founded_by", "Steve Jobs"),
            ("Tim Cook", "ceo_of", "Apple"),
            ("Apple", "produces", "iPhone"),
            ("Apple", "produces", "iPad"),
            ("Apple", "produces", "Mac computers"),
        ]
        mock_extract_typed.return_value = (mock_entities, mock_relations)

        schema = _get_test_schema()
        text = _get_test_text()

        # Call extract_triples
        triples = extract_triples(text, schema)
        triples = list(dict.fromkeys(triples))  # Duplicate'leri kaldır

        # Verify extract_typed was called
        mock_extract_typed.assert_called_once()
        call_args = mock_extract_typed.call_args
        assert call_args[0][0] == text  # First positional arg is text
        assert call_args[0][1] == schema  # Second positional arg is schema

        # Verify triples structure
        assert len(triples) > 0, "En az bir triple olmalı"
        assert all(len(triple) == 3 for triple in triples), (
            "Her triple (source, relation, target) formatında olmalı"
        )

        # Verify specific triples
        triple_set = set(triples)
        assert ("Apple", "founded_by", "Steve Jobs") in triple_set
        assert ("Apple", "produces", "iPhone") in triple_set

    @patch("drg.extract.extract_typed")
    def test_extract_triples_empty_result(self, mock_extract_typed):
        """Test extract_triples with empty extraction result."""
        # Mock extract_typed to return empty results
        mock_extract_typed.return_value = ([], [])

        schema = _get_test_schema()
        text = "Some text with no entities"

        triples = extract_triples(text, schema)

        # Should return empty list
        assert triples == []
        mock_extract_typed.assert_called_once()


class TestKGConstructionFromTriples:
    """Test KG construction from extracted triples (logic tests, no API calls)."""

    def test_enhanced_kg_from_triples(self):
        """Test EnhancedKG construction from triples without API calls."""
        # Simulated extraction results (as if from extract_triples)
        triples = [
            ("Apple", "founded_by", "Steve Jobs"),
            ("Tim Cook", "ceo_of", "Apple"),
            ("Apple", "produces", "iPhone"),
            ("Apple", "produces", "iPad"),
        ]

        # EnhancedKG oluştur
        enhanced_kg = EnhancedKG()

        # Entities ekle
        entity_map = {}
        for source, _relation, target in triples:
            if source not in entity_map:
                entity_map[source] = KGNode(id=source, type=None)
                enhanced_kg.add_node(entity_map[source])
            if target not in entity_map:
                entity_map[target] = KGNode(id=target, type=None)
                enhanced_kg.add_node(entity_map[target])

        # Edges ekle
        for source, relation, target in triples:
            edge = KGEdge(
                source=source,
                target=target,
                relationship_type=relation,
                relationship_detail=f"{source} {relation} {target}",
                metadata={},
            )
            enhanced_kg.add_edge(edge)

        # Assertions
        assert len(enhanced_kg.nodes) > 0, "En az bir node olmalı"
        assert len(enhanced_kg.edges) > 0, "En az bir edge olmalı"

        # Apple entity'si olmalı
        assert "Apple" in enhanced_kg.nodes, "Apple entity'si bulunmalı"

        # iPhone veya iPad gibi bir product olmalı
        product_found = any(
            "iPhone" in node_id or "iPad" in node_id for node_id in enhanced_kg.nodes.keys()
        )
        assert product_found, "En az bir product entity'si bulunmalı"


def test_schema_with_relation_descriptions():
    """Schema'daki relation description'larının doğru yüklendiğini test et."""
    schema = _get_test_schema()

    # Relation description'ları kontrol et
    produces_rel = next((r for r in schema.relations if r.name == "produces"), None)
    assert produces_rel is not None, "produces relation bulunmalı"
    assert hasattr(produces_rel, "description"), "Relation'da description field'ı olmalı"
    assert produces_rel.description == "Şirket ürün üretir", "Description doğru yüklenmeli"

    founded_by_rel = next((r for r in schema.relations if r.name == "founded_by"), None)
    assert founded_by_rel is not None
    assert founded_by_rel.description == "Şirket kişi tarafından kuruldu"


def test_enhanced_kg_structure():
    """EnhancedKG yapısının doğru çalıştığını test et."""
    enhanced_kg = EnhancedKG()

    # Node ekle
    node1 = KGNode(id="Apple", type="Company")
    node2 = KGNode(id="iPhone", type="Product")
    enhanced_kg.add_node(node1)
    enhanced_kg.add_node(node2)

    # Edge ekle
    edge = KGEdge(
        source="Apple",
        target="iPhone",
        relationship_type="produces",
        relationship_detail="Apple iPhone üretir",
        metadata={},
    )
    enhanced_kg.add_edge(edge)

    # Assertions
    assert len(enhanced_kg.nodes) == 2
    assert len(enhanced_kg.edges) == 1
    assert "Apple" in enhanced_kg.nodes
    assert "iPhone" in enhanced_kg.nodes

    # Edge kontrolü
    assert edge.source == "Apple"
    assert edge.target == "iPhone"
    assert edge.relationship_type == "produces"
    assert "iPhone" in edge.relationship_detail
