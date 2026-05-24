"""
Integration tests for DRG (require API keys).

These tests actually call external APIs and require API keys to be set.
They are separated from unit tests to make it clear they require external dependencies.

To run integration tests:
    pytest tests/test_integration.py -v

To skip integration tests:
    pytest -v  # Only runs unit tests (test_basic.py, test_extract_mock.py)
"""

import os

import pytest

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


def _check_api_key_and_set_model(provider: str) -> str:
    """
    Provider'a göre API key kontrolü yap ve model ayarla.

    Args:
        provider: "openai", "gemini", "openrouter", "anthropic"

    Returns:
        Model adı

    Raises:
        pytest.skip: API key yoksa test'i atla
    """
    model_map = {
        "openai": "openai/gpt-4o-mini",
        "gemini": "gemini/gemini-2.0-flash-exp",
        "openrouter": "openrouter/anthropic/claude-3-haiku",
        "anthropic": "anthropic/claude-3-haiku",
    }

    api_key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    model = model_map.get(provider)
    api_key_env = api_key_map.get(provider)

    if not model or not api_key_env:
        pytest.skip(f"Unknown provider: {provider}")

    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(
            f"No {api_key_env} found. Set {api_key_env} environment variable to run this test."
        )

    # Model'i environment variable'a set et
    os.environ["DRG_MODEL"] = model

    return model


@pytest.mark.integration
def test_extract_entities_and_relations_with_openai():
    """OpenAI model ile entity ve relation extraction integration testi."""
    _check_api_key_and_set_model("openai")

    schema = _get_test_schema()
    text = _get_test_text()

    triples = extract_triples(text, schema)
    triples = list(dict.fromkeys(triples))  # Duplicate'leri kaldır

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


@pytest.mark.integration
def test_extract_entities_and_relations_with_gemini():
    """Gemini model ile entity ve relation extraction integration testi."""
    _check_api_key_and_set_model("gemini")

    schema = _get_test_schema()
    text = _get_test_text()

    triples = extract_triples(text, schema)
    triples = list(dict.fromkeys(triples))

    # EnhancedKG oluştur
    enhanced_kg = EnhancedKG()

    # Entities ve edges ekle
    entity_map = {}
    for source, relation, target in triples:
        if source not in entity_map:
            entity_map[source] = KGNode(id=source, type=None)
            enhanced_kg.add_node(entity_map[source])
        if target not in entity_map:
            entity_map[target] = KGNode(id=target, type=None)
            enhanced_kg.add_node(entity_map[target])

        edge = KGEdge(
            source=source,
            target=target,
            relationship_type=relation,
            relationship_detail=f"{source} {relation} {target}",
            metadata={},
        )
        enhanced_kg.add_edge(edge)

    # Assertions
    assert len(enhanced_kg.nodes) > 0
    assert len(enhanced_kg.edges) > 0
    assert "Apple" in enhanced_kg.nodes
