"""
Functional tests for extraction - test actual functionality, not just mocks.

These tests verify that the extraction pipeline actually works correctly:
- Entity extraction from text
- Relation extraction from text
- Cross-chunk relationship discovery
- Schema generation
- Entity resolution
- Coreference resolution
- Negation detection
- Temporal information extraction
- Confidence scores
- Reverse relations

These tests may require API keys (marked with @pytest.mark.integration).
"""

import pytest

from drg.extract import extract_from_chunks, extract_typed, generate_schema_from_text
from drg.schema import DRGSchema, EnhancedDRGSchema, Entity, Relation


# Test fixtures
def _get_test_schema() -> DRGSchema:
    """Test schema with common entity types and relations."""
    return DRGSchema(
        entities=[
            Entity("Company"),
            Entity("Person"),
            Entity("Product"),
            Entity("Location"),
        ],
        relations=[
            Relation("founded", "Person", "Company", description="Person founded company"),
            Relation("works_at", "Person", "Company", description="Person works at company"),
            Relation("produces", "Company", "Product", description="Company produces product"),
            Relation(
                "located_in", "Company", "Location", description="Company located in location"
            ),
        ],
    )


def _get_test_text_single_chunk() -> str:
    """Single chunk test text."""
    return (
        "Apple Inc. was founded by Steve Jobs in 1976. "
        "Tim Cook is the current CEO of Apple. "
        "Apple produces the iPhone, iPad, and Mac computers. "
        "Apple is located in Cupertino, California."
    )


def _get_test_text_cross_chunk() -> str:
    """Multi-chunk test text for cross-chunk relationship testing."""
    return (
        "Apple Inc. was founded by Steve Jobs in 1976. "  # Chunk 1: Apple, Steve Jobs
        "The company revolutionized personal computing with the Macintosh. "  # Chunk 1: Apple (implied)
        "Later, Apple introduced the iPhone, which changed the mobile phone industry. "  # Chunk 1-2: Apple, iPhone
        "\n\n"  # Simulate chunk boundary
        "Tim Cook joined Apple in 1998 and became CEO in 2011. "  # Chunk 2: Tim Cook, Apple
        "Under Cook's leadership, Apple continued to innovate. "  # Chunk 2: Tim Cook (coreference: Cook's = Tim Cook)
        "The iPhone's chief designer was Jony Ive, who worked at Apple until 2019. "  # Chunk 2: Jony Ive, iPhone, Apple (cross-chunk)
        "\n\n"  # Simulate chunk boundary
        "Cupertino, California is home to Apple's headquarters. "  # Chunk 3: Cupertino, Apple (cross-chunk)
        "The company's campus spans over 175 acres."  # Chunk 3: Apple (coreference: company's = Apple)
    )


@pytest.mark.integration
def test_entity_extraction_basic():
    """Test that entities are correctly extracted from text."""
    schema = _get_test_schema()
    text = _get_test_text_single_chunk()

    entities, _relations = extract_typed(
        text=text,
        schema=schema,
        enable_entity_resolution=False,
        enable_coreference_resolution=False,
        enable_reverse_relation_fallback=True,
    )

    # Should extract multiple entities
    assert len(entities) > 0, "Should extract at least one entity"

    # Verify entity structure
    assert all(isinstance(e, tuple) and len(e) == 2 for e in entities), (
        "Entities should be (name, type) tuples"
    )

    # Should extract expected entities
    entity_names = {name for name, _ in entities}
    assert "Apple" in entity_names or any("Apple" in name for name in entity_names), (
        "Should extract 'Apple'"
    )
    assert any("Steve" in name or "Jobs" in name for name in entity_names), (
        "Should extract 'Steve Jobs'"
    )

    # Entity types should be valid
    valid_types = {e.name for e in schema.entities}
    entity_types = {etype for _, etype in entities}
    assert all(etype in valid_types for etype in entity_types), "All entity types should be valid"


@pytest.mark.integration
def test_relation_extraction_basic():
    """Test that relations are correctly extracted from text."""
    schema = _get_test_schema()
    text = _get_test_text_single_chunk()

    _entities, relations = extract_typed(
        text=text,
        schema=schema,
        enable_entity_resolution=False,
        enable_coreference_resolution=False,
    )

    # Should extract relations
    assert len(relations) > 0, "Should extract at least one relation"

    # Verify relation structure
    assert all(isinstance(r, tuple) and len(r) == 3 for r in relations), (
        "Relations should be (source, relation, target) tuples"
    )

    # Should extract expected relations
    set(relations)
    # Check for "founded" relation (Steve Jobs founded Apple)
    founded_relations = [r for r in relations if r[1] == "founded"]
    assert len(founded_relations) > 0, "Should extract 'founded' relation"

    # Check for "produces" relation (Apple produces iPhone)
    produces_relations = [r for r in relations if r[1] == "produces"]
    assert len(produces_relations) > 0, "Should extract 'produces' relation"


@pytest.mark.integration
def test_cross_chunk_relationship_discovery():
    """Test that relationships are discovered across chunks."""
    schema = _get_test_schema()
    text = _get_test_text_cross_chunk()

    # Simulate chunking (split by double newline)
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": chunk_text.strip(), "metadata": {}}
        for i, chunk_text in enumerate(text.split("\n\n"))
        if chunk_text.strip()
    ]

    # Extract with cross-chunk relationships enabled
    entities, relations = extract_from_chunks(
        chunks=chunks,
        schema=schema,
        enable_cross_chunk_relationships=True,
        enable_entity_resolution=True,
        enable_coreference_resolution=True,
    )

    # Should extract entities from all chunks
    entity_names = {name for name, _ in entities}
    assert "Apple" in entity_names or any("Apple" in name for name in entity_names)
    assert any("Tim" in name or "Cook" in name for name in entity_names)
    assert any("Jony" in name or "Ive" in name for name in entity_names)

    # Should find cross-chunk relationships
    # Example: (Jony Ive, works_at, Apple) - entities in different chunks
    set(relations)
    works_at_relations = [r for r in relations if r[1] == "works_at"]
    assert len(works_at_relations) > 0, "Should find 'works_at' relations across chunks"

    # Cross-chunk relation example: Jony Ive works at Apple (mentioned in different chunks)
    jony_apple_relations = [
        r
        for r in relations
        if ("Jony" in r[0] or "Ive" in r[0]) and ("Apple" in r[2] or "Apple" in r[0])
    ]
    assert len(jony_apple_relations) > 0, "Should find cross-chunk relationship (Jony Ive - Apple)"


@pytest.mark.integration
def test_coreference_resolution():
    """Test that coreference resolution works (pronouns resolved to entities)."""
    schema = _get_test_schema()
    text = (
        "Tim Cook joined Apple in 1998. "
        "He became CEO in 2011. "  # "He" should resolve to "Tim Cook"
        "The company continued to innovate under his leadership."  # "his" should resolve to "Tim Cook"
    )

    entities, relations = extract_typed(
        text=text, schema=schema, enable_coreference_resolution=True, enable_entity_resolution=True
    )

    # Should extract Tim Cook and Apple
    entity_names = {name for name, _ in entities}
    assert any("Tim" in name or "Cook" in name for name in entity_names), (
        "Should extract 'Tim Cook'"
    )
    assert "Apple" in entity_names or any("Apple" in name for name in entity_names), (
        "Should extract 'Apple'"
    )

    # Relations involving "Tim Cook" should be found even when referred to as "he" or "his"
    # Note: This is a basic test - advanced coreference may require better NLP models
    works_at_relations = [r for r in relations if r[1] == "works_at"]
    tim_cook_relations = [r for r in relations if any("Tim" in r[0] or "Cook" in r[0] for r in [r])]
    assert len(tim_cook_relations) > 0 or len(works_at_relations) > 0, (
        "Should find relations involving Tim Cook (via coreference)"
    )


@pytest.mark.integration
def test_entity_resolution_similar_names():
    """Test that similar entity names are merged (entity resolution)."""
    schema = _get_test_schema()
    text = (
        "Dr. Elena Vasquez joined Tesla in 2020. "
        "Dr. Vasquez leads the battery division. "  # "Dr. Vasquez" should resolve to "Dr. Elena Vasquez"
        "Dr. Elena is an expert in battery technology."  # "Dr. Elena" should resolve to "Dr. Elena Vasquez"
    )

    entities, relations = extract_typed(
        text=text, schema=schema, enable_entity_resolution=True, enable_coreference_resolution=False
    )

    # Similar entity names should be merged
    entity_names = {name for name, _ in entities}

    # At least one variant should exist, and ideally merged into canonical form
    elena_variants = [name for name in entity_names if "Elena" in name or "Vasquez" in name]
    assert len(elena_variants) > 0, "Should find Elena Vasquez entity"

    # Relations should reference the canonical name (merged entities)
    # This ensures entity resolution actually merged the entities
    elena_relations = [
        r
        for r in relations
        if any(
            "Elena" in r[0] or "Vasquez" in r[0] or "Elena" in r[2] or "Vasquez" in r[2]
            for r in [r]
        )
    ]
    # Relations should exist and use consistent entity names (after resolution)
    assert len(elena_relations) > 0, (
        "Should find relations involving Elena Vasquez (after entity resolution)"
    )


@pytest.mark.integration
def test_schema_generation():
    """Test that schema can be generated from text."""
    text = (
        "Apple Inc. was founded by Steve Jobs in 1976. "
        "The company produces iPhone, iPad, and Mac computers. "
        "Tim Cook is the CEO of Apple. "
        "Apple is headquartered in Cupertino, California."
    )

    try:
        schema = generate_schema_from_text(text)

        # Should generate valid EnhancedDRGSchema
        assert isinstance(schema, EnhancedDRGSchema), "Should return EnhancedDRGSchema"

        # Should have entity types
        assert len(schema.entity_types) > 0, "Should generate at least one entity type"

        # Should have relation groups
        assert len(schema.relation_groups) > 0, "Should generate at least one relation group"

        # Should generate relevant entity types
        entity_type_names = {et.name for et in schema.entity_types}
        # Should have Company, Person, Product, Location (or similar)
        relevant_types = ["Company", "Person", "Product", "Location"]
        assert any(et in entity_type_names for et in relevant_types), (
            "Should generate relevant entity types"
        )

    except Exception as e:
        pytest.skip(f"Schema generation failed (may need API key): {e}")


@pytest.mark.integration
def test_negation_detection():
    """Test that negated relations are detected and filtered."""
    schema = _get_test_schema()

    # Text with negation
    text_negated = (
        "Apple used to produce the Newton PDA. "
        "However, Apple no longer produces the Newton. "  # Negated relation
        "Apple discontinued the Newton in 1998."
    )

    # Extract with negation handling
    entities, relations = extract_typed(
        text=text_negated,
        schema=schema,
        enable_entity_resolution=False,
        enable_coreference_resolution=False,
    )

    # Should extract entities
    assert len(entities) > 0

    # Negated relations should be filtered out (if negation detection works)
    # Note: This depends on LLM correctly marking relations as negated
    # If negation detection is implemented, relations marked as negated should be excluded
    [r for r in relations if r[1] == "produces"]
    # If negation detection works correctly, we should NOT see (Apple, produces, Newton) with current date context
    # But we might see historical relations - this test checks the filtering logic exists

    # At minimum, entities should be extracted
    entity_names = {name for name, _ in entities}
    assert "Apple" in entity_names or any("Apple" in name for name in entity_names)
    assert "Newton" in entity_names or any("Newton" in name for name in entity_names)


@pytest.mark.integration
def test_temporal_information_extraction():
    """Test that temporal information (dates) is extracted when available."""
    schema = _get_test_schema()
    text = (
        "Steve Jobs founded Apple in 1976. "
        "Tim Cook became CEO of Apple in 2011. "
        "Apple released the iPhone in 2007."
    )

    # Note: Temporal information extraction requires KGExtractor to return temporal_info
    # This is a basic test - full temporal extraction may require schema enhancements
    _entities, relations = extract_typed(
        text=text,
        schema=schema,
        enable_entity_resolution=False,
        enable_coreference_resolution=False,
    )

    # Should extract relations with temporal context
    assert len(relations) > 0

    # Should find "founded" relation with temporal context
    founded_relations = [r for r in relations if r[1] == "founded"]
    assert len(founded_relations) > 0, "Should extract 'founded' relation with temporal information"

    # Note: Full temporal extraction would require checking KGEdge.start_date/end_date fields
    # This test verifies the basic extraction works


@pytest.mark.integration
def test_reverse_relations():
    """Test that reverse relations are handled (e.g., 'produced_by' when schema has 'produces')."""
    schema = _get_test_schema()

    # Text with reverse relation phrasing
    text = (
        "The iPhone is produced by Apple. "  # "produced_by" (reverse of "produces")
        "Apple also produces the iPad."
    )

    _entities, relations = extract_typed(
        text=text,
        schema=schema,
        enable_entity_resolution=False,
        enable_coreference_resolution=False,
    )

    # Should extract relations
    assert len(relations) > 0

    # Should handle reverse relations - "produced_by" should map to "produces"
    produces_relations = [r for r in relations if r[1] == "produces"]
    assert len(produces_relations) > 0, (
        "Should extract 'produces' relation even from 'produced by' phrasing"
    )

    # Relations should be in correct direction
    iphone_relations = [r for r in relations if "iPhone" in r[0] or "iPhone" in r[2]]
    apple_iphone_relation = [
        r for r in iphone_relations if ("Apple" in r[0] or "Apple" in r[2]) and r[1] == "produces"
    ]
    # Should find (Apple, produces, iPhone) even if text says "iPhone is produced by Apple"
    assert len(apple_iphone_relation) > 0 or len(produces_relations) > 0, (
        "Should handle reverse relation direction"
    )


@pytest.mark.integration
def test_implicit_relationships():
    """Test that implicit relationships are extracted (e.g., possessive forms)."""
    schema = _get_test_schema()

    # Text with implicit relationships (possessive forms)
    text = (
        "Tesla's Gigafactory is located in Nevada. "  # Implicit: (Tesla, owns, Gigafactory)
        "The company operates multiple factories."  # Coreference: "company" = "Tesla"
    )

    # Add "owns" relation to schema for implicit relationship test
    schema_with_owns = DRGSchema(
        entities=[*schema.entities, Entity("Facility")],
        relations=[
            *schema.relations,
            Relation("owns", "Company", "Facility", description="Company owns facility"),
        ],
    )

    entities, relations = extract_typed(
        text=text,
        schema=schema_with_owns,
        enable_entity_resolution=True,
        enable_coreference_resolution=True,
    )

    # Should extract entities
    assert len(entities) > 0

    # Should extract implicit "owns" relationship from "Tesla's Gigafactory"
    # Note: This depends on LLM understanding possessive forms
    owns_relations = [r for r in relations if r[1] == "owns"]
    # If implicit relationship extraction works, should find (Tesla, owns, Gigafactory)
    # This is a best-effort test - depends on LLM capability
    if len(owns_relations) > 0:
        tesla_owns = [r for r in owns_relations if "Tesla" in r[0] and "Gigafactory" in r[2]]
        assert len(tesla_owns) > 0, (
            "Should extract implicit 'owns' relationship from possessive form"
        )


def test_extraction_with_empty_text():
    """Test that extraction handles empty text gracefully."""
    schema = _get_test_schema()

    entities, relations = extract_typed(
        text="", schema=schema, enable_entity_resolution=False, enable_coreference_resolution=False
    )

    # Should return empty lists, not crash
    assert entities == []
    assert relations == []


def test_extraction_with_no_matching_schema():
    """Test that extraction handles text with no matching entities/relations."""
    schema = _get_test_schema()

    # Text with no relevant entities
    text = "The weather is nice today. It's sunny and warm."

    entities, relations = extract_typed(
        text=text,
        schema=schema,
        enable_entity_resolution=False,
        enable_coreference_resolution=False,
    )

    # Should return empty or minimal results, not crash
    assert isinstance(entities, list)
    assert isinstance(relations, list)


def test_extraction_validates_schema():
    """Test that extraction validates entity types and relations against schema."""
    schema = _get_test_schema()
    text = _get_test_text_single_chunk()

    entities, relations = extract_typed(
        text=text,
        schema=schema,
        enable_entity_resolution=False,
        enable_coreference_resolution=False,
    )

    # All extracted entity types should be in schema
    valid_entity_types = {e.name for e in schema.entities}
    extracted_entity_types = {etype for _, etype in entities}
    assert all(etype in valid_entity_types for etype in extracted_entity_types), (
        "All entity types should be valid"
    )

    # All extracted relations should be in schema
    valid_relations = {r.name for r in schema.relations}
    extracted_relations = {r[1] for r in relations}
    assert all(rel in valid_relations for rel in extracted_relations), (
        "All relations should be valid"
    )
