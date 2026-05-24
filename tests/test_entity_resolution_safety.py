from drg.entity_resolution import resolve_entities_and_relations


def test_entity_resolution_merges_short_alias_when_unambiguous():
    entities = [("Dr. Elena Vasquez", "Person"), ("Elena", "Person")]
    relations = [("Elena", "works_at", "ACME")]
    # "ACME" not in entities; it's fine for relation resolution mapping test.

    resolved_entities, resolved_relations = resolve_entities_and_relations(
        entities,
        relations,
        similarity_threshold=0.65,
        adaptive_threshold=True,
        use_embedding=False,
    )
    assert ("Dr. Elena Vasquez", "Person") in resolved_entities
    assert ("Elena", "Person") not in resolved_entities
    assert ("Dr. Elena Vasquez", "works_at", "ACME") in resolved_relations


def test_entity_resolution_does_not_merge_two_different_single_token_names():
    entities = [("Elena", "Person"), ("Selena", "Person")]
    relations = []
    resolved_entities, _ = resolve_entities_and_relations(
        entities,
        relations,
        similarity_threshold=0.65,
        adaptive_threshold=True,
        use_embedding=False,
    )
    assert ("Elena", "Person") in resolved_entities
    assert ("Selena", "Person") in resolved_entities


def test_entity_resolution_abstains_when_short_alias_is_ambiguous():
    entities = [
        ("Dr. Elena Vasquez", "Person"),
        ("Elena Gilbert", "Person"),
        ("Elena", "Person"),
    ]
    relations = [("Elena", "spoke_about", "iPhone")]
    resolved_entities, resolved_relations = resolve_entities_and_relations(
        entities,
        relations,
        similarity_threshold=0.65,
        adaptive_threshold=True,
        use_embedding=False,
    )
    # Should not guess which Elena; keep "Elena" as separate node.
    assert ("Elena", "Person") in resolved_entities
    assert ("Elena", "spoke_about", "iPhone") in resolved_relations
