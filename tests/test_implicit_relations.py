from drg.extract import _infer_implicit_relations
from drg.schema import DRGSchema, Entity, Relation


def test_infer_implicit_possessive_english_schema_gated():
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Facility")],
        relations=[Relation("owns", "Company", "Facility")],
    )
    text = "Tesla's Gigafactory is in Nevada."
    entities = [("Tesla", "Company"), ("Gigafactory", "Facility")]

    inferred = _infer_implicit_relations(text=text, entities=entities, schema=schema)
    assert ("Tesla", "owns", "Gigafactory") in inferred


def test_infer_implicit_possessive_turkish_schema_gated():
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Facility")],
        relations=[Relation("owns", "Company", "Facility")],
    )
    text = "Tesla'nın Gigafactory'si Nevada'da."
    entities = [("Tesla", "Company"), ("Gigafactory", "Facility")]

    inferred = _infer_implicit_relations(text=text, entities=entities, schema=schema)
    assert ("Tesla", "owns", "Gigafactory") in inferred


def test_infer_implicit_does_not_emit_if_relation_not_in_schema():
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Facility")],
        relations=[],  # owns not allowed
    )
    text = "Tesla's Gigafactory is in Nevada."
    entities = [("Tesla", "Company"), ("Gigafactory", "Facility")]

    inferred = _infer_implicit_relations(text=text, entities=entities, schema=schema)
    assert inferred == []


def test_infer_two_hop_owner_operates_in_location_schema_gated():
    # Tesla'nın Gigafactory'si Nevada'da. -> owns + (facility located_in location) -> operates_in
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Facility"), Entity("Location")],
        relations=[
            Relation("owns", "Company", "Facility"),
            Relation("located_in", "Facility", "Location"),
            Relation("operates_in", "Company", "Location"),
        ],
    )
    text = "Tesla'nın Gigafactory'si Nevada'da. Şirket orada 7.000 kişi çalıştırıyor."
    entities = [("Tesla", "Company"), ("Gigafactory", "Facility"), ("Nevada", "Location")]

    existing_triples = [("Gigafactory", "located_in", "Nevada")]
    inferred = _infer_implicit_relations(
        text=text, entities=entities, schema=schema, existing_triples=existing_triples
    )
    assert ("Tesla", "owns", "Gigafactory") in inferred
    assert ("Tesla", "operates_in", "Nevada") in inferred


def test_infer_two_hop_does_not_overgenerate_without_operation_cue():
    schema = DRGSchema(
        entities=[Entity("Person"), Entity("Book"), Entity("Location")],
        relations=[
            Relation("owns", "Person", "Book"),
            Relation("located_in", "Book", "Location"),
            Relation("operates_in", "Person", "Location"),
        ],
    )
    text = "John's book is in the library."  # no operate/run/employ/work cue
    entities = [("John", "Person"), ("book", "Book"), ("library", "Location")]
    existing_triples = [("book", "located_in", "library")]

    inferred = _infer_implicit_relations(
        text=text, entities=entities, schema=schema, existing_triples=existing_triples
    )
    assert ("John", "owns", "book") in inferred
    assert ("John", "operates_in", "library") not in inferred
