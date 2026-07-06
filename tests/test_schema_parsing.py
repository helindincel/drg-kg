import pytest

from drg.errors import SchemaError
from drg.schema import EnhancedDRGSchema, load_schema_from_json


def test_enhanced_schema_from_dict_parses_minimal():
    data = {
        "entity_types": [
            {"name": "Person", "description": "Individuals", "examples": ["Alice"]},
            {"name": "Company", "description": "Organizations", "examples": ["ACME"]},
        ],
        "relation_groups": [
            {
                "name": "employment",
                "description": "Employment relations",
                "relations": [
                    {
                        "name": "works_at",
                        "source": "Person",
                        "target": "Company",
                        "description": "Person works at company",
                        "detail": "Employment relationship",
                        "properties": {"since": "Start time of the relationship"},
                    }
                ],
            }
        ],
        "auto_discovery": True,
    }

    schema = EnhancedDRGSchema.from_dict(data)
    assert len(schema.entity_types) == 2
    assert len(schema.relation_groups) == 1
    assert schema.is_valid_relation("works_at", "Person", "Company")
    relation = schema.relation_groups[0].relations[0]
    assert relation.properties["since"] == "Start time of the relationship"
    assert schema.to_dict()["relation_groups"][0]["relations"][0]["properties"] == {
        "since": "Start time of the relationship"
    }


def test_enhanced_schema_from_dict_accepts_src_dst_aliases():
    data = {
        "entity_types": [
            {"name": "A", "description": "Type A"},
            {"name": "B", "description": "Type B"},
        ],
        "relation_groups": [
            {
                "name": "g",
                "description": "group",
                "relations": [{"name": "rel", "src": "A", "dst": "B"}],
            }
        ],
    }
    schema = EnhancedDRGSchema.from_dict(data)
    assert schema.is_valid_relation("rel", "A", "B")


def test_enhanced_schema_from_dict_round_trips_groups():
    data = {
        "entity_types": [
            {"name": "Person", "description": "Individuals"},
            {"name": "Company", "description": "Organizations"},
        ],
        "relation_groups": [
            {
                "name": "employment",
                "relations": [{"name": "works_at", "source": "Person", "target": "Company"}],
            }
        ],
        "entity_groups": [
            {
                "name": "actors",
                "description": "Active participants",
                "entity_types": ["Person", {"name": "Company"}],
                "examples": [{"text": "Alice at ACME"}],
            }
        ],
        "property_groups": [
            {
                "name": "audit",
                "description": "Shared audit fields",
                "properties": {"source": "Where the value came from"},
                "examples": [{"source": "doc-1"}],
            }
        ],
    }

    schema = EnhancedDRGSchema.from_dict(data)

    assert [group.name for group in schema.entity_groups] == ["actors"]
    assert schema.entity_groups[0].get_entity_type_names() == ["Person", "Company"]
    assert [group.name for group in schema.property_groups] == ["audit"]
    assert schema.to_dict()["property_groups"][0]["properties"]["source"] == (
        "Where the value came from"
    )


def test_enhanced_schema_from_dict_rejects_empty():
    with pytest.raises(SchemaError):
        EnhancedDRGSchema.from_dict({})


def test_load_schema_rejects_non_object_json_root(tmp_path):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("[]")

    with pytest.raises(SchemaError, match="root must be an object"):
        load_schema_from_json(schema_path)


def test_load_schema_rejects_malformed_legacy_entities(tmp_path):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"entities": ["Person"], "relations": []}')

    with pytest.raises(SchemaError, match="entity at index 0"):
        load_schema_from_json(schema_path)
