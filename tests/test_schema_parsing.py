import pytest

from drg.schema import EnhancedDRGSchema


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


def test_enhanced_schema_from_dict_rejects_empty():
    with pytest.raises(ValueError):
        EnhancedDRGSchema.from_dict({})
