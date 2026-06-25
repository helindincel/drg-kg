"""
Declarative schema definitions for DRG - Signature-like structure.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import SchemaError


@dataclass(frozen=True)
class Entity:
    """Legacy Entity class for backward compatibility."""

    name: str


@dataclass(frozen=True)
class Relation:
    """Relation definition between entity types."""

    name: str
    src: str  # Source entity type name
    dst: str  # Destination entity type name
    description: str = ""  # Why this relation exists (relationship type description)
    detail: str = ""  # One-sentence evidence for the connection


@dataclass
class EntityType:
    """Entity type definition with metadata."""

    name: str
    description: str
    examples: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate entity type."""
        if not self.name:
            raise SchemaError("EntityType name cannot be empty")
        if not self.description:
            raise SchemaError("EntityType description cannot be empty")


@dataclass
class EntityGroup:
    """Group of related entity types."""

    name: str
    description: str
    entity_types: list[EntityType]
    examples: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate entity group."""
        if not self.name:
            raise SchemaError("EntityGroup name cannot be empty")
        if not self.entity_types:
            raise SchemaError("EntityGroup must contain at least one EntityType")

    def get_entity_type_names(self) -> list[str]:
        """Get list of entity type names in this group."""
        return [et.name for et in self.entity_types]


@dataclass
class PropertyGroup:
    """Group of properties that can be shared across entity types."""

    name: str
    description: str
    properties: dict[str, Any]
    examples: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate property group."""
        if not self.name:
            raise SchemaError("PropertyGroup name cannot be empty")
        if not self.properties:
            raise SchemaError("PropertyGroup must contain at least one property")


@dataclass
class RelationGroup:
    """Group of related relations with semantic meaning."""

    name: str
    description: str = ""
    relations: list[Relation] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate relation group."""
        if not self.name:
            raise SchemaError("RelationGroup name cannot be empty")
        if not self.relations:
            raise SchemaError("RelationGroup must contain at least one Relation")

    def get_relation_triples(self) -> list[tuple[str, str, str]]:
        """Get list of (relation_name, src, dst) tuples."""
        return [(r.name, r.src, r.dst) for r in self.relations]


class DRGSchema:
    """Legacy schema class for backward compatibility."""

    def __init__(self, entities: list[Entity], relations: list[Relation]):
        self.entities = entities
        self.relations = relations
        self._validate()

    def _validate(self):
        entity_names = {e.name for e in self.entities}
        for r in self.relations:
            if r.src not in entity_names or r.dst not in entity_names:
                raise SchemaError(f"Relation {r.name} refers to unknown entity: {r.src}->{r.dst}")

    def relation_types(self) -> list[tuple[str, str, str]]:
        return [(r.name, r.src, r.dst) for r in self.relations]


class EnhancedDRGSchema:
    """Enhanced declarative schema with grouping capabilities."""

    def __init__(
        self,
        entity_types: list[EntityType],
        relation_groups: list[RelationGroup],
        entity_groups: list[EntityGroup] | None = None,
        property_groups: list[PropertyGroup] | None = None,
        auto_discovery: bool = False,
    ):
        self.entity_types = entity_types
        self.relation_groups = relation_groups
        self.entity_groups = entity_groups or []
        self.property_groups = property_groups or []
        self.auto_discovery = auto_discovery

        self._validate()
        self._build_indexes()

    def _validate(self):
        """Validate schema consistency."""
        # Check entity type names are unique
        entity_names = {et.name for et in self.entity_types}
        if len(entity_names) != len(self.entity_types):
            raise SchemaError("EntityType names must be unique")

        # Check relation groups reference valid entity types
        all_relation_triples = []
        for rg in self.relation_groups:
            for rel in rg.relations:
                if rel.src not in entity_names:
                    raise SchemaError(
                        f"Relation {rel.name} in group '{rg.name}' references unknown entity type: {rel.src}"
                    )
                if rel.dst not in entity_names:
                    raise SchemaError(
                        f"Relation {rel.name} in group '{rg.name}' references unknown entity type: {rel.dst}"
                    )
                all_relation_triples.append((rel.name, rel.src, rel.dst))

        # Check entity groups reference valid entity types
        entity_type_map = {et.name: et for et in self.entity_types}
        for eg in self.entity_groups:
            for et in eg.entity_types:
                if et.name not in entity_type_map:
                    raise SchemaError(
                        f"EntityGroup '{eg.name}' references unknown EntityType: {et.name}"
                    )

    def _build_indexes(self):
        """Build internal indexes for fast lookup."""
        # Entity type name -> EntityType
        self._entity_type_map = {et.name: et for et in self.entity_types}

        # Relation name -> List of (src, dst) pairs
        self._relation_map: dict[str, list[tuple[str, str]]] = {}
        for rg in self.relation_groups:
            for rel in rg.relations:
                if rel.name not in self._relation_map:
                    self._relation_map[rel.name] = []
                self._relation_map[rel.name].append((rel.src, rel.dst))

        # Entity type -> List of relations it can participate in
        self._entity_relations: dict[str, list[tuple[str, str, str]]] = {}
        for rg in self.relation_groups:
            for rel in rg.relations:
                if rel.src not in self._entity_relations:
                    self._entity_relations[rel.src] = []
                if rel.dst not in self._entity_relations:
                    self._entity_relations[rel.dst] = []
                self._entity_relations[rel.src].append((rel.name, rel.src, rel.dst))
                self._entity_relations[rel.dst].append((rel.name, rel.src, rel.dst))

    def get_entity_type(self, name: str) -> EntityType | None:
        """Get entity type by name."""
        return self._entity_type_map.get(name)

    def get_all_relations(self) -> list[Relation]:
        """Get all relations from all relation groups."""
        relations = []
        for rg in self.relation_groups:
            relations.extend(rg.relations)
        return relations

    def get_relations_for_entity_type(self, entity_type_name: str) -> list[tuple[str, str, str]]:
        """Get all relations that involve a given entity type."""
        return self._entity_relations.get(entity_type_name, [])

    def is_valid_relation(self, relation_name: str, src_type: str, dst_type: str) -> bool:
        """Check if a relation is valid for given entity types."""
        if relation_name not in self._relation_map:
            return False
        return (src_type, dst_type) in self._relation_map[relation_name]

    def to_legacy_schema(self) -> DRGSchema:
        """Convert to legacy DRGSchema for backward compatibility."""
        entities = [Entity(et.name) for et in self.entity_types]
        relations = self.get_all_relations()
        return DRGSchema(entities=entities, relations=relations)

    def to_dict(self) -> dict[str, Any]:
        """Convert EnhancedDRGSchema to dictionary format (for JSON serialization)."""
        return {
            "entity_types": [
                {
                    "name": et.name,
                    "description": et.description,
                    "examples": et.examples,
                    "properties": et.properties,
                }
                for et in self.entity_types
            ],
            "relation_groups": [
                {
                    "name": rg.name,
                    "description": rg.description,
                    "relations": [
                        {
                            "name": r.name,
                            "source": r.src,
                            "target": r.dst,
                            "description": r.description,
                            "detail": r.detail,
                        }
                        for r in rg.relations
                    ],
                    "examples": rg.examples,
                }
                for rg in self.relation_groups
            ],
            "entity_groups": [
                {
                    "name": eg.name,
                    "description": eg.description,
                    "entity_types": eg.get_entity_type_names(),
                    "examples": eg.examples,
                }
                for eg in self.entity_groups
            ]
            if self.entity_groups
            else [],
            "property_groups": [
                {
                    "name": pg.name,
                    "description": pg.description,
                    "properties": pg.properties,
                    "examples": pg.examples,
                }
                for pg in self.property_groups
            ]
            if self.property_groups
            else [],
            "auto_discovery": self.auto_discovery,
        }

    @classmethod
    def from_dict(cls, schema_data: dict[str, Any]) -> "EnhancedDRGSchema":
        """Load EnhancedDRGSchema from a dictionary (JSON-compatible).

        Accepts both canonical keys and common aliases:
        - relation endpoints may be provided as source/target or src/dst
        """
        if not isinstance(schema_data, dict):
            raise SchemaError(f"Schema must be a dict, got {type(schema_data).__name__}")

        raw_entity_types = schema_data.get("entity_types", [])
        if not isinstance(raw_entity_types, list) or not raw_entity_types:
            raise SchemaError("EnhancedDRGSchema requires non-empty 'entity_types' list")

        entity_types: list[EntityType] = []
        for et in raw_entity_types:
            if not isinstance(et, dict):
                continue
            name = et.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            description = et.get("description", "")
            if not isinstance(description, str) or not description.strip():
                # EntityType enforces non-empty description
                description = "Auto-generated entity type"
            examples = et.get("examples", [])
            if not isinstance(examples, list):
                examples = []
            properties = et.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}
            entity_types.append(
                EntityType(
                    name=name.strip(),
                    description=description.strip(),
                    examples=[str(x) for x in examples][:20],
                    properties=properties,
                )
            )

        if not entity_types:
            raise SchemaError("EnhancedDRGSchema parsing produced empty entity_types")

        raw_relation_groups = schema_data.get("relation_groups", [])
        if not isinstance(raw_relation_groups, list) or not raw_relation_groups:
            raise SchemaError("EnhancedDRGSchema requires non-empty 'relation_groups' list")

        relation_groups: list[RelationGroup] = []
        for rg in raw_relation_groups:
            if not isinstance(rg, dict):
                continue
            rg_name = rg.get("name")
            if not isinstance(rg_name, str) or not rg_name.strip():
                continue
            rg_desc = rg.get("description", "")
            if not isinstance(rg_desc, str) or not rg_desc.strip():
                rg_desc = "Auto-generated relation group"

            raw_relations = rg.get("relations", [])
            if not isinstance(raw_relations, list) or not raw_relations:
                continue

            relations: list[Relation] = []
            for r in raw_relations:
                if not isinstance(r, dict):
                    continue
                rel_name = r.get("name") or r.get("relation")
                src = r.get("source") or r.get("src")
                dst = r.get("target") or r.get("dst")
                if not rel_name or not src or not dst:
                    continue
                rel_desc = r.get("description", "")
                rel_detail = r.get("detail", "")
                if not isinstance(rel_desc, str):
                    rel_desc = ""
                if not isinstance(rel_detail, str):
                    rel_detail = ""
                relations.append(
                    Relation(
                        name=str(rel_name).strip(),
                        src=str(src).strip(),
                        dst=str(dst).strip(),
                        description=rel_desc,
                        detail=rel_detail,
                    )
                )

            if not relations:
                continue

            examples = rg.get("examples", [])
            if not isinstance(examples, list):
                examples = []

            relation_groups.append(
                RelationGroup(
                    name=rg_name.strip(),
                    description=rg_desc.strip(),
                    relations=relations,
                    examples=examples,
                )
            )

        if not relation_groups:
            raise SchemaError("EnhancedDRGSchema parsing produced empty relation_groups")

        return cls(
            entity_types=entity_types,
            relation_groups=relation_groups,
            auto_discovery=bool(schema_data.get("auto_discovery", False)),
        )

    def get_schema_summary(self) -> dict[str, Any]:
        """Get a summary of the schema for display/debugging."""
        return {
            "entity_types": [
                {
                    "name": et.name,
                    "description": et.description,
                    "examples": et.examples,
                    "properties": et.properties,
                }
                for et in self.entity_types
            ],
            "relation_groups": [
                {
                    "name": rg.name,
                    "description": rg.description,
                    "relations": [(r.name, r.src, r.dst) for r in rg.relations],
                    "examples": rg.examples,
                }
                for rg in self.relation_groups
            ],
            "entity_groups": [
                {
                    "name": eg.name,
                    "description": eg.description,
                    "entity_types": eg.get_entity_type_names(),
                    "examples": eg.examples,
                }
                for eg in self.entity_groups
            ],
            "property_groups": [
                {
                    "name": pg.name,
                    "description": pg.description,
                    "properties": pg.properties,
                    "examples": pg.examples,
                }
                for pg in self.property_groups
            ],
            "auto_discovery": self.auto_discovery,
        }


def load_schema_from_json(schema_path: str | Path) -> DRGSchema | EnhancedDRGSchema:
    """Load schema from JSON file (supports both Enhanced and legacy formats).

    Args:
        schema_path: Path to JSON schema file

    Returns:
        DRGSchema or EnhancedDRGSchema instance

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema JSON is invalid
    """
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    try:
        with open(path, encoding="utf-8") as f:
            schema_data = json.load(f)
    except json.JSONDecodeError as e:
        raise SchemaError(f"Invalid JSON in schema file: {e}") from e

    if not isinstance(schema_data, dict):
        raise SchemaError(f"Schema JSON root must be an object, got {type(schema_data).__name__}")

    # Check whether this is an enhanced schema (entity_types key present)
    if "entity_types" in schema_data:
        return EnhancedDRGSchema.from_dict(schema_data)
    else:
        # Legacy format
        raw_entities = schema_data.get("entities", [])
        raw_relations = schema_data.get("relations", [])
        if not isinstance(raw_entities, list):
            raise SchemaError("Legacy schema field 'entities' must be a list")
        if not isinstance(raw_relations, list):
            raise SchemaError("Legacy schema field 'relations' must be a list")

        entities = []
        for idx, entity in enumerate(raw_entities):
            if not isinstance(entity, dict):
                raise SchemaError(f"Legacy schema entity at index {idx} must be an object")
            name = entity.get("name")
            if not isinstance(name, str) or not name.strip():
                raise SchemaError(f"Legacy schema entity at index {idx} requires non-empty 'name'")
            entities.append(Entity(name.strip()))

        relations = []
        for idx, relation in enumerate(raw_relations):
            if not isinstance(relation, dict):
                raise SchemaError(f"Legacy schema relation at index {idx} must be an object")
            name = relation.get("name")
            src = relation.get("source", relation.get("src", ""))
            dst = relation.get("target", relation.get("dst", ""))
            if not name or not src or not dst:
                raise SchemaError(
                    f"Legacy schema relation at index {idx} requires name/source/target"
                )
            description = relation.get("description", "")
            detail = relation.get("detail", "")
            relations.append(
                Relation(
                    name=str(name).strip(),
                    src=str(src).strip(),
                    dst=str(dst).strip(),
                    description=description if isinstance(description, str) else "",
                    detail=detail if isinstance(detail, str) else "",
                )
            )

        return DRGSchema(entities=entities, relations=relations)
