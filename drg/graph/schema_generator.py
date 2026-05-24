"""
Dataset-Agnostic Schema Generation Module

This module provides a structured, extensible system for generating entity schemas
with rich property definitions. It supports Person, Location, Event, and other
entity types with properties that include name, description, and example_value.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PropertyDefinition:
    """
    Structured property definition for entity types.
    Each property has a name, description, and example value.
    """

    name: str
    description: str
    example_value: Any

    def __post_init__(self):
        """Validate property definition."""
        if not self.name:
            raise ValueError("Property name cannot be empty")
        if not self.description:
            raise ValueError("Property description cannot be empty")
        if self.example_value is None:
            raise ValueError("Property example_value cannot be None")

    def to_dict(self) -> dict[str, Any]:
        """Convert property to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "example_value": self.example_value,
        }


@dataclass
class EntityClassDefinition:
    """
    Entity class definition with structured properties.
    This represents a dataset-agnostic entity type that can be used
    across different domains.
    """

    name: str
    description: str
    properties: list[PropertyDefinition] = field(default_factory=list)

    def __post_init__(self):
        """Validate entity class definition."""
        if not self.name:
            raise ValueError("Entity class name cannot be empty")
        if not self.description:
            raise ValueError("Entity class description cannot be empty")

    def add_property(self, prop: PropertyDefinition) -> None:
        """Add a property to this entity class."""
        # Check for duplicate property names
        if any(p.name == prop.name for p in self.properties):
            raise ValueError(f"Property '{prop.name}' already exists in entity class '{self.name}'")
        self.properties.append(prop)

    def get_property(self, name: str) -> PropertyDefinition | None:
        """Get a property by name."""
        for prop in self.properties:
            if prop.name == name:
                return prop
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert entity class to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "properties": [prop.to_dict() for prop in self.properties],
        }


class DatasetAgnosticSchemaGenerator:
    """
    Generator for dataset-agnostic entity schemas.
    Provides predefined entity classes (Person, Location, Event) and
    allows extension with custom entity types.
    """

    def __init__(self):
        self.entity_classes: dict[str, EntityClassDefinition] = {}
        self._initialize_default_entities()

    def _initialize_default_entities(self) -> None:
        """Initialize default entity classes: Person, Location, Event."""

        # Person entity class
        person = EntityClassDefinition(
            name="Person",
            description="A human individual with emotions, intentions, traits, and relationships",
        )
        person.add_property(
            PropertyDefinition(
                name="emotion",
                description="Current or predominant emotional state of the person",
                example_value="joyful",
            )
        )
        person.add_property(
            PropertyDefinition(
                name="intent",
                description="Primary intention or goal of the person",
                example_value="to seek truth",
            )
        )
        person.add_property(
            PropertyDefinition(
                name="traits",
                description="Character traits or personality attributes",
                example_value=["curious", "brave", "loyal"],
            )
        )
        person.add_property(
            PropertyDefinition(
                name="relationships",
                description="Relationships with other entities (people, organizations, etc.)",
                example_value={"mother": "Sarah", "mentor": "Dr. Johnson"},
            )
        )
        person.add_property(
            PropertyDefinition(
                name="role",
                description="Role or position of the person in the context",
                example_value="protagonist",
            )
        )
        person.add_property(
            PropertyDefinition(
                name="age", description="Age or age range of the person", example_value=25
            )
        )
        self.entity_classes["Person"] = person

        # Location entity class
        location = EntityClassDefinition(
            name="Location",
            description="A place or geographical entity with atmosphere and symbolic meaning",
        )
        location.add_property(
            PropertyDefinition(
                name="atmosphere",
                description="The emotional or sensory atmosphere of the location",
                example_value="mysterious and foreboding",
            )
        )
        location.add_property(
            PropertyDefinition(
                name="symbolism",
                description="Symbolic meaning or representation of the location",
                example_value="represents isolation and introspection",
            )
        )
        location.add_property(
            PropertyDefinition(
                name="type",
                description="Type or category of the location",
                example_value="mountain",
            )
        )
        location.add_property(
            PropertyDefinition(
                name="coordinates",
                description="Geographical coordinates (latitude, longitude) if applicable",
                example_value={"lat": 40.7128, "lon": -74.0060},
            )
        )
        location.add_property(
            PropertyDefinition(
                name="features",
                description="Notable features or characteristics of the location",
                example_value=["ancient ruins", "dense forest", "hidden cave"],
            )
        )
        self.entity_classes["Location"] = location

        # Event entity class
        event = EntityClassDefinition(
            name="Event",
            description="An occurrence or happening with actors, outcomes, and temporal scope",
        )
        event.add_property(
            PropertyDefinition(
                name="actors",
                description="Entities (persons, organizations) involved in the event",
                example_value=["Alice", "Bob", "The Council"],
            )
        )
        event.add_property(
            PropertyDefinition(
                name="outcomes",
                description="Results or consequences of the event",
                example_value="The treaty was signed, ending the conflict",
            )
        )
        event.add_property(
            PropertyDefinition(
                name="temporal_scope",
                description="Time period or duration of the event",
                example_value="2024-01-15 to 2024-01-20",
            )
        )
        event.add_property(
            PropertyDefinition(
                name="type", description="Category or type of the event", example_value="ceremony"
            )
        )
        event.add_property(
            PropertyDefinition(
                name="significance",
                description="Importance or impact level of the event",
                example_value="high",
            )
        )
        event.add_property(
            PropertyDefinition(
                name="cause",
                description="Cause or trigger of the event",
                example_value="The discovery of ancient artifacts",
            )
        )
        self.entity_classes["Event"] = event

    def add_entity_class(self, entity_class: EntityClassDefinition) -> None:
        """Add a custom entity class to the schema."""
        if entity_class.name in self.entity_classes:
            raise ValueError(
                f"Entity class '{entity_class.name}' already exists. Use update_entity_class() to modify."
            )
        self.entity_classes[entity_class.name] = entity_class

    def update_entity_class(self, entity_class: EntityClassDefinition) -> None:
        """Update an existing entity class."""
        if entity_class.name not in self.entity_classes:
            raise ValueError(
                f"Entity class '{entity_class.name}' does not exist. Use add_entity_class() to add new classes."
            )
        self.entity_classes[entity_class.name] = entity_class

    def get_entity_class(self, name: str) -> EntityClassDefinition | None:
        """Get an entity class by name."""
        return self.entity_classes.get(name)

    def remove_entity_class(self, name: str) -> None:
        """Remove an entity class from the schema."""
        if name not in self.entity_classes:
            raise ValueError(f"Entity class '{name}' does not exist")
        if name in ["Person", "Location", "Event"]:
            raise ValueError(f"Cannot remove default entity class '{name}'")
        del self.entity_classes[name]

    def to_dict(self) -> dict[str, Any]:
        """Convert the entire schema to a dictionary."""
        return {
            "schema_version": "1.0",
            "description": "Dataset-agnostic entity schema with structured properties",
            "entity_classes": {
                name: entity.to_dict() for name, entity in sorted(self.entity_classes.items())
            },
        }

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """
        Export schema to JSON format.

        Args:
            indent: Number of spaces for indentation
            ensure_ascii: If True, non-ASCII characters are escaped

        Returns:
            JSON string representation of the schema
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)

    def to_yaml(self) -> str:
        """
        Export schema to YAML format.

        Returns:
            YAML string representation of the schema

        Raises:
            ImportError: If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError as err:
            raise ImportError(
                "PyYAML is required for YAML export. Install it with: pip install pyyaml"
            ) from err

        # Use default_flow_style=False for readable YAML
        return yaml.dump(
            self.to_dict(), default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    def save_json(self, filepath: str | Path, indent: int = 2) -> None:
        """
        Save schema to a JSON file.

        Args:
            filepath: Path to the output JSON file
            indent: Number of spaces for indentation
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")

    def save_yaml(self, filepath: str | Path) -> None:
        """
        Save schema to a YAML file.

        Args:
            filepath: Path to the output YAML file

        Raises:
            ImportError: If PyYAML is not installed
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetAgnosticSchemaGenerator":
        """
        Create a schema generator from a dictionary.

        Args:
            data: Dictionary containing schema definition

        Returns:
            DatasetAgnosticSchemaGenerator instance
        """
        generator = cls()
        # Clear default entities if rebuilding from scratch
        generator.entity_classes = {}

        entity_classes_data = data.get("entity_classes", {})
        for entity_name, entity_data in entity_classes_data.items():
            properties = [
                PropertyDefinition(
                    name=prop_data["name"],
                    description=prop_data["description"],
                    example_value=prop_data["example_value"],
                )
                for prop_data in entity_data.get("properties", [])
            ]

            entity_class = EntityClassDefinition(
                name=entity_data["name"],
                description=entity_data["description"],
                properties=properties,
            )
            generator.entity_classes[entity_name] = entity_class

        return generator

    @classmethod
    def from_json(cls, filepath: str | Path) -> "DatasetAgnosticSchemaGenerator":
        """
        Load schema generator from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            DatasetAgnosticSchemaGenerator instance
        """
        path = Path(filepath)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "DatasetAgnosticSchemaGenerator":
        """
        Load schema generator from a YAML file.

        Args:
            filepath: Path to the YAML file

        Returns:
            DatasetAgnosticSchemaGenerator instance

        Raises:
            ImportError: If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError as err:
            raise ImportError(
                "PyYAML is required for YAML import. Install it with: pip install pyyaml"
            ) from err

        path = Path(filepath)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def create_default_schema() -> DatasetAgnosticSchemaGenerator:
    """
    Create a schema generator with default entity classes.

    Returns:
        DatasetAgnosticSchemaGenerator with Person, Location, and Event entities
    """
    return DatasetAgnosticSchemaGenerator()
