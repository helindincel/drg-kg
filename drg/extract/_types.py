"""Lightweight result containers and Pydantic models for extraction.

These types are deliberately separate from the main extractor module so they
can be imported without pulling in heavy DSPy machinery.

This is an internal module; symbols are re-exported from `drg.extract` where
appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic.config import ConfigDict


@dataclass(frozen=True)
class ExtractionResult:
    """Extraction result (prediction-like) container.

    Attributes:
        entities: List of (entity_name, entity_type) tuples.
        relations: List of (source, relation, target) triples.
        enriched_relations: Optional per-relation metadata aligned with `relations`.
    """

    entities: list[tuple[str, str]]
    relations: list[tuple[str, str, str]]
    enriched_relations: list[dict[str, Any]] | None = None


class EntityMention(BaseModel):
    """Typed DSPy output for one entity mention."""

    model_config = ConfigDict(extra="ignore")
    name: str
    type: str
    aliases: list[str] = []
    evidence: str | None = None
    properties: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class TemporalInfo(BaseModel):
    """Typed temporal metadata for a relation."""

    model_config = ConfigDict(extra="ignore")
    start: str | None = None
    end: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    precision: str | None = None
    text: str | None = None


class ExtractedRelation(BaseModel):
    """Typed DSPy output for one relation and its metadata."""

    model_config = ConfigDict(extra="ignore")
    source: str
    relation: str
    target: str
    confidence: float | None = None
    evidence: str | None = None
    temporal: TemporalInfo | dict[str, Any] | None = None
    is_negated: bool = False
    metadata: dict[str, Any] = {}


class EntityList(BaseModel):
    """Structured output model for entity extraction."""

    model_config = ConfigDict(extra="ignore")
    entities: list[EntityMention]


class RelationList(BaseModel):
    """Structured output model for relation extraction."""

    model_config = ConfigDict(extra="ignore")
    relations: list[ExtractedRelation]


class SchemaEntityType(BaseModel):
    """Structured output model for one generated entity type."""

    model_config = ConfigDict(extra="ignore")
    name: str
    description: str
    examples: list[str] = []
    properties: dict[str, Any] = {}


class SchemaRelation(BaseModel):
    """Structured output model for one generated relation definition."""

    model_config = ConfigDict(extra="ignore")
    name: str
    source: str
    target: str
    description: str = ""
    detail: str = ""
    properties: dict[str, Any] = {}


class SchemaRelationGroup(BaseModel):
    """Structured output model for one generated relation group."""

    model_config = ConfigDict(extra="ignore")
    name: str
    description: str
    relations: list[SchemaRelation]
    examples: list[dict[str, Any]] = []


class SchemaOutput(BaseModel):
    """Structured output model for schema generation."""

    model_config = ConfigDict(extra="ignore")
    entity_types: list[SchemaEntityType]
    relation_groups: list[SchemaRelationGroup]
    auto_discovery: bool = False
