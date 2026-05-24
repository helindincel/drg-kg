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


class EntityList(BaseModel):
    """Structured output model for entity extraction."""

    model_config = ConfigDict(extra="ignore")
    entities: list[tuple[str, str]]


class RelationList(BaseModel):
    """Structured output model for relation extraction."""

    model_config = ConfigDict(extra="ignore")
    relations: list[tuple[str, str, str]]


class SchemaOutput(BaseModel):
    """Structured output model for schema generation."""

    model_config = ConfigDict(extra="ignore")
    generated_schema: str
