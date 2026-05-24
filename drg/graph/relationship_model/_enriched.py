"""Enriched relationship data model.

A small dataclass with input validation in ``__post_init__``. The validation
raises plain :class:`ValueError` (not :class:`drg.errors.GraphError`) because
these are basic data-shape contracts — callers typically catch them at
construction sites alongside other ``ValueError``s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._types import RelationshipType

__all__ = ["EnrichedRelationship", "create_enriched_relationship"]


@dataclass
class EnrichedRelationship:
    """Enriched relationship with type, detail and metadata.

    Fields:
        source: Source entity identifier.
        target: Target entity identifier.
        relationship_type: Type from :class:`RelationshipType` taxonomy.
        relationship_detail: Short sentence explaining the relationship.
        confidence: Confidence score in ``[0, 1]``.
        source_ref: Reference (chunk_id / document_id) for provenance.
            Currently unused by the pipeline, reserved for future tracking.
    """

    source: str
    target: str
    relationship_type: RelationshipType
    relationship_detail: str
    confidence: float = 1.0
    source_ref: str | None = field(default=None)

    def __post_init__(self):
        if not self.source:
            raise ValueError("Source cannot be empty")
        if not self.target:
            raise ValueError("Target cannot be empty")
        if not self.relationship_detail:
            raise ValueError("Relationship detail cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.source == self.target:
            raise ValueError("Source and target cannot be the same")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type.value,
            "relationship_detail": self.relationship_detail,
            "confidence": self.confidence,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnrichedRelationship:
        return cls(
            source=data["source"],
            target=data["target"],
            relationship_type=RelationshipType(data["relationship_type"]),
            relationship_detail=data["relationship_detail"],
            confidence=data.get("confidence", 1.0),
            source_ref=data.get("source_ref"),
        )

    def to_enriched_format(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type.value,
            "relationship_detail": self.relationship_detail,
            "confidence": self.confidence,
            "source_ref": self.source_ref or "unknown",
        }


def create_enriched_relationship(
    source: str,
    target: str,
    relationship_type: RelationshipType,
    relationship_detail: str,
    confidence: float = 1.0,
    source_ref: str | None = None,
) -> EnrichedRelationship:
    """Factory for :class:`EnrichedRelationship` — kept for backward compatibility.

    Calling the dataclass directly works identically; this is just an explicit
    entry point that some legacy call sites depend on.
    """
    return EnrichedRelationship(
        source=source,
        target=target,
        relationship_type=relationship_type,
        relationship_detail=relationship_detail,
        confidence=confidence,
        source_ref=source_ref,
    )
