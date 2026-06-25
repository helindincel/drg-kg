"""Type definitions for multi-document reasoning.

These types represent the data structures used during inference:
- ``InferenceRule`` — abstract base for all inference rules
- ``InferredEdge`` — an edge produced by reasoning (not direct extraction)
- ``EvidenceLink`` — pointer back to the source edges that justify an inferred edge
- ``InferenceReport`` — summary of a single reasoning pass
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from drg.graph.kg_core import EnhancedKG, KGEdge


__all__ = [
    "EvidenceLink",
    "InferenceReport",
    "InferenceRule",
    "InferredEdge",
]


@dataclass(frozen=True)
class EvidenceLink:
    """Reference to a source edge that supports an inferred relationship."""

    source_node: str
    target_node: str
    relationship_type: str
    document_id: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source_node,
            "target": self.target_node,
            "relationship_type": self.relationship_type,
        }
        if self.document_id is not None:
            out["document_id"] = self.document_id
        if self.confidence is not None:
            out["confidence"] = self.confidence
        return out


@dataclass
class InferredEdge:
    """An edge produced by a reasoning rule rather than direct extraction."""

    source: str
    target: str
    relationship_type: str
    relationship_detail: str
    confidence: float
    rule_name: str
    evidence: list[EvidenceLink] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type,
            "relationship_detail": self.relationship_detail,
            "confidence": self.confidence,
            "rule_name": self.rule_name,
            "evidence": [e.to_dict() for e in self.evidence],
            "metadata": self.metadata,
        }


@dataclass
class InferenceReport:
    """Summary of a single reasoning pass over a knowledge graph."""

    document_id: str | None
    rules_applied: list[str]
    edges_inferred: int
    edges_added: int
    skipped_low_confidence: int
    dry_run: bool
    inferred_edges: list[InferredEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "rules_applied": self.rules_applied,
            "edges_inferred": self.edges_inferred,
            "edges_added": self.edges_added,
            "skipped_low_confidence": self.skipped_low_confidence,
            "dry_run": self.dry_run,
            "inferred_edges": [e.to_dict() for e in self.inferred_edges],
        }


class InferenceRule(abc.ABC):
    """Abstract base class for all reasoning rules.

    Subclasses must implement :meth:`apply` which receives the current
    ``EnhancedKG`` and returns a (possibly empty) list of :class:`InferredEdge`
    candidates.  The engine filters candidates by ``min_confidence`` and
    deduplicates before writing them back to the graph.
    """

    #: Short identifier used in ``ReasoningConfig.disabled_rules``
    name: str = ""

    @abc.abstractmethod
    def apply(
        self,
        kg: "EnhancedKG",
        *,
        document_id: str | None = None,
    ) -> list[InferredEdge]:
        """Return inferred edge candidates for *kg*.

        Implementations must be pure functions with respect to *kg* — they
        **must not** mutate the graph.  The engine handles all mutations.
        """
