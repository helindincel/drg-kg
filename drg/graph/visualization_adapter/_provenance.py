"""Provenance graph data model.

Three lightweight dataclasses describing a query → answer chain. Used by the
API server to surface "why did the system return this answer?" diagrams to
end users. Kept here (rather than in ``kg_core``) because they are purely a
visualization concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ProvenanceEdge", "ProvenanceGraph", "ProvenanceNode"]


@dataclass
class ProvenanceNode:
    """Node in a provenance chain.

    ``type`` is one of: ``"query"``, ``"chunk"``, ``"community"``,
    ``"summary"``, ``"answer"``.
    """

    id: str
    type: str
    label: str
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceEdge:
    """Edge in a provenance chain.

    ``type`` is typically one of: ``"retrieved_from"``, ``"summarized_in"``,
    ``"generated_from"``.
    """

    source: str
    target: str
    type: str
    label: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceGraph:
    """Complete provenance graph for an explainable query flow."""

    nodes: list[ProvenanceNode]
    edges: list[ProvenanceEdge]
    query: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "label": node.label,
                    "data": node.data,
                    "metadata": node.metadata,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "label": edge.label,
                    "weight": edge.weight,
                    "metadata": edge.metadata,
                }
                for edge in self.edges
            ],
            "metadata": self.metadata,
        }
