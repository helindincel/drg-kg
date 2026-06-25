"""Provenance/metadata helpers for inferred edges.

Provides :func:`build_inference_metadata` which stamps the inference
provenance block onto an inferred edge's metadata dict before it is written
back to the graph.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ._types import InferredEdge

__all__ = ["build_inference_metadata", "stamp_inferred_edge"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_inference_metadata(
    edge: InferredEdge,
    *,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Return a metadata dict suitable for a :class:`~drg.graph.kg_core.KGEdge`.

    The dict includes:
    * ``inferred: True`` — flags the edge as not directly extracted
    * ``inference`` — provenance block with rule name, confidence, evidence
      chain, and timestamp
    """
    inference_block: dict[str, Any] = {
        "rule": edge.rule_name,
        "confidence": edge.confidence,
        "inferred_at": _utc_now_iso(),
        "evidence": [ev.to_dict() for ev in edge.evidence],
    }
    if document_id is not None:
        inference_block["document_id"] = document_id

    meta: dict[str, Any] = dict(edge.metadata)
    meta["inferred"] = True
    meta["inference"] = inference_block
    return meta


def stamp_inferred_edge(
    edge: InferredEdge,
    *,
    document_id: str | None = None,
) -> InferredEdge:
    """Return a copy of *edge* with fully populated inference metadata."""
    return InferredEdge(
        source=edge.source,
        target=edge.target,
        relationship_type=edge.relationship_type,
        relationship_detail=edge.relationship_detail,
        confidence=edge.confidence,
        rule_name=edge.rule_name,
        evidence=edge.evidence,
        metadata=build_inference_metadata(edge, document_id=document_id),
    )
