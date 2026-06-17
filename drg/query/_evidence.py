"""Evidence extraction helpers for query results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._types import EdgeView, EvidenceBundle, EvidenceItem, Provenance

if TYPE_CHECKING:
    from ..graph.kg_core import KGEdge, KGNode
    from ._backend import QueryBackend
    from ._types import EntityView

__all__ = [
    "edge_to_view",
    "evidence_bundle_for_triple",
    "evidence_from_edge",
    "merge_provenance",
    "node_to_view",
]


def _is_inferred(edge: KGEdge) -> bool:
    return bool(edge.metadata) and bool(edge.metadata.get("inferred"))


def _source_ref(edge: KGEdge) -> str | None:
    if not edge.metadata:
        return None
    ref = edge.metadata.get("source_ref")
    return ref if isinstance(ref, str) and ref else None


def evidence_from_edge(edge: KGEdge) -> EvidenceItem:
    """Build an :class:`EvidenceItem` from a single :class:`KGEdge`."""
    inferred = _is_inferred(edge)
    inference: dict[str, Any] | None = None
    if inferred and edge.metadata:
        raw = edge.metadata.get("inference")
        if isinstance(raw, dict):
            inference = dict(raw)

    return EvidenceItem(
        triple=(edge.source, edge.relationship_type, edge.target),
        source_ref=_source_ref(edge),
        snippet=edge.relationship_detail or None,
        confidence=edge.confidence,
        is_inferred=inferred,
        inference=inference,
    )


def edge_to_view(edge: KGEdge, *, cluster_id: str | None = None) -> EdgeView:
    """Convert a :class:`KGEdge` into an :class:`EdgeView` with provenance."""
    item = evidence_from_edge(edge)
    docs: list[str] = []
    if item.source_ref:
        docs.append(item.source_ref)
    if item.inference:
        for doc in item.inference.get("source_documents") or []:
            if isinstance(doc, str) and doc and doc not in docs:
                docs.append(doc)

    return EdgeView(
        source=edge.source,
        target=edge.target,
        relationship_type=edge.relationship_type,
        relationship_detail=edge.relationship_detail,
        metadata=dict(edge.metadata) if edge.metadata else {},
        confidence=edge.confidence,
        is_inferred=_is_inferred(edge),
        start_time=edge.start_time,
        end_time=edge.end_time,
        valid_from=edge.start_time,
        valid_to=edge.end_time,
        created_at=getattr(edge, "created_at", None),
        updated_at=getattr(edge, "updated_at", None),
        is_negated=edge.is_negated,
        provenance=Provenance(
            source_documents=docs,
            evidence=[item],
            cluster_id=cluster_id,
        ),
    )


def node_to_view(
    node: KGNode,
    backend: QueryBackend,
) -> EntityView:
    from ._types import EntityView as _EntityView

    cluster = backend.cluster_for(node.id)
    cluster_id = cluster[0] if cluster else None
    return _EntityView(
        id=node.id,
        type=node.type,
        properties=dict(node.properties) if node.properties else {},
        metadata=dict(node.metadata) if node.metadata else {},
        confidence=node.confidence,
        cluster_id=cluster_id,
    )


def merge_provenance(*provenances: Provenance) -> Provenance:
    """Merge multiple provenance bags, deduplicating documents and evidence."""
    docs_seen: set[str] = set()
    docs: list[str] = []
    evidence_seen: set[tuple[Any, ...]] = set()
    evidence: list[EvidenceItem] = []
    cluster_id: str | None = None

    for prov in provenances:
        if prov.cluster_id and not cluster_id:
            cluster_id = prov.cluster_id
        for doc in prov.source_documents:
            if doc not in docs_seen:
                docs_seen.add(doc)
                docs.append(doc)
        for item in prov.evidence:
            key = (
                item.triple,
                item.source_ref,
                item.snippet,
                item.is_inferred,
            )
            if key not in evidence_seen:
                evidence_seen.add(key)
                evidence.append(item)

    return Provenance(
        source_documents=docs,
        evidence=evidence,
        cluster_id=cluster_id,
    )


def evidence_bundle_for_triple(
    backend: QueryBackend,
    triple: tuple[str, str, str],
    *,
    include_inferred: bool = True,
) -> EvidenceBundle:
    """Collect all edges and evidence supporting a relationship triple."""
    src, rel, tgt = triple
    edges_raw = backend.edges_matching(
        source=src,
        target=tgt,
        relationship_type=rel,
        include_inferred=include_inferred,
    )
    if not edges_raw:
        edges_raw = backend.edges_matching(
            source=src,
            target=tgt,
            include_inferred=include_inferred,
        )
        rel_norm = rel.strip().lower()
        edges_raw = [e for e in edges_raw if e.relationship_type.lower() == rel_norm]

    edge_views = tuple(edge_to_view(e) for e in edges_raw)
    evidence_items: list[EvidenceItem] = []
    docs_seen: set[str] = set()
    docs: list[str] = []

    for ev in edge_views:
        for item in ev.provenance.evidence:
            evidence_items.append(item)
            if item.source_ref and item.source_ref not in docs_seen:
                docs_seen.add(item.source_ref)
                docs.append(item.source_ref)
            if item.inference:
                for doc in item.inference.get("source_documents") or []:
                    if isinstance(doc, str) and doc and doc not in docs_seen:
                        docs_seen.add(doc)
                        docs.append(doc)

    if edge_views:
        summary = (
            f"Found {len(edge_views)} edge(s) supporting "
            f"{src} —[{rel}]→ {tgt} from {len(docs)} source document(s)."
        )
    else:
        summary = f"No edges found supporting {src} —[{rel}]→ {tgt}."

    return EvidenceBundle(
        triple=triple,
        edges=edge_views,
        evidence=tuple(evidence_items),
        source_documents=tuple(docs),
        summary=summary,
    )
