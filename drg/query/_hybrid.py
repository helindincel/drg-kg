"""Hybrid retrieval over graph structure and semantic document chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._backend import QueryBackend
from ._communities import is_event_type
from ._evidence import edge_to_view, merge_provenance, node_to_view
from ._search import find_entities
from ._traversal import bfs_neighborhood
from ._types import (
    EdgeView,
    EventView,
    EvidenceItem,
    HybridRankingWeights,
    HybridSearchExplanation,
    HybridSearchResult,
    Provenance,
    QueryError,
    VectorChunkHit,
)
from ._vector import VectorStore

__all__ = ["hybrid_search"]

_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "for",
    "to",
    "and",
    "or",
    "with",
    "about",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "which",
    "that",
    "companies",
    "company",
    "connected",
    "relationship",
    "relationships",
    "involve",
    "involves",
    "involving",
}


@dataclass
class _Candidate:
    entity_id: str
    graph_relevance: float = 0.0
    vector_similarity: float = 0.0
    confidence: float = 0.0
    evidence_quality: float = 0.0
    hop_closeness: float = 0.0
    min_hops: int | None = None
    edges: dict[tuple[str, str, str], EdgeView] = field(default_factory=dict)
    chunks: dict[str, VectorChunkHit] = field(default_factory=dict)
    why: list[str] = field(default_factory=list)
    graph_structures: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _tokens(text: str) -> list[str]:
    cleaned = (
        _normalize(text)
        .replace("(", " ")
        .replace(")", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("?", " ")
        .replace("!", " ")
    )
    return [t for t in cleaned.split() if t and t not in _STOPWORDS and len(t) > 1]


def _edge_key(edge: EdgeView) -> tuple[str, str, str]:
    return (edge.source, edge.relationship_type, edge.target)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _edge_query_score(edge: EdgeView, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    edge_text = " ".join(
        [
            edge.source,
            edge.relationship_type,
            edge.target,
            edge.relationship_detail,
            str(edge.metadata.get("relationship_description", "")),
        ]
    )
    edge_tokens = set(_tokens(edge_text))
    if not edge_tokens:
        return 0.0
    return len(edge_tokens.intersection(query_tokens)) / len(query_tokens)


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _add_reason(candidate: _Candidate, reason: str) -> None:
    if reason not in candidate.why:
        candidate.why.append(reason)


def _add_structure(candidate: _Candidate, structure: str) -> None:
    if structure not in candidate.graph_structures:
        candidate.graph_structures.append(structure)


def _mention_score(entity_id: str, text: str) -> bool:
    return _normalize(entity_id) in _normalize(text)


def _incident_events(
    backend: QueryBackend,
    entity_id: str,
    *,
    include_inferred: bool,
) -> tuple[EventView, ...]:
    events: list[EventView] = []
    for edge in backend.edges_incident(
        entity_id,
        direction="both",
        include_inferred=include_inferred,
    ):
        other = edge.target if edge.source == entity_id else edge.source
        node = backend.get_node(other)
        if node is None or not is_event_type(node.type):
            continue
        incident = backend.edges_incident(
            other,
            direction="both",
            include_inferred=include_inferred,
        )
        edge_views = tuple(edge_to_view(e) for e in incident)
        prov = (
            merge_provenance(*(ev.provenance for ev in edge_views))
            if edge_views
            else Provenance()
        )
        events.append(
            EventView(
                event=node_to_view(node, backend),
                incident_edges=edge_views,
                provenance=prov,
            )
        )
    events.sort(key=lambda e: e.event.id.lower())
    return tuple(events)


def _score_candidate(
    candidate: _Candidate,
    *,
    weights: HybridRankingWeights,
) -> tuple[float, dict[str, float]]:
    components = {
        "graph_relevance": _clamp(candidate.graph_relevance),
        "vector_similarity": _clamp(candidate.vector_similarity),
        "confidence": _clamp(candidate.confidence),
        "evidence_quality": _clamp(candidate.evidence_quality),
        "hop_closeness": _clamp(candidate.hop_closeness),
    }
    score = (
        weights.graph_relevance * components["graph_relevance"]
        + weights.vector_similarity * components["vector_similarity"]
        + weights.confidence * components["confidence"]
        + weights.evidence_quality * components["evidence_quality"]
        + weights.hop_closeness * components["hop_closeness"]
    )
    components["final"] = _clamp(score)
    return components["final"], components


def hybrid_search(
    backend: QueryBackend,
    query: str,
    *,
    vector_store: VectorStore | None = None,
    embedding_provider: Any | None = None,
    query_embedding: list[float] | None = None,
    limit: int = 10,
    vector_limit: int = 10,
    graph_seed_limit: int = 8,
    max_hops: int = 2,
    include_inferred: bool = True,
    weights: HybridRankingWeights | None = None,
) -> list[HybridSearchResult]:
    """Retrieve and rank entities using graph and vector evidence together."""
    if max_hops < 1:
        raise QueryError("max_hops must be >= 1")

    weights = weights or HybridRankingWeights()
    query_tokens = set(_tokens(query))
    candidates: dict[str, _Candidate] = {}

    def candidate_for(entity_id: str) -> _Candidate:
        existing = candidates.get(entity_id)
        if existing is not None:
            return existing
        created = _Candidate(entity_id=entity_id)
        candidates[entity_id] = created
        return created

    entity_matches = find_entities(backend, query, limit=graph_seed_limit)
    seed_ids = [m.entity.id for m in entity_matches]
    for match in entity_matches:
        cand = candidate_for(match.entity.id)
        cand.graph_relevance = max(cand.graph_relevance, _clamp(match.score / 5.0))
        cand.hop_closeness = max(cand.hop_closeness, 1.0)
        cand.min_hops = 0
        if match.entity.confidence is not None:
            cand.confidence = max(cand.confidence, match.entity.confidence)
        _add_reason(cand, "entity_name_match")
        for reason in match.match_reasons:
            _add_structure(cand, f"entity:{match.entity.id}:{reason}")

    # Graph expansion: promote neighbors and multi-hop entities, with distance
    # penalties and edge relevance from relation/detail token overlap.
    for seed in seed_ids:
        hood = bfs_neighborhood(
            backend,
            seed,
            hops=max_hops,
            include_inferred=include_inferred,
            max_edges=200,
        )
        distances: dict[str, int] = {seed: 0}
        frontier = {seed}
        for hop in range(1, max_hops + 1):
            next_frontier: set[str] = set()
            for nid in frontier:
                for neighbor in backend.neighbors(
                    nid,
                    direction="both",
                    include_inferred=include_inferred,
                ):
                    if neighbor not in distances:
                        distances[neighbor] = hop
                        next_frontier.add(neighbor)
            frontier = next_frontier

        for edge in hood.edges:
            edge_score = _edge_query_score(edge, query_tokens)
            for entity_id in (edge.source, edge.target):
                if backend.get_node(entity_id) is None:
                    continue
                cand = candidate_for(entity_id)
                cand.edges[_edge_key(edge)] = edge
                hops = distances.get(entity_id, max_hops)
                cand.min_hops = hops if cand.min_hops is None else min(cand.min_hops, hops)
                cand.hop_closeness = max(cand.hop_closeness, 1.0 / (1.0 + hops))
                cand.graph_relevance = max(
                    cand.graph_relevance,
                    _clamp((0.70 / (1.0 + hops)) + 0.30 * edge_score),
                )
                _add_reason(cand, "multi_hop_graph_expansion" if hops > 1 else "direct_graph_neighbor")
                _add_structure(
                    cand,
                    f"{edge.source}-[{edge.relationship_type}]->{edge.target}",
                )

    # Semantic chunk retrieval. The vector store is optional; without it, this
    # function remains graph-only and deterministic.
    vector_hits: list[VectorChunkHit] = []
    if vector_store is not None:
        if query_embedding is None and embedding_provider is not None:
            query_embedding = embedding_provider.embed(query)
        if query_embedding is not None:
            vector_hits = vector_store.search(query_embedding, limit=vector_limit)

    for hit in vector_hits:
        for entity_id in backend.all_node_ids():
            if not _mention_score(entity_id, hit.chunk.text):
                continue
            cand = candidate_for(entity_id)
            cand.chunks[hit.chunk.chunk_id] = hit
            cand.vector_similarity = max(cand.vector_similarity, hit.score)
            cand.graph_relevance = max(cand.graph_relevance, 0.25)
            _add_reason(cand, "semantic_chunk_match")
            _add_structure(cand, f"chunk_mentions_entity:{hit.chunk.chunk_id}:{entity_id}")

    results: list[HybridSearchResult] = []
    for entity_id, cand in candidates.items():
        node = backend.get_node(entity_id)
        if node is None:
            continue

        edge_values = tuple(
            sorted(
                cand.edges.values(),
                key=lambda e: (
                    cand.min_hops if cand.min_hops is not None else max_hops,
                    e.source.lower(),
                    e.relationship_type.lower(),
                    e.target.lower(),
                ),
            )
        )
        evidence_items: list[EvidenceItem] = [
            item for edge in edge_values for item in edge.provenance.evidence
        ]
        confidence_values = [e.confidence for e in edge_values if e.confidence is not None]
        if node.confidence is not None:
            confidence_values.append(node.confidence)
        cand.confidence = max(cand.confidence, _avg(confidence_values))

        evidence_count = len(evidence_items)
        source_count = len(
            {
                item.source_ref
                for item in evidence_items
                if item.source_ref is not None
            }
        )
        chunk_count = len(cand.chunks)
        cand.evidence_quality = max(
            cand.evidence_quality,
            _clamp(0.20 * evidence_count + 0.20 * source_count + 0.15 * chunk_count),
        )

        score, components = _score_candidate(cand, weights=weights)
        chunk_hits = tuple(
            sorted(cand.chunks.values(), key=lambda h: (-h.score, h.chunk.chunk_id))
        )
        explanation = HybridSearchExplanation(
            why=tuple(cand.why),
            graph_structures=tuple(cand.graph_structures),
            document_chunks=tuple(hit.chunk.chunk_id for hit in chunk_hits),
            evidence=tuple(evidence_items),
            ranking_components=components,
        )
        results.append(
            HybridSearchResult(
                entity=node_to_view(node, backend),
                score=score,
                graph_edges=edge_values,
                events=_incident_events(
                    backend,
                    entity_id,
                    include_inferred=include_inferred,
                ),
                document_chunks=chunk_hits,
                explanation=explanation,
            )
        )

    results.sort(key=lambda r: (-r.score, r.entity.id.lower()))
    return results[: max(0, limit)]
