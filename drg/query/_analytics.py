"""Small graph analytics algorithms over the query backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._evidence import node_to_view
from ._types import GraphMetricScore, QueryError

if TYPE_CHECKING:
    from ._backend import QueryBackend

__all__ = ["degree_centrality", "influence_scores", "pagerank"]


def degree_centrality(
    backend: QueryBackend,
    *,
    limit: int = 10,
    include_inferred: bool = True,
) -> list[GraphMetricScore]:
    """Rank entities by normalized undirected degree centrality."""
    node_ids = backend.all_node_ids()
    denom = max(1, len(node_ids) - 1)
    scores = {
        node_id: len(
            backend.edges_incident(node_id, direction="both", include_inferred=include_inferred)
        )
        / denom
        for node_id in node_ids
    }
    return _rank_scores(backend, scores, "degree_centrality", limit=limit)


def pagerank(
    backend: QueryBackend,
    *,
    limit: int = 10,
    iterations: int = 30,
    damping: float = 0.85,
    include_inferred: bool = True,
) -> list[GraphMetricScore]:
    """Compute lightweight PageRank over the graph adjacency."""
    node_ids = backend.all_node_ids()
    n = len(node_ids)
    if n == 0:
        return []
    ranks = dict.fromkeys(node_ids, 1.0 / n)
    base = (1.0 - damping) / n

    for _ in range(max(1, iterations)):
        next_ranks = dict.fromkeys(node_ids, base)
        for node_id in node_ids:
            neighbors = backend.neighbors(
                node_id, direction="both", include_inferred=include_inferred
            )
            if not neighbors:
                share = damping * ranks[node_id] / n
                for target in node_ids:
                    next_ranks[target] += share
                continue
            share = damping * ranks[node_id] / len(neighbors)
            for target in neighbors:
                if target in next_ranks:
                    next_ranks[target] += share
        ranks = next_ranks

    return _rank_scores(backend, ranks, "pagerank", limit=limit)


def influence_scores(
    backend: QueryBackend,
    *,
    limit: int = 10,
    include_inferred: bool = True,
) -> list[GraphMetricScore]:
    """Blend PageRank and degree centrality into one influence score."""
    pr = {
        item.entity.id: item.score
        for item in pagerank(backend, limit=10_000, include_inferred=include_inferred)
    }
    degree = {
        item.entity.id: item.score
        for item in degree_centrality(
            backend,
            limit=10_000,
            include_inferred=include_inferred,
        )
    }
    max_pr = max(pr.values(), default=1.0) or 1.0
    scores = {
        node_id: 0.6 * (pr.get(node_id, 0.0) / max_pr) + 0.4 * degree.get(node_id, 0.0)
        for node_id in backend.all_node_ids()
    }
    return _rank_scores(backend, scores, "influence_score", limit=limit)


def _rank_scores(
    backend: QueryBackend,
    scores: dict[str, float],
    metric: str,
    *,
    limit: int,
) -> list[GraphMetricScore]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].lower()))
    out: list[GraphMetricScore] = []
    for rank, (node_id, score) in enumerate(ranked[: max(1, limit)], start=1):
        node = backend.get_node(node_id)
        if node is None:
            raise QueryError(f"Entity not found while ranking analytics score: {node_id!r}")
        out.append(
            GraphMetricScore(
                entity=node_to_view(node, backend),
                metric=metric,
                score=float(score),
                rank=rank,
                details={"degree": backend.node_degree(node_id)},
            )
        )
    return out
