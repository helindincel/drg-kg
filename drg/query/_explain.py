"""Path explanation builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._communities import community_of
from ._traversal import find_paths
from ._types import EvidenceItem, Explanation, GraphPath, QueryError

if TYPE_CHECKING:
    from ._backend import QueryBackend

__all__ = ["build_explanation"]


def _dedupe_evidence(paths: tuple[GraphPath, ...]) -> tuple[EvidenceItem, ...]:
    seen: set[tuple] = set()
    items: list[EvidenceItem] = []
    for path in paths:
        for edge in path.edges:
            for item in edge.provenance.evidence:
                key = (item.triple, item.source_ref, item.snippet, item.is_inferred)
                if key not in seen:
                    seen.add(key)
                    items.append(item)
    return tuple(items)


def _shared_community(backend: QueryBackend, source: str, target: str):
    src_cluster = backend.cluster_for(source)
    tgt_cluster = backend.cluster_for(target)
    if src_cluster and tgt_cluster and src_cluster[0] == tgt_cluster[0]:
        return community_of(backend, source)
    return None


def _summarize(
    source: str,
    target: str,
    paths: tuple[GraphPath, ...],
    evidence: tuple[EvidenceItem, ...],
    shared_community,
) -> str:
    if not paths:
        return (
            f"No graph path was found between {source!r} and {target!r} "
            f"within the requested hop limit. No unsupported connection is reported."
        )

    best = paths[0]
    hop_phrase = f"{best.hop_count} hop(s)" if best.hop_count else "0 hops"
    via = " → ".join(best.nodes)
    doc_ids = sorted(
        {d for item in evidence for d in ([item.source_ref] if item.source_ref else [])}
        | {
            doc
            for item in evidence
            if item.inference
            for doc in (item.inference.get("source_documents") or [])
            if isinstance(doc, str)
        }
    )
    docs_phrase = f" Source document(s): {', '.join(doc_ids)}." if doc_ids else ""
    community_phrase = ""
    if shared_community:
        community_phrase = f" Both entities belong to community {shared_community.cluster_id}."

    inferred = any(item.is_inferred for item in evidence)
    inferred_phrase = " Some supporting edges are inferred (rule-based)." if inferred else ""

    return (
        f"{source} is connected to {target} via {hop_phrase} "
        f"({via}).{docs_phrase}{community_phrase}{inferred_phrase}"
    )


def build_explanation(
    backend: QueryBackend,
    source: str,
    target: str,
    *,
    max_hops: int = 3,
    max_paths: int = 5,
    include_inferred: bool = True,
) -> Explanation:
    """Explain why ``source`` and ``target`` are connected (or not)."""
    if backend.get_node(source) is None:
        raise QueryError(f"Entity not found: {source!r}")
    if backend.get_node(target) is None:
        raise QueryError(f"Entity not found: {target!r}")

    paths_list = find_paths(
        backend,
        source,
        target,
        max_hops=max_hops,
        max_paths=max_paths,
        include_inferred=include_inferred,
    )
    paths = tuple(paths_list)
    evidence = _dedupe_evidence(paths)
    shared = _shared_community(backend, source, target)

    return Explanation(
        source=source,
        target=target,
        connected=bool(paths),
        paths=paths,
        evidence=evidence,
        shared_community=shared,
        summary=_summarize(source, target, paths, evidence, shared),
    )
