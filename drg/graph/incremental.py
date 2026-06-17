"""Incremental graph updates for :class:`drg.graph.kg_core.EnhancedKG`.

This module lets callers add new documents to an existing knowledge graph
**without rebuilding from scratch**. It is the persistence-layer counterpart
to the in-chunk dedup logic that already lives in :mod:`drg.entity_resolution`
— that one merges entity mentions inside a single document, this one merges
freshly-extracted graphs into a long-lived KG on disk.

Design constraints honoured here
---------------------------------
- **No new heavy dependencies.** Pure stdlib + the existing
  :mod:`drg.entity_resolution` package (already on the import path).
- **Optional, never required.** The legacy "build from scratch" workflow is
  untouched; callers that don't import this module see no behaviour change.
- **Conservative merge defaults.** When in doubt, prefer the existing
  graph (callers can override this by picking a different
  :class:`MergeStrategy`). Refusing a merge is always safer than
  fabricating a false equivalence.
- **Typed, side-effect-free helpers.** Every public function returns a
  :class:`KGDiff` so callers can audit what changed without re-reading the
  graph.

Public surface
--------------

- :class:`MergeStrategy` — value-class describing how to combine matched
  nodes (``PREFER_EXISTING`` / ``PREFER_NEW`` / ``UNION``) and edges
  (skip / append-evidence / max-confidence).
- :class:`KGDiff` — structured report of every node/edge that was added,
  matched, merged, or skipped during a single ``merge`` call.
- :class:`GraphMerger` — stateless merger; one instance can be reused
  across many incremental updates.
- :func:`merge_graphs` — top-level convenience that wires sane defaults.

Quick example
-------------

::

    from drg.graph import EnhancedKG, GraphMerger, MergeStrategy
    from drg.graph.builders import build_enhanced_kg

    # 1. Re-hydrate the persisted KG.
    kg = EnhancedKG.load_json("outputs/global_kg.json")

    # 2. Run the existing extraction pipeline on a new document.
    new_kg = build_enhanced_kg(
        entities_typed=entities,
        triples=triples,
        source_text=text,
    )

    # 3. Merge — entity matching, dedup, version bump all happen here.
    merger = GraphMerger()
    diff = merger.merge(kg, new_kg, document_id="doc_42")
    print(diff.summary())

    # 4. Persist; the graph now carries an incremented version + a
    #    history entry describing this merge.
    kg.save_json("outputs/global_kg.json")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ..entity_resolution import normalize_entity_name
from ..utils.logging import get_logger
from .kg_core import Cluster, EnhancedKG, KGEdge, KGNode

logger = get_logger(__name__)

__all__ = [
    "EdgeMergePolicy",
    "GraphMerger",
    "KGDiff",
    "MergeStrategy",
    "NodeMergePolicy",
    "merge_graphs",
]


# ---------------------------------------------------------------------------
# Strategy value-objects
# ---------------------------------------------------------------------------


class NodeMergePolicy(str, Enum):
    """How to combine an incoming node with a matched existing node."""

    PREFER_EXISTING = "prefer_existing"
    """Keep the existing node verbatim. The incoming node's metadata is
    *recorded* in ``metadata.merged_from`` for provenance but the existing
    node's properties / type / embedding win. Safest default."""

    PREFER_NEW = "prefer_new"
    """The incoming node overwrites the existing one's mutable fields
    (``type``, ``properties``, ``metadata``, ``embedding``, ``confidence``).
    Useful when the new document is known to be more authoritative."""

    UNION = "union"
    """Take the union of properties / metadata dicts (incoming wins on
    overlapping keys), keep the higher confidence, and average the
    embeddings element-wise when both are present and same-dimension."""


class EdgeMergePolicy(str, Enum):
    """How to handle an incoming edge whose canonical key already exists."""

    SKIP = "skip"
    """Drop the duplicate. Default — keeps the graph small and avoids
    fabricating new "evidence" out of repeated extractions."""

    APPEND_EVIDENCE = "append_evidence"
    """Keep the original edge but extend its ``metadata.evidence_refs``
    list with the duplicate's evidence/source_ref so downstream consumers
    can see *which* documents independently observed the same fact."""

    MAX_CONFIDENCE = "max_confidence"
    """Replace the existing edge with the duplicate when the latter has a
    strictly higher confidence; otherwise behave like
    ``APPEND_EVIDENCE``. The lower-confidence variant is preserved in
    ``metadata.alt_confidences``."""


@dataclass(frozen=True)
class MergeStrategy:
    """Bundle of merge policies + matching parameters.

    The defaults are deliberately conservative and dependency-light:
    matching uses normalized-name equality with type agreement, the node
    policy is ``PREFER_EXISTING`` and the edge policy is ``SKIP``. Override
    fields individually for finer control; the :func:`MergeStrategy.default`
    classmethod returns the same defaults but reads more clearly at call
    sites.
    """

    node_policy: NodeMergePolicy = NodeMergePolicy.PREFER_EXISTING
    edge_policy: EdgeMergePolicy = EdgeMergePolicy.SKIP
    require_type_match: bool = True
    """When ``True``, two nodes only match if their ``type`` fields agree
    (or both are ``None``). Setting this to ``False`` lets nodes whose
    type changed across schema versions still merge — at the cost of
    occasionally collapsing genuinely different entities that happen to
    share a surface form."""

    use_normalized_match: bool = True
    """When ``True``, ``"Apple Inc."`` and ``"apple inc"`` map to the
    same canonical id during matching. Disable to require byte-exact
    name equality (rarely useful)."""

    case_insensitive_relation: bool = True
    """When ``True``, ``RUNS_ON`` and ``runs_on`` are treated as the
    same relation type for edge dedup. Surface form on existing edges is
    preserved."""

    @classmethod
    def default(cls) -> MergeStrategy:
        """Return the recommended default — explicit at call sites."""
        return cls()


# ---------------------------------------------------------------------------
# Diff report
# ---------------------------------------------------------------------------


@dataclass
class KGDiff:
    """Structured report of a single ``merge`` call.

    Every list contains the raw KG element identifiers used by the
    public surface (node ids and ``(source, type, target)`` triples for
    edges), so callers can replay the diff against any other KG snapshot
    without holding live ``KGNode`` / ``KGEdge`` references.
    """

    added_nodes: list[str] = field(default_factory=list)
    """Node ids that did not exist in the base graph and were inserted."""

    merged_nodes: list[tuple[str, str]] = field(default_factory=list)
    """``(existing_id, incoming_id)`` pairs where the matcher found an
    existing node and the incoming node's data was folded in (or
    discarded, depending on the node policy). When the two ids are
    equal it means an exact id-level match; otherwise the incoming
    name was normalised to an existing canonical form."""

    skipped_nodes: list[tuple[str, str]] = field(default_factory=list)
    """Incoming node ids deliberately *not* merged — currently only
    populated when ``require_type_match=True`` blocks an otherwise
    name-equal match. ``(incoming_id, reason)``."""

    added_edges: list[tuple[str, str, str]] = field(default_factory=list)
    """``(canonical_source, relationship_type, canonical_target)`` triples
    that were inserted into the base graph."""

    skipped_edges: list[tuple[str, str, str]] = field(default_factory=list)
    """Edges whose canonical triple already existed in the base graph."""

    rewritten_edges: list[tuple[str, str, str]] = field(default_factory=list)
    """Edges whose source/target was rewritten because the matcher mapped
    one of the endpoints onto an existing canonical node id."""

    added_clusters: list[str] = field(default_factory=list)
    """Cluster ids that were copied verbatim from the incoming graph."""

    skipped_clusters: list[str] = field(default_factory=list)
    """Cluster ids dropped because they already existed in the base graph."""

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """``True`` iff the merge was a complete no-op."""
        return not (
            self.added_nodes
            or self.merged_nodes
            or self.added_edges
            or self.skipped_edges
            or self.rewritten_edges
            or self.added_clusters
            or self.skipped_clusters
            or self.skipped_nodes
        )

    def summary(self) -> dict[str, int]:
        """Compact stats dict — handy for logging."""
        return {
            "added_nodes": len(self.added_nodes),
            "merged_nodes": len(self.merged_nodes),
            "skipped_nodes": len(self.skipped_nodes),
            "added_edges": len(self.added_edges),
            "skipped_edges": len(self.skipped_edges),
            "rewritten_edges": len(self.rewritten_edges),
            "added_clusters": len(self.added_clusters),
            "skipped_clusters": len(self.skipped_clusters),
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form of the diff (useful for history entries
        and CLI output)."""
        return {
            "added_nodes": list(self.added_nodes),
            "merged_nodes": [list(p) for p in self.merged_nodes],
            "skipped_nodes": [list(p) for p in self.skipped_nodes],
            "added_edges": [list(t) for t in self.added_edges],
            "skipped_edges": [list(t) for t in self.skipped_edges],
            "rewritten_edges": [list(t) for t in self.rewritten_edges],
            "added_clusters": list(self.added_clusters),
            "skipped_clusters": list(self.skipped_clusters),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Timezone-aware ISO-8601 timestamp; canonical for history entries."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _normalize_relation_type(rtype: str, case_insensitive: bool) -> str:
    if not rtype:
        return ""
    rtype = rtype.strip()
    return rtype.lower() if case_insensitive else rtype


def _node_key(name: str, etype: str | None, *, normalize: bool) -> tuple[str, str]:
    """Canonical lookup key for matching ``(canonical name, type)``.

    When ``normalize=True`` we run :func:`normalize_entity_name` (handles
    case + honorifics + whitespace). When ``normalize=False`` we use the
    name verbatim — callers who opt out of normalization usually want
    *byte-exact* id equality rather than a lighter touch, otherwise the
    flag would be a half-feature.
    """
    canon = normalize_entity_name(name) if normalize else name
    return canon, (etype or "")


# ---------------------------------------------------------------------------
# GraphMerger
# ---------------------------------------------------------------------------


class GraphMerger:
    """Merge an "incoming" :class:`EnhancedKG` into a "base" KG in-place.

    The merger is stateless across ``merge`` calls — the only state it
    carries is the strategy. That's deliberate: callers can hold a single
    ``GraphMerger`` instance and feed it many documents.

    Instances are safe to reuse but **not** thread-safe; if you fan out
    across threads, give each worker its own merger.
    """

    def __init__(self, strategy: MergeStrategy | None = None) -> None:
        self.strategy = strategy or MergeStrategy.default()
        # Per-call scratch slot used by `_merge_edges` / `_merge_nodes` to
        # stamp the active document_id onto incoming elements that don't
        # already carry one. Reset at the start of every `merge()` call so
        # the merger remains stateless across calls.
        self._document_id: str | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def merge(
        self,
        base: EnhancedKG,
        incoming: EnhancedKG,
        *,
        document_id: str | None = None,
        record_history: bool = True,
    ) -> KGDiff:
        """Fold ``incoming`` into ``base`` and return a diff report.

        ``base`` is mutated in place; ``incoming`` is left untouched.
        When ``record_history=True`` (the default), the merger:

        - bumps ``base.metadata['version']`` (initialising it to ``1``
          when the base graph has never been touched by the incremental
          layer);
        - sets ``base.metadata['updated_at']`` (and
          ``created_at`` on first touch);
        - appends a fresh entry to ``base.metadata['history']`` capturing
          the diff summary plus optional ``document_id``.

        Pass ``record_history=False`` to silently merge — useful for
        tests and for callers that maintain their own version metadata.
        """
        diff = KGDiff()
        if not isinstance(base, EnhancedKG) or not isinstance(incoming, EnhancedKG):
            raise TypeError("GraphMerger.merge() requires EnhancedKG instances on both sides")

        # Scope the active document_id for this merge call. Used by
        # `_merge_nodes` / `_merge_edges` to stamp `source_ref` and
        # `source_documents` for downstream multi-document inference. The
        # try/finally guarantees the scratch slot is cleared even on
        # error so the merger remains safe to reuse.
        previous_doc_id = self._document_id
        self._document_id = document_id
        try:
            return self._merge_inner(
                base,
                incoming,
                diff=diff,
                document_id=document_id,
                record_history=record_history,
            )
        finally:
            self._document_id = previous_doc_id

    def _merge_inner(
        self,
        base: EnhancedKG,
        incoming: EnhancedKG,
        *,
        diff: KGDiff,
        document_id: str | None,
        record_history: bool,
    ) -> KGDiff:
        # 1) Index the base graph for fast lookups.
        base_index = self._build_node_index(base)

        # 2) Walk incoming nodes; build incoming -> canonical (existing) id map.
        id_remap = self._merge_nodes(base, incoming, base_index, diff)

        # 3) Rewrite edges through the id remap and merge them.
        self._merge_edges(base, incoming, id_remap, diff)

        # 4) Copy clusters that don't already exist (id-equal collisions
        #    are skipped; renaming is out of scope).
        self._merge_clusters(base, incoming, id_remap, diff)

        # 5) Optional graph-level metadata bookkeeping.
        if record_history:
            self._update_metadata(base, diff, document_id=document_id)

        logger.info(
            "Incremental merge: %s",
            ", ".join(f"{k}={v}" for k, v in diff.summary().items() if v),
        )
        return diff

    # ------------------------------------------------------------------
    # Phase 1: node index
    # ------------------------------------------------------------------

    def _build_node_index(self, base: EnhancedKG) -> dict[tuple[str, str], str]:
        """Map ``(normalized name, type)`` -> existing canonical node id.

        Built fresh per ``merge`` call so the merger doesn't carry stale
        state across documents (and so callers who mutate ``base`` between
        calls don't see ghost matches).
        """
        index: dict[tuple[str, str], str] = {}
        norm = self.strategy.use_normalized_match
        for node_id, node in base.nodes.items():
            key = _node_key(node_id, node.type, normalize=norm)
            # First-write-wins: if two existing nodes happen to share a
            # normalized key, we don't try to second-guess the base graph.
            index.setdefault(key, node_id)
        return index

    # ------------------------------------------------------------------
    # Phase 2: nodes
    # ------------------------------------------------------------------

    def _merge_nodes(
        self,
        base: EnhancedKG,
        incoming: EnhancedKG,
        base_index: dict[tuple[str, str], str],
        diff: KGDiff,
    ) -> dict[str, str]:
        """Merge incoming nodes; return a remap of incoming id -> base id."""
        remap: dict[str, str] = {}
        norm = self.strategy.use_normalized_match

        for inc_id, inc_node in incoming.nodes.items():
            # Tier 1: exact id match — cheapest, always preferred.
            if inc_id in base.nodes:
                remap[inc_id] = inc_id
                self._fold_node_data(base.nodes[inc_id], inc_node)
                diff.merged_nodes.append((inc_id, inc_id))
                continue

            # Tier 2: type-aware normalized match.
            key = _node_key(inc_id, inc_node.type, normalize=norm)
            existing_id = base_index.get(key)

            # Tier 3: name-only match across types — only consulted when
            # type-match is not required (caller opted in).
            if existing_id is None and not self.strategy.require_type_match:
                # Try every (norm-name, *) key that matches the name part.
                norm_name = key[0]
                for (cand_name, _ctype), cand_id in base_index.items():
                    if cand_name == norm_name:
                        existing_id = cand_id
                        break

            if existing_id is None:
                # Type mismatch with same normalized name? Record the
                # skip so callers see why the merger refused.
                if self.strategy.require_type_match:
                    norm_name = key[0]
                    type_conflict = any(
                        cand_name == norm_name for (cand_name, _ctype) in base_index.keys()
                    )
                    if type_conflict:
                        diff.skipped_nodes.append((inc_id, "type_mismatch"))
                # Insert as a brand-new node, preserving the incoming id.
                new_metadata = dict(inc_node.metadata)
                # Backfill source_documents with the active document_id
                # when missing — keeps node-level provenance aligned with
                # the edge-level source_ref stamping (see _merge_edges).
                if self._document_id and "source_documents" not in new_metadata:
                    new_metadata["source_documents"] = [self._document_id]
                new_node = KGNode(
                    id=inc_node.id,
                    type=inc_node.type,
                    properties=dict(inc_node.properties),
                    metadata=new_metadata,
                    embedding=list(inc_node.embedding) if inc_node.embedding else None,
                    confidence=inc_node.confidence,
                )
                base.nodes[new_node.id] = new_node
                base_index[_node_key(new_node.id, new_node.type, normalize=norm)] = new_node.id
                remap[inc_id] = new_node.id
                diff.added_nodes.append(new_node.id)
                continue

            # Match found by canonical key.
            self._fold_node_data(base.nodes[existing_id], inc_node)
            remap[inc_id] = existing_id
            diff.merged_nodes.append((existing_id, inc_id))

        return remap

    def _union_source_documents(self, existing: KGNode, incoming: KGNode) -> None:
        """Append source-document hints to ``existing.metadata['source_documents']``.

        Sources considered (deduped, order-preserving):

        1. The active ``self._document_id`` (set per merge call).
        2. The incoming node's own ``source_documents`` list, if any.

        Touches metadata in place; never mutates other node fields.
        """
        merged: list[str] = list(existing.metadata.get("source_documents", []) or [])
        added = False
        if self._document_id and self._document_id not in merged:
            merged.append(self._document_id)
            added = True
        for doc in incoming.metadata.get("source_documents", []) or []:
            if doc and doc not in merged:
                merged.append(doc)
                added = True
        if added:
            existing.metadata["source_documents"] = merged

    def _fold_node_data(self, existing: KGNode, incoming: KGNode) -> None:
        """Mutate ``existing`` according to the configured node policy."""
        policy = self.strategy.node_policy

        # Document-level provenance is policy-independent: every merge
        # event extends the existing node's `source_documents` list with
        # the active document_id (and any incoming nodes' lists). This
        # gives multi-document reasoning a node-level view of which
        # documents touched a given entity.
        self._union_source_documents(existing, incoming)

        if policy is NodeMergePolicy.PREFER_EXISTING:
            # Record provenance only — never mutate the existing node's
            # primary fields.
            merged_from = existing.metadata.setdefault("merged_from", [])
            entry = {
                "id": incoming.id,
                "type": incoming.type,
            }
            if incoming.properties:
                entry["properties"] = dict(incoming.properties)
            if incoming.metadata:
                # Drop nested merged_from to avoid pathological growth.
                clean = {k: v for k, v in incoming.metadata.items() if k != "merged_from"}
                if clean:
                    entry["metadata"] = clean
            if entry not in merged_from:
                merged_from.append(entry)
            return

        if policy is NodeMergePolicy.PREFER_NEW:
            existing.type = incoming.type if incoming.type is not None else existing.type
            if incoming.properties:
                existing.properties = dict(incoming.properties)
            if incoming.metadata:
                existing.metadata = dict(incoming.metadata)
            if incoming.embedding:
                existing.embedding = list(incoming.embedding)
            if incoming.confidence is not None:
                existing.confidence = incoming.confidence
            return

        if policy is NodeMergePolicy.UNION:
            # Properties / metadata: shallow merge with incoming winning.
            if incoming.properties:
                merged_props = dict(existing.properties)
                merged_props.update(incoming.properties)
                existing.properties = merged_props
            if incoming.metadata:
                merged_meta = dict(existing.metadata)
                merged_meta.update(incoming.metadata)
                existing.metadata = merged_meta
            # Embedding: average element-wise iff dimensions match.
            if incoming.embedding:
                if existing.embedding and len(existing.embedding) == len(incoming.embedding):
                    existing.embedding = [
                        (a + b) / 2.0
                        for a, b in zip(existing.embedding, incoming.embedding, strict=True)
                    ]
                elif not existing.embedding:
                    existing.embedding = list(incoming.embedding)
            # Confidence: keep the higher.
            if incoming.confidence is not None:
                if existing.confidence is None or incoming.confidence > existing.confidence:
                    existing.confidence = incoming.confidence
            return

    # ------------------------------------------------------------------
    # Phase 3: edges
    # ------------------------------------------------------------------

    def _merge_edges(
        self,
        base: EnhancedKG,
        incoming: EnhancedKG,
        id_remap: dict[str, str],
        diff: KGDiff,
    ) -> None:
        case_insensitive = self.strategy.case_insensitive_relation

        # Build canonical edge index over base graph.
        base_edge_index: dict[tuple[str, str, str], int] = {}
        for i, edge in enumerate(base.edges):
            key = (
                edge.source,
                _normalize_relation_type(edge.relationship_type, case_insensitive),
                edge.target,
            )
            base_edge_index.setdefault(key, i)

        for edge in incoming.edges:
            new_source = id_remap.get(edge.source, edge.source)
            new_target = id_remap.get(edge.target, edge.target)
            rewritten = (new_source, new_target) != (edge.source, edge.target)

            # Self-loops can appear when both endpoints map to the same
            # canonical node — drop them; KGEdge.__post_init__ would
            # raise otherwise.
            if new_source == new_target:
                diff.skipped_edges.append((new_source, edge.relationship_type, new_target))
                continue

            # Endpoints must exist in the base graph at this point.
            # (The node phase guarantees this for any node that came
            # through ``incoming.nodes``; an edge whose endpoint was
            # missing on the incoming side gets added here defensively.)
            if new_source not in base.nodes:
                base.nodes[new_source] = KGNode(id=new_source)
                diff.added_nodes.append(new_source)
            if new_target not in base.nodes:
                base.nodes[new_target] = KGNode(id=new_target)
                diff.added_nodes.append(new_target)

            key = (
                new_source,
                _normalize_relation_type(edge.relationship_type, case_insensitive),
                new_target,
            )

            if key in base_edge_index:
                existing = base.edges[base_edge_index[key]]
                self._fold_edge_data(existing, edge)
                diff.skipped_edges.append((new_source, edge.relationship_type, new_target))
                continue

            # Carry the document_id forward as an edge-level source_ref so
            # multi-document reasoning rules (`drg.reasoning`) can later
            # tell apart edges that came from different documents. The
            # incoming edge's own `source_ref` always wins — the document_id
            # is only a fallback for edges that don't carry one yet.
            new_metadata = dict(edge.metadata)
            if self._document_id and "source_ref" not in new_metadata:
                new_metadata["source_ref"] = self._document_id

            new_edge = KGEdge(
                source=new_source,
                target=new_target,
                relationship_type=edge.relationship_type,
                relationship_detail=edge.relationship_detail,
                metadata=new_metadata,
                start_time=edge.start_time,
                end_time=edge.end_time,
                confidence=edge.confidence,
                is_negated=edge.is_negated,
            )
            base.edges.append(new_edge)
            base_edge_index[key] = len(base.edges) - 1
            diff.added_edges.append((new_source, edge.relationship_type, new_target))
            if rewritten:
                diff.rewritten_edges.append((new_source, edge.relationship_type, new_target))

    def _fold_edge_data(self, existing: KGEdge, incoming: KGEdge) -> None:
        """Apply the configured edge policy to a duplicate edge."""
        policy = self.strategy.edge_policy

        if policy is EdgeMergePolicy.SKIP:
            return

        # Both APPEND_EVIDENCE and MAX_CONFIDENCE want to remember the
        # duplicate's evidence; track it under a single canonical key.
        evidence_refs = existing.metadata.setdefault("evidence_refs", [])
        ref_entry: dict[str, Any] = {}
        if incoming.metadata.get("source_ref"):
            ref_entry["source_ref"] = incoming.metadata["source_ref"]
        if incoming.metadata.get("evidence"):
            ref_entry["evidence"] = incoming.metadata["evidence"]
        if incoming.confidence is not None:
            ref_entry["confidence"] = incoming.confidence
        if ref_entry and ref_entry not in evidence_refs:
            evidence_refs.append(ref_entry)

        if policy is EdgeMergePolicy.MAX_CONFIDENCE:
            inc_c = incoming.confidence
            ex_c = existing.confidence
            if inc_c is not None and (ex_c is None or inc_c > ex_c):
                # Bump confidence and remember the previous value.
                alt = existing.metadata.setdefault("alt_confidences", [])
                if ex_c is not None and ex_c not in alt:
                    alt.append(ex_c)
                existing.confidence = inc_c

    # ------------------------------------------------------------------
    # Phase 4: clusters
    # ------------------------------------------------------------------

    def _merge_clusters(
        self,
        base: EnhancedKG,
        incoming: EnhancedKG,
        id_remap: dict[str, str],
        diff: KGDiff,
    ) -> None:
        for cluster_id, cluster in incoming.clusters.items():
            if cluster_id in base.clusters:
                diff.skipped_clusters.append(cluster_id)
                continue
            remapped = {id_remap.get(nid, nid) for nid in cluster.node_ids}
            valid = {nid for nid in remapped if nid in base.nodes}
            if not valid:
                diff.skipped_clusters.append(cluster_id)
                continue
            base.clusters[cluster_id] = Cluster(
                id=cluster.id,
                node_ids=valid,
                metadata=dict(cluster.metadata),
            )
            diff.added_clusters.append(cluster_id)

    # ------------------------------------------------------------------
    # Phase 5: graph-level metadata
    # ------------------------------------------------------------------

    def _update_metadata(
        self,
        base: EnhancedKG,
        diff: KGDiff,
        *,
        document_id: str | None,
    ) -> None:
        meta = base.metadata
        now = _utc_now_iso()
        if not meta:
            meta["created_at"] = now
            meta["version"] = 1
        else:
            meta["version"] = int(meta.get("version", 1)) + 1

        meta["updated_at"] = now

        history: list[dict[str, Any]] = meta.setdefault("history", [])
        entry: dict[str, Any] = {
            "version": meta["version"],
            "operation": "merge",
            "timestamp": now,
            **diff.summary(),
        }
        if document_id:
            entry["document_id"] = document_id
        history.append(entry)


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def merge_graphs(
    base: EnhancedKG,
    incoming: EnhancedKG,
    *,
    strategy: MergeStrategy | None = None,
    document_id: str | None = None,
    record_history: bool = True,
) -> KGDiff:
    """One-call convenience wrapper around :class:`GraphMerger`.

    Equivalent to::

        GraphMerger(strategy).merge(
            base, incoming,
            document_id=document_id,
            record_history=record_history,
        )

    Useful when the caller doesn't need to keep the merger around (e.g. a
    one-off CLI invocation).
    """
    return GraphMerger(strategy).merge(
        base,
        incoming,
        document_id=document_id,
        record_history=record_history,
    )
