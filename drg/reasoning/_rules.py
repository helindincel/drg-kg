"""Built-in inference rules for multi-document reasoning.

Each rule is a deterministic, side-effect-free transformation of the
input :class:`drg.graph.kg_core.EnhancedKG` into a list of
:class:`InferredEdge` candidates. The engine handles dedup, idempotency
and persistence — rules only need to express "what".

Rule catalog
============

* :class:`PathBridgeRule` — the headline multi-document rule. Given a
  shared "bridge" entity that appears in edges from **different
  documents**, propose a ``connected_via_<bridge>`` edge between the
  two outer endpoints.
* :class:`InverseRule` — propose the schema-defined inverse for a small
  set of stable inverse pairs (``founded`` ↔ ``founded_by``).
* :class:`SymmetricRule` — propose the back-edge for symmetric
  predicates (``works_with``, ``collaborates_with``).
* :class:`TransitiveRule` — propose the transitive closure step for a
  small whitelist of transitive predicates (``part_of``, ``subclass_of``,
  ``located_in``).
* :class:`CompositionRule` — propose ``operates_in(A, L)`` when
  ``owns(A, B)`` and ``located_in(B, L)`` co-exist (graph-only;
  text-free, complementing the existing extract-time heuristic).

Every rule that fires attaches **complete provenance** — the source
``EvidenceLink``s, the documents they came from, and an
:class:`drg.reasoning._explain`-friendly explanation string.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from ._explain import (
    explain_composition,
    explain_inverse,
    explain_path_bridge,
    explain_symmetric,
    explain_transitive,
)
from ._types import EvidenceLink, InferenceRule, InferredEdge, ReasoningConfig

if TYPE_CHECKING:
    from ..graph.kg_core import EnhancedKG, KGEdge

__all__ = [
    "CompositionRule",
    "InverseRule",
    "PathBridgeRule",
    "SymmetricRule",
    "TransitiveRule",
    "default_rules",
]


# ---------------------------------------------------------------------------
# Helpers shared across rules
# ---------------------------------------------------------------------------


def _edge_source_ref(edge: KGEdge) -> str | None:
    """Read the document/chunk identifier off an edge's metadata.

    The merger and the (new) ``build_enhanced_kg(document_id=...)``
    path both write under ``metadata['source_ref']``; older edges
    without this field surface as ``None`` and rules treat them as
    "unknown document" rather than guessing.
    """
    if not edge.metadata:
        return None
    ref = edge.metadata.get("source_ref")
    return ref if isinstance(ref, str) and ref else None


def _is_inferred_edge(edge: KGEdge) -> bool:
    return bool(edge.metadata) and bool(edge.metadata.get("inferred"))


def _link_from_edge(edge: KGEdge) -> EvidenceLink:
    return EvidenceLink(
        triple=(edge.source, edge.relationship_type, edge.target),
        source_ref=_edge_source_ref(edge),
        confidence=edge.confidence,
        is_inferred=_is_inferred_edge(edge),
    )


def _candidate_edges(
    kg: EnhancedKG,
    config: ReasoningConfig,
) -> list[KGEdge]:
    """Return edges eligible as evidence under the current config.

    By default this filters out previously-inferred edges so rules
    can't chain off themselves (avoids confidence inflation and
    runaway inference). Opt back in with
    :attr:`ReasoningConfig.allow_inferred_in_evidence`.
    """
    if config.allow_inferred_in_evidence:
        return list(kg.edges)
    return [e for e in kg.edges if not _is_inferred_edge(e)]


def _slug(value: str) -> str:
    """Filename-friendly snake_case slug — used to build the inferred
    relationship name (``connected_via_<bridge_slug>``)."""
    out: list[str] = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    while out and out[-1] == "_":
        out.pop()
    return "".join(out) or "entity"


# ---------------------------------------------------------------------------
# PathBridgeRule — the multi-document workhorse
# ---------------------------------------------------------------------------


class PathBridgeRule(InferenceRule):
    """Connect two entities that share a bridge node sourced from
    different documents.

    Concretely, given two evidence edges:

    - ``(A) --r1--> (B)``  with ``source_ref="doc_A"``
    - ``(C) --r2--> (B)``  with ``source_ref="doc_B"``

    emit an inferred edge ``A --connected_via_b--> C`` (and the
    symmetric direction is collapsed: we always emit the lexicographic
    smaller endpoint as the source so the rule is order-stable).

    Guarantees:

    - Fires **only** when the two evidence edges have distinct
      ``source_ref`` (true cross-document inference).
    - When :attr:`ReasoningConfig.require_distinct_bridge_relations` is
      true, also requires ``r1 != r2`` so a fact re-observed in two
      documents doesn't trigger spurious connections.
    - Confidence is the product of the evidence confidences, clipped to
      [:attr:`ReasoningConfig.bridge_confidence_floor`, 1.0].
    - Per-bridge fan-out is capped by
      :attr:`ReasoningConfig.max_bridge_candidates_per_node` to keep
      hub bridges (e.g. a city) from exploding the output.
    """

    name = "path_bridge"
    requires_cross_document = True

    def apply(
        self,
        kg: EnhancedKG,
        config: ReasoningConfig,
    ) -> list[InferredEdge]:
        edges = _candidate_edges(kg, config)
        if len(edges) < 2:
            return []

        # Index incident edges by endpoint. We treat "bridge" symmetrically:
        # any node that two edges *both* mention can be a bridge.
        bridge_map: dict[str, list[tuple[KGEdge, str]]] = defaultdict(list)
        for e in edges:
            bridge_map[e.source].append((e, "out"))
            bridge_map[e.target].append((e, "in"))

        candidates: list[InferredEdge] = []
        for bridge, incident in bridge_map.items():
            if len(incident) < 2:
                continue

            # Hard cap on per-bridge work. Order is stable (edges are
            # appended in graph-walk order) so the cap deterministically
            # picks the same incident edges across runs.
            local = incident[: config.max_bridge_candidates_per_node]
            for i in range(len(local)):
                e1, role1 = local[i]
                ref1 = _edge_source_ref(e1)
                if not ref1:
                    continue
                for j in range(i + 1, len(local)):
                    e2, role2 = local[j]
                    ref2 = _edge_source_ref(e2)
                    if not ref2:
                        continue
                    if ref1 == ref2:
                        continue  # same document — not multi-document inference
                    if (
                        config.require_distinct_bridge_relations
                        and e1.relationship_type.lower() == e2.relationship_type.lower()
                    ):
                        continue

                    outer1 = e1.target if role1 == "out" else e1.source
                    outer2 = e2.target if role2 == "out" else e2.source
                    if outer1 == outer2 or outer1 == bridge or outer2 == bridge:
                        continue

                    # Deterministic endpoint ordering (lexicographic on
                    # the lowercased id) so re-runs produce the same
                    # ``(source, target)`` direction.
                    if outer1.lower() <= outer2.lower():
                        src, dst = outer1, outer2
                    else:
                        src, dst = outer2, outer1

                    c1 = e1.confidence if e1.confidence is not None else 0.8
                    c2 = e2.confidence if e2.confidence is not None else 0.8
                    conf = max(
                        config.bridge_confidence_floor,
                        min(1.0, c1 * c2),
                    )

                    chain = [_link_from_edge(e1), _link_from_edge(e2)]
                    candidates.append(
                        InferredEdge(
                            source=src,
                            target=dst,
                            relationship_type=f"connected_via_{_slug(bridge)}",
                            rule_name=self.name,
                            evidence_chain=chain,
                            explanation=explain_path_bridge(
                                src=src,
                                dst=dst,
                                bridge=bridge,
                                evidence=chain,
                            ),
                            confidence=conf,
                            bridge_entity=bridge,
                        )
                    )

        return candidates


# ---------------------------------------------------------------------------
# InverseRule — close inverse-relation pairs
# ---------------------------------------------------------------------------


# Conservative inverse pairs. Each direction is generated independently so
# the rule fires either way — but only for relation names that are stable
# inverses across schemas. New pairs can be added without changing any rule
# code.
INVERSE_RELATION_PAIRS: dict[str, str] = {
    "founded": "founded_by",
    "founded_by": "founded",
    "owns": "owned_by",
    "owned_by": "owns",
    "acquired": "acquired_by",
    "acquired_by": "acquired",
    "produces": "produced_by",
    "produced_by": "produces",
    "manufactures": "manufactured_by",
    "manufactured_by": "manufactures",
    "employs": "employed_by",
    "employed_by": "employs",
    "parent_of": "child_of",
    "child_of": "parent_of",
    "contains": "part_of",
    "part_of": "contains",
    "has_part": "part_of",
}


class InverseRule(InferenceRule):
    """Emit the inverse of an existing edge when its inverse is missing.

    Operates purely on relation names from :data:`INVERSE_RELATION_PAIRS`;
    no language model, no schema introspection. Confidence is identical
    to the source edge.

    This rule is **not** cross-document specific — its main value in a
    multi-document setting is that after merging, it completes the
    inverse view across documents that only ever stated one direction.
    """

    name = "inverse"
    requires_cross_document = False

    def apply(
        self,
        kg: EnhancedKG,
        config: ReasoningConfig,
    ) -> list[InferredEdge]:
        edges = _candidate_edges(kg, config)
        if not edges:
            return []

        existing = {(e.source, e.relationship_type.lower(), e.target) for e in kg.edges}
        candidates: list[InferredEdge] = []
        for e in edges:
            inverse = INVERSE_RELATION_PAIRS.get(e.relationship_type.lower())
            if not inverse:
                continue
            key = (e.target, inverse, e.source)
            if key in existing:
                continue
            link = _link_from_edge(e)
            conf = e.confidence if e.confidence is not None else 0.85
            candidates.append(
                InferredEdge(
                    source=e.target,
                    target=e.source,
                    relationship_type=inverse,
                    rule_name=self.name,
                    evidence_chain=[link],
                    explanation=explain_inverse(
                        source=e.target,
                        relation=inverse,
                        target=e.source,
                        original=link,
                    ),
                    confidence=conf,
                )
            )
        return candidates


# ---------------------------------------------------------------------------
# SymmetricRule — close symmetric-relation pairs
# ---------------------------------------------------------------------------


SYMMETRIC_RELATIONS: frozenset[str] = frozenset(
    {
        "works_with",
        "collaborates_with",
        "married_to",
        "sibling_of",
        "similar_to",
        "related_to",
        "near",
        "communicates_with",
    }
)


class SymmetricRule(InferenceRule):
    """Emit the back-edge for symmetric predicates.

    Stricter than :class:`InverseRule` because the relation name does
    not change — we just add the missing ``(target, rel, source)``
    direction. Confidence is identical to the source edge.
    """

    name = "symmetric"
    requires_cross_document = False

    def apply(
        self,
        kg: EnhancedKG,
        config: ReasoningConfig,
    ) -> list[InferredEdge]:
        edges = _candidate_edges(kg, config)
        if not edges:
            return []

        existing = {(e.source, e.relationship_type.lower(), e.target) for e in kg.edges}
        candidates: list[InferredEdge] = []
        for e in edges:
            rel_lower = e.relationship_type.lower()
            if rel_lower not in SYMMETRIC_RELATIONS:
                continue
            key = (e.target, rel_lower, e.source)
            if key in existing:
                continue
            link = _link_from_edge(e)
            conf = e.confidence if e.confidence is not None else 0.9
            candidates.append(
                InferredEdge(
                    source=e.target,
                    target=e.source,
                    relationship_type=e.relationship_type,
                    rule_name=self.name,
                    evidence_chain=[link],
                    explanation=explain_symmetric(
                        source=e.target,
                        relation=e.relationship_type,
                        target=e.source,
                        original=link,
                    ),
                    confidence=conf,
                )
            )
        return candidates


# ---------------------------------------------------------------------------
# TransitiveRule — single-hop transitive closure
# ---------------------------------------------------------------------------


TRANSITIVE_RELATIONS: frozenset[str] = frozenset(
    {
        "part_of",
        "subclass_of",
        "located_in",
        "contains",
        "ancestor_of",
        "descendant_of",
    }
)


class TransitiveRule(InferenceRule):
    """Apply single-step transitive closure for whitelisted predicates.

    Given ``A --rel--> B`` and ``B --rel--> C`` with ``rel`` in
    :data:`TRANSITIVE_RELATIONS`, emit ``A --rel--> C``. Cross-document
    chains are particularly common here (e.g. doc 1 says
    ``Paris part_of France``, doc 2 says ``Eiffel Tower part_of Paris``).

    The rule applies **one** hop per ``reason()`` invocation. Callers
    needing the full closure can call ``reason()`` repeatedly until
    :class:`InferenceReport.is_empty()` returns ``True``; the engine's
    idempotency check makes this safe.
    """

    name = "transitive"
    requires_cross_document = False

    def apply(
        self,
        kg: EnhancedKG,
        config: ReasoningConfig,
    ) -> list[InferredEdge]:
        edges = _candidate_edges(kg, config)
        if len(edges) < 2:
            return []

        by_source: dict[tuple[str, str], list[KGEdge]] = defaultdict(list)
        existing: set[tuple[str, str, str]] = set()
        for e in kg.edges:
            existing.add((e.source, e.relationship_type.lower(), e.target))
        for e in edges:
            rel = e.relationship_type.lower()
            if rel in TRANSITIVE_RELATIONS:
                by_source[(e.source, rel)].append(e)

        candidates: list[InferredEdge] = []
        for e in edges:
            rel = e.relationship_type.lower()
            if rel not in TRANSITIVE_RELATIONS:
                continue
            for next_edge in by_source.get((e.target, rel), []):
                if next_edge.target == e.source:
                    continue
                key = (e.source, rel, next_edge.target)
                if key in existing:
                    continue
                # De-duplicate within the same run.
                existing.add(key)

                l1 = _link_from_edge(e)
                l2 = _link_from_edge(next_edge)
                c1 = e.confidence if e.confidence is not None else 0.85
                c2 = next_edge.confidence if next_edge.confidence is not None else 0.85
                conf = max(0.5, min(1.0, c1 * c2))
                candidates.append(
                    InferredEdge(
                        source=e.source,
                        target=next_edge.target,
                        relationship_type=e.relationship_type,
                        rule_name=self.name,
                        evidence_chain=[l1, l2],
                        explanation=explain_transitive(
                            head=e.source,
                            mid=e.target,
                            tail=next_edge.target,
                            relation=e.relationship_type,
                        ),
                        confidence=conf,
                        extra_metadata={"intermediate": e.target},
                    )
                )
        return candidates


# ---------------------------------------------------------------------------
# CompositionRule — owner+location → owner-located
# ---------------------------------------------------------------------------


OWNERSHIP_RELATIONS_COMPOSITION: frozenset[str] = frozenset(
    {"owns", "has_part", "operates", "controls"}
)
LOCATION_RELATIONS_COMPOSITION: frozenset[str] = frozenset(
    {"located_in", "located_at", "headquartered_in", "based_in"}
)


class CompositionRule(InferenceRule):
    """Graph-only complement to ``_infer_implicit_relations``'s two-hop step.

    Given ``A --owns--> B`` and ``B --located_in--> L``, emit
    ``A --operates_in--> L``. The text-based extract-time heuristic
    requires the source text to be present and contain "operation
    cues"; this graph-only version fires after the merge so it can
    bridge documents that contributed the two evidence edges
    independently. Confidence is the product of the evidence edges'
    confidences, clipped to [0.5, 1.0].
    """

    name = "composition"
    requires_cross_document = False

    def apply(
        self,
        kg: EnhancedKG,
        config: ReasoningConfig,
    ) -> list[InferredEdge]:
        edges = _candidate_edges(kg, config)
        if len(edges) < 2:
            return []

        owns_index: dict[str, list[KGEdge]] = defaultdict(list)
        locs_index: dict[str, list[KGEdge]] = defaultdict(list)
        existing: set[tuple[str, str, str]] = set()
        for e in kg.edges:
            existing.add((e.source, e.relationship_type.lower(), e.target))
        for e in edges:
            rel = e.relationship_type.lower()
            if rel in OWNERSHIP_RELATIONS_COMPOSITION:
                owns_index[e.source].append(e)
            if rel in LOCATION_RELATIONS_COMPOSITION:
                locs_index[e.source].append(e)

        candidates: list[InferredEdge] = []
        for owner, owns_edges in owns_index.items():
            for own_edge in owns_edges:
                asset = own_edge.target
                for loc_edge in locs_index.get(asset, []):
                    loc = loc_edge.target
                    if loc == owner:
                        continue
                    key = (owner, "operates_in", loc)
                    if key in existing:
                        continue
                    existing.add(key)

                    l1 = _link_from_edge(own_edge)
                    l2 = _link_from_edge(loc_edge)
                    c1 = own_edge.confidence if own_edge.confidence is not None else 0.85
                    c2 = loc_edge.confidence if loc_edge.confidence is not None else 0.85
                    conf = max(0.5, min(1.0, c1 * c2))

                    candidates.append(
                        InferredEdge(
                            source=owner,
                            target=loc,
                            relationship_type="operates_in",
                            rule_name=self.name,
                            evidence_chain=[l1, l2],
                            explanation=explain_composition(
                                owner=owner,
                                asset=asset,
                                location=loc,
                            ),
                            confidence=conf,
                            extra_metadata={"intermediate": asset},
                        )
                    )
        return candidates


# ---------------------------------------------------------------------------
# Default rule bundle
# ---------------------------------------------------------------------------


def default_rules() -> list[InferenceRule]:
    """Return the default rule set used by :class:`MultiDocumentReasoner`
    when no explicit rules are passed.

    Order matters slightly: rules earlier in the list are applied
    first, so their inferences are visible to later rules via the
    engine's per-rule reapplication (when
    :attr:`ReasoningConfig.allow_inferred_in_evidence` is true).
    """
    return [
        InverseRule(),
        SymmetricRule(),
        TransitiveRule(),
        CompositionRule(),
        PathBridgeRule(),
    ]
