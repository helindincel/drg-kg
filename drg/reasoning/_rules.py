"""Built-in inference rules for multi-document reasoning.

Rules
-----
PathBridgeRule
    Infers cross-document bridges: if A→B in doc-1 and B→C in doc-2,
    creates A→C with a ``connected_via`` relationship.
InverseRule
    If A→B with a relation that has a known inverse, infers B→A with the
    inverse relation.
SymmetricRule
    If A→B with a symmetric relation (e.g. ``related_to``), infers B→A
    with the same relation.
TransitiveRule
    For transitive relations (e.g. ``part_of``): if A→B and B→C both hold,
    infers A→C.
CompositionRule
    Composes two relations: if A ``owns`` B and B ``located_in`` C, infers
    A ``operates_in`` C.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import EvidenceLink, InferenceRule, InferredEdge

if TYPE_CHECKING:
    from drg.graph.kg_core import EnhancedKG

__all__ = [
    "CompositionRule",
    "InverseRule",
    "PathBridgeRule",
    "SymmetricRule",
    "TransitiveRule",
]

# ---------------------------------------------------------------------------
# Relation sets (defaults — users can override via subclassing)
# ---------------------------------------------------------------------------

_SYMMETRIC_RELATIONS: frozenset[str] = frozenset(
    {
        "related_to",
        "associated_with",
        "sibling_of",
        "co-occurs_with",
        "allied_with",
        "married_to",
        "collaborated_with",
        "competes_with",
    }
)

_TRANSITIVE_RELATIONS: frozenset[str] = frozenset(
    {
        "part_of",
        "subset_of",
        "subclass_of",
        "is_a",
        "located_in",
        "contained_in",
        "owned_by",
        "reports_to",
    }
)

# (first_relation, second_relation) → inferred_relation
_COMPOSITION_MAP: dict[tuple[str, str], str] = {
    ("owns", "located_in"): "operates_in",
    ("subsidiary_of", "located_in"): "operates_in",
    ("works_for", "located_in"): "works_in",
    ("member_of", "located_in"): "active_in",
    ("founded_by", "born_in"): "founder_from",
}

# relation → its inverse
_INVERSE_MAP: dict[str, str] = {
    "parent_of": "child_of",
    "child_of": "parent_of",
    "owns": "owned_by",
    "owned_by": "owns",
    "employs": "employed_by",
    "employed_by": "employs",
    "founded": "founded_by",
    "founded_by": "founded",
    "located_in": "location_of",
    "location_of": "located_in",
    "produces": "produced_by",
    "produced_by": "produces",
    "leads": "led_by",
    "led_by": "leads",
    "part_of": "has_part",
    "has_part": "part_of",
    "member_of": "has_member",
    "has_member": "member_of",
    "acquired": "acquired_by",
    "acquired_by": "acquired",
}


def _edge_doc_id(edge) -> str | None:
    """Extract document_id from an edge's metadata (best-effort)."""
    meta = getattr(edge, "metadata", {}) or {}
    prov = meta.get("provenance") or {}
    if isinstance(prov, dict):
        return prov.get("document_id") or meta.get("source_ref")
    return meta.get("source_ref")


def _edge_confidence(edge) -> float:
    conf = getattr(edge, "confidence", None)
    return conf if conf is not None else 0.5


class PathBridgeRule(InferenceRule):
    """Infer cross-document bridges: A→B + B→C ⟹ A→C (connected_via)."""

    name = "path_bridge"

    def apply(
        self,
        kg: EnhancedKG,
        *,
        document_id: str | None = None,
    ) -> list[InferredEdge]:
        results: list[InferredEdge] = []
        edges = list(kg.edges) if hasattr(kg, "edges") else []
        # Build adjacency: target → list of edges ending there
        by_target: dict[str, list] = {}
        for edge in edges:
            by_target.setdefault(edge.target, []).append(edge)

        seen: set[tuple[str, str, str]] = set()
        for edge_ab in edges:
            # Only bridge across documents when document_id filtering is active
            if document_id is not None:
                doc_ab = _edge_doc_id(edge_ab)
                if doc_ab == document_id:
                    continue  # skip same-document edges as "first hop"
            for edge_bc in by_target.get(edge_ab.source, []):
                if edge_bc.source == edge_ab.target:
                    continue
                key = (edge_bc.source, "connected_via", edge_ab.target)
                if key in seen:
                    continue
                seen.add(key)
                conf = round(_edge_confidence(edge_ab) * _edge_confidence(edge_bc) * 0.8, 4)
                evidence = [
                    EvidenceLink(
                        source_node=edge_bc.source,
                        target_node=edge_bc.target,
                        relationship_type=edge_bc.relationship_type,
                        document_id=_edge_doc_id(edge_bc),
                        confidence=_edge_confidence(edge_bc),
                    ),
                    EvidenceLink(
                        source_node=edge_ab.source,
                        target_node=edge_ab.target,
                        relationship_type=edge_ab.relationship_type,
                        document_id=_edge_doc_id(edge_ab),
                        confidence=_edge_confidence(edge_ab),
                    ),
                ]
                results.append(
                    InferredEdge(
                        source=edge_bc.source,
                        target=edge_ab.target,
                        relationship_type="connected_via",
                        relationship_detail=(
                            f"{edge_bc.source} is connected to {edge_ab.target} "
                            f"via shared node {edge_ab.source}"
                        ),
                        confidence=conf,
                        rule_name=self.name,
                        evidence=evidence,
                        metadata={
                            "inferred": True,
                            "inference": {
                                "rule": self.name,
                                "bridge_node": edge_ab.source,
                            },
                        },
                    )
                )
        return results


class InverseRule(InferenceRule):
    """Infer inverse relations based on a known inverse map."""

    name = "inverse"

    def __init__(
        self,
        inverse_map: dict[str, str] | None = None,
    ) -> None:
        self._map = inverse_map if inverse_map is not None else _INVERSE_MAP

    def apply(
        self,
        kg: EnhancedKG,
        *,
        document_id: str | None = None,
    ) -> list[InferredEdge]:
        results: list[InferredEdge] = []
        edges = list(kg.edges) if hasattr(kg, "edges") else []
        seen: set[tuple[str, str, str]] = set()
        for edge in edges:
            inv_rel = self._map.get(edge.relationship_type)
            if inv_rel is None:
                continue
            key = (edge.target, inv_rel, edge.source)
            if key in seen:
                continue
            # Skip if this inverse already exists
            existing = [
                e
                for e in edges
                if e.source == edge.target
                and e.target == edge.source
                and e.relationship_type == inv_rel
            ]
            if existing:
                continue
            seen.add(key)
            conf = round(_edge_confidence(edge) * 0.9, 4)
            results.append(
                InferredEdge(
                    source=edge.target,
                    target=edge.source,
                    relationship_type=inv_rel,
                    relationship_detail=(
                        f"{edge.target} {inv_rel} {edge.source} "
                        f"(inverse of {edge.relationship_type})"
                    ),
                    confidence=conf,
                    rule_name=self.name,
                    evidence=[
                        EvidenceLink(
                            source_node=edge.source,
                            target_node=edge.target,
                            relationship_type=edge.relationship_type,
                            document_id=_edge_doc_id(edge),
                            confidence=_edge_confidence(edge),
                        )
                    ],
                    metadata={
                        "inferred": True,
                        "inference": {
                            "rule": self.name,
                            "original_relation": edge.relationship_type,
                        },
                    },
                )
            )
        return results


class SymmetricRule(InferenceRule):
    """Infer symmetric counterparts for symmetric relations."""

    name = "symmetric"

    def __init__(
        self,
        symmetric_relations: frozenset[str] | None = None,
    ) -> None:
        self._relations = (
            symmetric_relations if symmetric_relations is not None else _SYMMETRIC_RELATIONS
        )

    def apply(
        self,
        kg: EnhancedKG,
        *,
        document_id: str | None = None,
    ) -> list[InferredEdge]:
        results: list[InferredEdge] = []
        edges = list(kg.edges) if hasattr(kg, "edges") else []
        seen: set[tuple[str, str, str]] = set()
        for edge in edges:
            if edge.relationship_type not in self._relations:
                continue
            key = (edge.target, edge.relationship_type, edge.source)
            if key in seen:
                continue
            existing = [
                e
                for e in edges
                if e.source == edge.target
                and e.target == edge.source
                and e.relationship_type == edge.relationship_type
            ]
            if existing:
                continue
            seen.add(key)
            conf = round(_edge_confidence(edge) * 0.95, 4)
            results.append(
                InferredEdge(
                    source=edge.target,
                    target=edge.source,
                    relationship_type=edge.relationship_type,
                    relationship_detail=(
                        f"{edge.target} {edge.relationship_type} {edge.source} (symmetric)"
                    ),
                    confidence=conf,
                    rule_name=self.name,
                    evidence=[
                        EvidenceLink(
                            source_node=edge.source,
                            target_node=edge.target,
                            relationship_type=edge.relationship_type,
                            document_id=_edge_doc_id(edge),
                            confidence=_edge_confidence(edge),
                        )
                    ],
                    metadata={
                        "inferred": True,
                        "inference": {"rule": self.name},
                    },
                )
            )
        return results


class TransitiveRule(InferenceRule):
    """Infer transitive closures for transitive relations."""

    name = "transitive"

    def __init__(
        self,
        transitive_relations: frozenset[str] | None = None,
    ) -> None:
        self._relations = (
            transitive_relations if transitive_relations is not None else _TRANSITIVE_RELATIONS
        )

    def apply(
        self,
        kg: EnhancedKG,
        *,
        document_id: str | None = None,
    ) -> list[InferredEdge]:
        results: list[InferredEdge] = []
        edges = list(kg.edges) if hasattr(kg, "edges") else []
        seen: set[tuple[str, str, str]] = set()

        for rel in self._relations:
            rel_edges = [e for e in edges if e.relationship_type == rel]
            # Build forward map: A → {B, ...}
            fwd: dict[str, list] = {}
            for e in rel_edges:
                fwd.setdefault(e.source, []).append(e)
            for edge_ab in rel_edges:
                for edge_bc in fwd.get(edge_ab.target, []):
                    if edge_bc.target == edge_ab.source:
                        continue  # avoid cycles
                    key = (edge_ab.source, rel, edge_bc.target)
                    if key in seen:
                        continue
                    # Skip if already exists
                    exists = any(
                        e.source == edge_ab.source
                        and e.target == edge_bc.target
                        and e.relationship_type == rel
                        for e in rel_edges
                    )
                    if exists:
                        continue
                    seen.add(key)
                    conf = round(
                        _edge_confidence(edge_ab) * _edge_confidence(edge_bc) * 0.85,
                        4,
                    )
                    results.append(
                        InferredEdge(
                            source=edge_ab.source,
                            target=edge_bc.target,
                            relationship_type=rel,
                            relationship_detail=(
                                f"{edge_ab.source} {rel} {edge_bc.target} "
                                f"(transitive via {edge_ab.target})"
                            ),
                            confidence=conf,
                            rule_name=self.name,
                            evidence=[
                                EvidenceLink(
                                    source_node=edge_ab.source,
                                    target_node=edge_ab.target,
                                    relationship_type=rel,
                                    document_id=_edge_doc_id(edge_ab),
                                    confidence=_edge_confidence(edge_ab),
                                ),
                                EvidenceLink(
                                    source_node=edge_bc.source,
                                    target_node=edge_bc.target,
                                    relationship_type=rel,
                                    document_id=_edge_doc_id(edge_bc),
                                    confidence=_edge_confidence(edge_bc),
                                ),
                            ],
                            metadata={
                                "inferred": True,
                                "inference": {
                                    "rule": self.name,
                                    "via_node": edge_ab.target,
                                },
                            },
                        )
                    )
        return results


class CompositionRule(InferenceRule):
    """Compose two relations to infer a third (e.g. owns + located_in → operates_in)."""

    name = "composition"

    def __init__(
        self,
        composition_map: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self._map = composition_map if composition_map is not None else _COMPOSITION_MAP

    def apply(
        self,
        kg: EnhancedKG,
        *,
        document_id: str | None = None,
    ) -> list[InferredEdge]:
        results: list[InferredEdge] = []
        edges = list(kg.edges) if hasattr(kg, "edges") else []
        seen: set[tuple[str, str, str]] = set()

        # index by (source, rel_type) for fast lookup
        by_src_rel: dict[tuple[str, str], list] = {}
        for e in edges:
            by_src_rel.setdefault((e.source, e.relationship_type), []).append(e)

        for (rel1, rel2), inferred_rel in self._map.items():
            # Find A→B via rel1, then B→C via rel2 ⟹ A→C via inferred_rel
            for edge_ab in [e for e in edges if e.relationship_type == rel1]:
                for edge_bc in by_src_rel.get((edge_ab.target, rel2), []):
                    key = (edge_ab.source, inferred_rel, edge_bc.target)
                    if key in seen:
                        continue
                    seen.add(key)
                    conf = round(
                        _edge_confidence(edge_ab) * _edge_confidence(edge_bc) * 0.75,
                        4,
                    )
                    results.append(
                        InferredEdge(
                            source=edge_ab.source,
                            target=edge_bc.target,
                            relationship_type=inferred_rel,
                            relationship_detail=(
                                f"{edge_ab.source} {inferred_rel} {edge_bc.target} "
                                f"(composed from {rel1} + {rel2})"
                            ),
                            confidence=conf,
                            rule_name=self.name,
                            evidence=[
                                EvidenceLink(
                                    source_node=edge_ab.source,
                                    target_node=edge_ab.target,
                                    relationship_type=rel1,
                                    document_id=_edge_doc_id(edge_ab),
                                    confidence=_edge_confidence(edge_ab),
                                ),
                                EvidenceLink(
                                    source_node=edge_bc.source,
                                    target_node=edge_bc.target,
                                    relationship_type=rel2,
                                    document_id=_edge_doc_id(edge_bc),
                                    confidence=_edge_confidence(edge_bc),
                                ),
                            ],
                            metadata={
                                "inferred": True,
                                "inference": {
                                    "rule": self.name,
                                    "composed_from": [rel1, rel2],
                                    "via_node": edge_ab.target,
                                },
                            },
                        )
                    )
        return results
