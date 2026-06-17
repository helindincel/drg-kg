"""Core data types for multi-document reasoning.

Every type defined here is **pure data** — no graph imports, no engine
imports — so that rules, the engine and external callers can all import
from a single light-weight surface without creating cycles.

Design notes
------------
- ``InferredEdge`` mirrors the **public-facing shape** of an inferred KG
  edge. The engine converts it into a real :class:`drg.graph.KGEdge` at
  the very last step; rules return ``InferredEdge`` so they don't have
  to import the heavier graph module.
- ``EvidenceLink`` captures a single extracted edge in an evidence chain
  with the source-document hint needed for cross-document provenance.
  The ``triple`` is the same ``(source, relation, target)`` tuple shape
  used everywhere else in DRG; ``source_ref`` reuses the existing
  ``metadata.source_ref`` convention so rules don't introduce a parallel
  vocabulary.
- ``InferenceRule`` is an ABC, not a Protocol, because we want a stable
  ``name`` attribute on every rule (used in provenance/explanation) and
  ABCs make that a hard requirement.
- ``ReasoningConfig`` is a frozen dataclass with conservative defaults
  matching the rest of DRG's "abstain when unsure" philosophy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph.kg_core import EnhancedKG

__all__ = [
    "EvidenceLink",
    "InferenceReport",
    "InferenceRule",
    "InferredEdge",
    "ReasoningConfig",
]


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceLink:
    """A single extracted edge cited by an inference.

    Attributes:
        triple: ``(source, relationship_type, target)`` of the extracted edge.
        source_ref: Document or chunk identifier the edge originated from
            (read from ``KGEdge.metadata['source_ref']``). ``None`` when
            the edge predates the multi-document workflow.
        confidence: Confidence of the extracted edge (may be ``None``).
        is_inferred: ``True`` when the cited edge is itself an inferred
            edge — captured so chains of inference are auditable and
            never silently feed back on themselves.
    """

    triple: tuple[str, str, str]
    source_ref: str | None = None
    confidence: float | None = None
    is_inferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"triple": list(self.triple)}
        if self.source_ref is not None:
            out["source_ref"] = self.source_ref
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.is_inferred:
            out["is_inferred"] = True
        return out


# ---------------------------------------------------------------------------
# Inferred edge model
# ---------------------------------------------------------------------------


@dataclass
class InferredEdge:
    """An edge proposed by an :class:`InferenceRule`.

    Rules return ``InferredEdge`` instances; the engine validates them
    (idempotency, endpoint existence, self-loop) and persists them as
    real ``KGEdge``s with the inference provenance baked into
    ``metadata['inference']``.

    Required fields mirror the minimum needed to build a ``KGEdge``;
    optional fields carry the provenance and explanation that make the
    inference *auditable*.
    """

    source: str
    target: str
    relationship_type: str
    rule_name: str
    evidence_chain: list[EvidenceLink]
    explanation: str
    confidence: float = 0.7
    bridge_entity: str | None = None
    """Set by chain rules (e.g. :class:`PathBridgeRule`) to point at the
    shared node connecting the two evidence edges. ``None`` for rules
    that don't operate on a single bridge node."""

    extra_metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form bag for rule-specific provenance the standard schema
    can't express (e.g. a transitive rule may want to record the
    intermediate nodes). Surfaces under
    ``KGEdge.metadata['inference']['extra']``."""

    def __post_init__(self) -> None:
        # Same validation surface as KGEdge — fail loudly here so rule
        # authors see the problem with their rule, not deep inside the
        # engine's add_edge call.
        if not self.source or not self.target:
            raise ValueError("InferredEdge source and target cannot be empty")
        if self.source == self.target:
            raise ValueError("InferredEdge source and target cannot be the same")
        if not self.relationship_type:
            raise ValueError("InferredEdge relationship_type cannot be empty")
        if not self.rule_name:
            raise ValueError("InferredEdge rule_name cannot be empty")
        if not self.evidence_chain:
            raise ValueError(
                "InferredEdge requires at least one EvidenceLink — "
                "inferences without evidence are not allowed"
            )
        if not self.explanation:
            raise ValueError("InferredEdge explanation cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"InferredEdge confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    @property
    def source_documents(self) -> list[str]:
        """Ordered, deduplicated list of document ids from the evidence chain."""
        seen: set[str] = set()
        ordered: list[str] = []
        for link in self.evidence_chain:
            if link.source_ref and link.source_ref not in seen:
                ordered.append(link.source_ref)
                seen.add(link.source_ref)
        return ordered

    def to_inference_metadata(self) -> dict[str, Any]:
        """Serialise into the dict written under ``KGEdge.metadata['inference']``.

        Keeping the serialisation centralised here means every rule
        gets the same provenance shape automatically — and downstream
        consumers (UI, query engine, ``drg.reasoning.explain``) can
        depend on the schema.
        """
        payload: dict[str, Any] = {
            "rule": self.rule_name,
            "evidence_chain": [link.to_dict() for link in self.evidence_chain],
            "source_documents": self.source_documents,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }
        if self.bridge_entity:
            payload["bridge_entity"] = self.bridge_entity
        if self.extra_metadata:
            payload["extra"] = dict(self.extra_metadata)
        return payload


# ---------------------------------------------------------------------------
# Rule contract
# ---------------------------------------------------------------------------


class InferenceRule(ABC):
    """Contract for a single multi-document reasoning rule.

    Subclasses must declare a stable :attr:`name` (used in provenance
    and ``ReasoningConfig.disabled_rules`` filtering) and implement
    :meth:`apply`, returning a list of :class:`InferredEdge` candidates.

    Rules **must not** mutate the graph passed in; the engine handles
    persistence after validation. Rules **must** be pure functions of
    the graph + config so that the engine's idempotency guarantees hold.
    """

    #: Stable identifier; surfaces under ``metadata.inference.rule``.
    name: str = ""

    #: When ``True`` the rule will only fire when the evidence edges
    #: have **distinct** ``source_ref`` values — i.e. genuinely
    #: cross-document inference. Defaults to ``False`` so simple rules
    #: like :class:`InverseRule` can apply within a single document too.
    requires_cross_document: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise TypeError(
                f"InferenceRule subclass {cls.__name__!r} must set a non-empty `name` attribute"
            )

    @abstractmethod
    def apply(
        self,
        kg: EnhancedKG,
        config: ReasoningConfig,
    ) -> list[InferredEdge]:
        """Return candidate inferred edges for ``kg`` under ``config``.

        Implementations must be deterministic, pure, and free of side
        effects. The engine handles dedup/idempotency, so rules don't
        need to check whether a candidate already exists in ``kg``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Reasoner configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReasoningConfig:
    """Knobs for the :class:`MultiDocumentReasoner`.

    Defaults are deliberately conservative: only the bridge rule has a
    higher confidence floor (it's the most speculative rule). All
    rule-specific knobs live here rather than in individual rules so
    callers see a single configuration surface.
    """

    min_confidence: float = 0.5
    """Drop candidates whose computed confidence is below this floor."""

    max_inferences_per_run: int | None = None
    """Hard cap on the total number of inferred edges produced per
    ``reason`` call. ``None`` means no cap. Useful for very large
    graphs where the bridge rule can produce many candidates."""

    disabled_rules: frozenset[str] = field(default_factory=frozenset)
    """Names of rules to skip — read from each rule's ``name``."""

    max_bridge_candidates_per_node: int = 32
    """Per-bridge-node cap for :class:`PathBridgeRule`. Hub nodes (e.g.
    a country mentioned in many documents) can otherwise explode the
    output. The cap keeps the worst case linear in graph size."""

    bridge_confidence_floor: float = 0.6
    """Lower bound applied specifically to bridge-rule confidences before
    :attr:`min_confidence` filtering. Bridge inferences are the most
    speculative, so we keep them on a slightly stricter leash."""

    require_distinct_bridge_relations: bool = True
    """When ``True``, the bridge rule only fires when the two evidence
    edges have **different** relationship types (Apple ACQUIRED Beats +
    Jimmy FOUNDED Beats → fires; Doc1 OWNS Beats + Doc2 OWNS Beats →
    skipped). Prevents inferring connections from two re-observations
    of the same fact."""

    allow_inferred_in_evidence: bool = False
    """When ``False`` (default), rules will only chain off **extracted**
    edges, never off previously-inferred ones. Setting to ``True`` lets
    rules build longer inference chains at the cost of weaker
    provenance."""


# ---------------------------------------------------------------------------
# Run report
# ---------------------------------------------------------------------------


@dataclass
class InferenceReport:
    """Summary of a single :meth:`MultiDocumentReasoner.reason` call.

    Mirrors :class:`drg.graph.incremental.KGDiff` in spirit: callers
    get a typed report they can log/serialise/inspect without having
    to re-walk the graph themselves.
    """

    added_edges: list[tuple[str, str, str]] = field(default_factory=list)
    """``(source, relationship_type, target)`` triples actually added."""

    skipped_existing: list[tuple[str, str, str]] = field(default_factory=list)
    """Candidates dropped because an equivalent edge already existed."""

    skipped_low_confidence: list[tuple[str, str, str]] = field(default_factory=list)
    """Candidates dropped by the confidence floor."""

    skipped_self_loop: list[tuple[str, str, str]] = field(default_factory=list)
    """Candidates dropped because endpoints collapsed to the same node."""

    skipped_missing_endpoint: list[tuple[str, str, str]] = field(default_factory=list)
    """Candidates dropped because a rule emitted an endpoint not in the
    graph (defensive — well-behaved rules don't hit this path)."""

    per_rule_counts: dict[str, int] = field(default_factory=dict)
    """``{rule_name: edges_added_by_that_rule}``."""

    def summary(self) -> dict[str, Any]:
        return {
            "added_edges": len(self.added_edges),
            "skipped_existing": len(self.skipped_existing),
            "skipped_low_confidence": len(self.skipped_low_confidence),
            "skipped_self_loop": len(self.skipped_self_loop),
            "skipped_missing_endpoint": len(self.skipped_missing_endpoint),
            "per_rule_counts": dict(self.per_rule_counts),
        }

    def is_empty(self) -> bool:
        return not (
            self.added_edges
            or self.skipped_existing
            or self.skipped_low_confidence
            or self.skipped_self_loop
            or self.skipped_missing_endpoint
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_edges": [list(t) for t in self.added_edges],
            "skipped_existing": [list(t) for t in self.skipped_existing],
            "skipped_low_confidence": [list(t) for t in self.skipped_low_confidence],
            "skipped_self_loop": [list(t) for t in self.skipped_self_loop],
            "skipped_missing_endpoint": [list(t) for t in self.skipped_missing_endpoint],
            "per_rule_counts": dict(self.per_rule_counts),
        }
