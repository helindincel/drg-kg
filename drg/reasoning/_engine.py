"""Multi-document reasoning engine.

The engine applies a configurable list of :class:`InferenceRule` instances
over an :class:`~drg.graph.kg_core.EnhancedKG` and writes back the inferred
edges that pass the ``min_confidence`` threshold.

Usage::

    from drg.reasoning import MultiDocumentReasoner, ReasoningConfig

    cfg = ReasoningConfig(min_confidence=0.4)
    report = MultiDocumentReasoner(config=cfg).reason(kg, document_id="doc-1")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._explain import stamp_inferred_edge
from ._rules import (
    CompositionRule,
    InverseRule,
    PathBridgeRule,
    SymmetricRule,
    TransitiveRule,
)
from ._types import InferenceReport, InferenceRule, InferredEdge

if TYPE_CHECKING:
    from drg.graph.kg_core import EnhancedKG

__all__ = ["MultiDocumentReasoner", "ReasoningConfig"]


@dataclass
class ReasoningConfig:
    """Configuration for :class:`MultiDocumentReasoner`.

    Attributes:
        min_confidence: Inferred edges below this threshold are discarded.
        disabled_rules: Set of rule *names* to skip (e.g. ``{"path_bridge"}``).
        max_bridge_candidates_per_node: Hard cap on PathBridgeRule candidates
            per shared node, preventing combinatorial explosion on dense graphs.
        record_history: Whether to append the :class:`InferenceReport` to
            ``kg.metadata["reasoning_history"]``.
    """

    min_confidence: float = 0.3
    disabled_rules: frozenset[str] = field(default_factory=frozenset)
    max_bridge_candidates_per_node: int = 50
    record_history: bool = True


# Default rule pipeline (order matters — simpler rules first)
_DEFAULT_RULES: list[InferenceRule] = [
    SymmetricRule(),
    InverseRule(),
    TransitiveRule(),
    CompositionRule(),
    PathBridgeRule(),
]


class MultiDocumentReasoner:
    """Apply a configurable rule pipeline over an :class:`~drg.graph.kg_core.EnhancedKG`.

    Parameters
    ----------
    config:
        Optional :class:`ReasoningConfig`.  Defaults are used when ``None``.
    rules:
        Override the default rule list.  Useful for testing or custom pipelines.
    """

    def __init__(
        self,
        *,
        config: ReasoningConfig | None = None,
        rules: list[InferenceRule] | None = None,
    ) -> None:
        self._config = config or ReasoningConfig()
        self._rules = (
            rules
            if rules is not None
            else [r for r in _DEFAULT_RULES if r.name not in self._config.disabled_rules]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(
        self,
        kg: EnhancedKG,
        *,
        document_id: str | None = None,
        record_history: bool | None = None,
        dry_run: bool = False,
    ) -> InferenceReport:
        """Run all enabled rules and write inferred edges back to *kg*.

        Parameters
        ----------
        kg:
            The knowledge graph to reason over.  Modified in place unless
            ``dry_run=True``.
        document_id:
            When set, rules that distinguish document provenance (e.g.
            :class:`PathBridgeRule`) use this to identify inter-document edges.
        record_history:
            Override :attr:`ReasoningConfig.record_history` for this call.
        dry_run:
            Collect candidates without writing them back to *kg*.

        Returns
        -------
        InferenceReport
            Summary of what was inferred (and added, if not dry_run).
        """
        should_record = (
            record_history if record_history is not None else self._config.record_history
        )

        all_candidates: list[InferredEdge] = []
        rules_applied: list[str] = []

        for rule in self._rules:
            candidates = rule.apply(kg, document_id=document_id)
            all_candidates.extend(candidates)
            if candidates:
                rules_applied.append(rule.name)

        # Filter by confidence
        passing = [c for c in all_candidates if c.confidence >= self._config.min_confidence]
        skipped = len(all_candidates) - len(passing)

        # Deduplicate by (source, relationship_type, target), keeping highest confidence
        deduped: dict[tuple[str, str, str], InferredEdge] = {}
        for candidate in passing:
            key = (candidate.source, candidate.relationship_type, candidate.target)
            if key not in deduped or candidate.confidence > deduped[key].confidence:
                deduped[key] = candidate

        final_candidates = list(deduped.values())
        edges_added = 0

        if not dry_run:
            edges_added = self._write_edges(kg, final_candidates, document_id)

        report = InferenceReport(
            document_id=document_id,
            rules_applied=rules_applied,
            edges_inferred=len(final_candidates),
            edges_added=edges_added,
            skipped_low_confidence=skipped,
            dry_run=dry_run,
            inferred_edges=final_candidates,
        )

        if should_record and not dry_run:
            self._record_history(kg, report)

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_edges(
        self,
        kg: EnhancedKG,
        candidates: list[InferredEdge],
        document_id: str | None,
    ) -> int:
        """Write *candidates* back into *kg*, skipping already-present edges."""
        from drg.graph.kg_core import KGEdge

        written = 0
        existing_keys: set[tuple[str, str, str]] = set()
        if hasattr(kg, "edges"):
            for e in kg.edges:
                existing_keys.add((e.source, e.relationship_type, e.target))

        for candidate in candidates:
            key = (candidate.source, candidate.relationship_type, candidate.target)
            if key in existing_keys:
                continue
            # Both nodes must exist in the graph
            nodes = getattr(kg, "nodes", {})
            if candidate.source not in nodes or candidate.target not in nodes:
                continue

            stamped = stamp_inferred_edge(candidate, document_id=document_id)
            edge = KGEdge(
                source=stamped.source,
                target=stamped.target,
                relationship_type=stamped.relationship_type,
                relationship_detail=stamped.relationship_detail,
                confidence=stamped.confidence,
                metadata=stamped.metadata,
            )
            try:
                kg.add_edge(edge)
                existing_keys.add(key)
                written += 1
            except Exception:
                pass

        return written

    @staticmethod
    def _record_history(kg: EnhancedKG, report: InferenceReport) -> None:
        """Append a reasoning history entry to ``kg.metadata``."""
        meta: dict[str, Any] = dict(getattr(kg, "metadata", {}) or {})
        history: list[dict[str, Any]] = list(meta.get("reasoning_history") or [])
        history.append(report.to_dict())
        meta["reasoning_history"] = history
        try:
            kg.metadata = meta
        except (AttributeError, TypeError):
            pass
