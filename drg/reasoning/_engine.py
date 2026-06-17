"""Multi-document reasoning engine.

The engine wires :class:`InferenceRule` instances over a real
:class:`drg.graph.kg_core.EnhancedKG`. Its job is the boring-but-vital
plumbing: apply the rules, validate every candidate, dedup against
already-present edges, mark each persisted edge as inferred with full
provenance, and return a typed report.

Design constraints respected here
---------------------------------

* **No mutation of inputs we don't own.** ``reason`` mutates the
  passed-in graph in place (adding inferred edges); it never touches
  pre-existing extracted edges. Callers that want a non-destructive
  run can deep-copy the graph first or use the ``dry_run=True`` flag.
* **Inferred edges are first-class but clearly marked.** Every edge
  the engine adds carries ``metadata['inferred'] = True`` and
  ``metadata['inference'] = {...full provenance...}``. The JSON output
  preserves these via the existing
  :meth:`drg.graph.kg_core.KGEdge.to_dict` round-trip.
* **Conservative, never speculative.** The engine refuses to add an
  edge whose endpoints don't exist in the graph (no node fabrication),
  whose endpoints collapse to the same node (no self-loops), or whose
  canonical key already exists (no duplicate facts).
* **Idempotent.** Running ``reason`` twice on the same graph emits no
  new edges the second time, because the canonical-key check
  recognises the first run's output as already-present.
* **Graph metadata trail.** The engine appends a ``reasoning`` entry
  to ``kg.metadata['history']`` describing the run, mirroring the
  bookkeeping :class:`drg.graph.incremental.GraphMerger` already does.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..utils.logging import get_logger
from ._rules import default_rules
from ._types import (
    InferenceReport,
    InferenceRule,
    InferredEdge,
    ReasoningConfig,
)

if TYPE_CHECKING:
    from ..graph.kg_core import EnhancedKG

logger = get_logger(__name__)

__all__ = ["MultiDocumentReasoner", "reason_over_graph"]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _normalize_rel(rel: str) -> str:
    return rel.strip().lower() if rel else ""


def _existing_edge_keys(kg: EnhancedKG) -> set[tuple[str, str, str]]:
    """Canonical edge-key set used for dedup.

    Mirrors :func:`drg.graph.incremental._normalize_relation_type`'s
    behaviour: case-insensitive relation type, exact endpoints (the
    merger has already canonicalised them at this point).
    """
    return {(e.source, _normalize_rel(e.relationship_type), e.target) for e in kg.edges}


class MultiDocumentReasoner:
    """Apply a configured list of :class:`InferenceRule`s to an :class:`EnhancedKG`.

    The reasoner is **stateless across runs**: configure once, call
    :meth:`reason` as many times as you like. Each call inspects the
    current graph, so previous inferences are visible to subsequent
    rule invocations.

    Constructor parameters
    ----------------------
    rules:
        Sequence of :class:`InferenceRule` instances. Defaults to
        :func:`drg.reasoning.default_rules`. Pass an empty list to make
        the reasoner a no-op (useful for unit-testing the engine in
        isolation).
    config:
        :class:`ReasoningConfig`. Defaults to a conservative profile.
    """

    def __init__(
        self,
        rules: list[InferenceRule] | None = None,
        config: ReasoningConfig | None = None,
    ) -> None:
        self.rules: list[InferenceRule] = list(rules) if rules is not None else default_rules()
        self.config: ReasoningConfig = config or ReasoningConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def reason(
        self,
        kg: EnhancedKG,
        *,
        document_id: str | None = None,
        record_history: bool = True,
        dry_run: bool = False,
    ) -> InferenceReport:
        """Run all enabled rules and (optionally) persist the inferred edges.

        Args:
            kg: The (typically merged) :class:`EnhancedKG` to reason over.
                Mutated in place when ``dry_run=False``.
            document_id: Optional label recorded in the history entry —
                handy when the reasoner runs as part of a pipeline that
                also persists a document_id for the underlying ingest.
            record_history: When ``True`` (default) and at least one
                inferred edge survives validation, append a ``reasoning``
                entry to ``kg.metadata['history']``.
            dry_run: When ``True``, validate and report what *would*
                be added without mutating ``kg``. The returned report
                lists the candidates in ``added_edges`` as usual; nothing
                is persisted.

        Returns:
            :class:`InferenceReport` summarising the run.
        """
        report = InferenceReport()
        from ..graph.kg_core import KGEdge  # local import: keeps cycle-free top-level

        if not isinstance(self.rules, list) or not self.rules:
            return report

        existing_keys = _existing_edge_keys(kg)
        added_count = 0
        max_inferences = self.config.max_inferences_per_run

        for rule in self.rules:
            if rule.name in self.config.disabled_rules:
                logger.debug("Skipping disabled rule: %s", rule.name)
                continue

            try:
                candidates = rule.apply(kg, self.config)
            except Exception as exc:  # rules must not crash the engine
                logger.warning(
                    "Rule %r raised %s: %s — skipping this rule.",
                    rule.name,
                    type(exc).__name__,
                    exc,
                )
                continue

            for candidate in candidates:
                if max_inferences is not None and added_count >= max_inferences:
                    break

                outcome = self._classify_candidate(
                    candidate=candidate,
                    kg=kg,
                    existing_keys=existing_keys,
                )
                triple = (
                    candidate.source,
                    candidate.relationship_type,
                    candidate.target,
                )

                if outcome == "missing_endpoint":
                    report.skipped_missing_endpoint.append(triple)
                    continue
                if outcome == "self_loop":
                    report.skipped_self_loop.append(triple)
                    continue
                if outcome == "duplicate":
                    report.skipped_existing.append(triple)
                    continue
                if outcome == "low_confidence":
                    report.skipped_low_confidence.append(triple)
                    continue
                # outcome == "ok"

                if not dry_run:
                    edge = self._build_edge(KGEdge, candidate)
                    kg.edges.append(edge)
                    existing_keys.add(
                        (
                            edge.source,
                            _normalize_rel(edge.relationship_type),
                            edge.target,
                        )
                    )

                report.added_edges.append(triple)
                report.per_rule_counts[rule.name] = report.per_rule_counts.get(rule.name, 0) + 1
                added_count += 1

        if record_history and not dry_run and not report.is_empty():
            self._record_history(kg, report, document_id=document_id)

        logger.info(
            "Multi-document reasoning: %s",
            ", ".join(f"{k}={v}" for k, v in report.summary().items() if v),
        )
        return report

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _classify_candidate(
        self,
        *,
        candidate: InferredEdge,
        kg: EnhancedKG,
        existing_keys: set[tuple[str, str, str]],
    ) -> str:
        """Return one of ``"ok"``, ``"missing_endpoint"``, ``"self_loop"``,
        ``"duplicate"``, ``"low_confidence"``."""
        if candidate.source not in kg.nodes or candidate.target not in kg.nodes:
            return "missing_endpoint"
        if candidate.source == candidate.target:
            return "self_loop"

        key = (
            candidate.source,
            _normalize_rel(candidate.relationship_type),
            candidate.target,
        )
        if key in existing_keys:
            return "duplicate"

        if candidate.confidence < self.config.min_confidence:
            return "low_confidence"
        return "ok"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _build_edge(KGEdge_cls: Any, candidate: InferredEdge):
        """Build a real :class:`KGEdge` from an :class:`InferredEdge`.

        Metadata layout (also documented in
        ``docs/multi_document_reasoning.md``)::

            {
              "inferred": True,
              "inference": {
                "rule": "<rule name>",
                "evidence_chain": [...],
                "source_documents": [...],
                "explanation": "...",
                "confidence": 0.81,
                ...optional: "bridge_entity", "extra"
              }
            }
        """
        inference_payload = candidate.to_inference_metadata()
        metadata: dict[str, Any] = {
            "inferred": True,
            "inference": inference_payload,
        }
        return KGEdge_cls(
            source=candidate.source,
            target=candidate.target,
            relationship_type=candidate.relationship_type,
            relationship_detail=candidate.explanation,
            metadata=metadata,
            confidence=candidate.confidence,
        )

    def _record_history(
        self,
        kg: EnhancedKG,
        report: InferenceReport,
        *,
        document_id: str | None,
    ) -> None:
        """Append a ``reasoning`` history entry mirroring the merger's format.

        We intentionally do *not* bump ``metadata['version']`` here — the
        version field tracks document-ingest history, and reasoning is a
        derived pass over the already-versioned graph. Callers that want
        a fresh version after reasoning can call
        :func:`drg.graph.incremental.GraphMerger.merge` against an empty
        graph or bump the version themselves.
        """
        meta = kg.metadata
        if not meta:
            meta["created_at"] = _utc_now_iso()
            meta["version"] = meta.get("version", 1)

        now = _utc_now_iso()
        meta["updated_at"] = now

        history: list[dict[str, Any]] = meta.setdefault("history", [])
        entry: dict[str, Any] = {
            "version": meta.get("version", 1),
            "operation": "reasoning",
            "timestamp": now,
            **report.summary(),
        }
        if document_id:
            entry["document_id"] = document_id
        history.append(entry)


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def reason_over_graph(
    kg: EnhancedKG,
    *,
    rules: list[InferenceRule] | None = None,
    config: ReasoningConfig | None = None,
    document_id: str | None = None,
    record_history: bool = True,
    dry_run: bool = False,
) -> InferenceReport:
    """One-call convenience around :class:`MultiDocumentReasoner`.

    Equivalent to::

        MultiDocumentReasoner(rules=rules, config=config).reason(
            kg,
            document_id=document_id,
            record_history=record_history,
            dry_run=dry_run,
        )
    """
    return MultiDocumentReasoner(rules=rules, config=config).reason(
        kg,
        document_id=document_id,
        record_history=record_history,
        dry_run=dry_run,
    )
