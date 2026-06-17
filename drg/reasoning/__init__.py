"""Multi-document reasoning for :class:`drg.graph.kg_core.EnhancedKG`.

This package layers **graph-level inference** on top of the existing
DRG extraction pipeline. It runs *after* extraction, *after*
:class:`drg.graph.incremental.GraphMerger` has merged per-document
graphs into a long-lived KG, and produces **new inferred edges** that
carry full provenance back to the extracted edges (and the documents)
that license them.

Why a separate module?
======================

The historic DRG pipeline:

1. ``extract.extract_typed`` / ``extract_from_chunks`` (DSPy) →
   ``(entities, triples)`` for one document.
2. ``graph.builders.build_enhanced_kg`` → ``EnhancedKG`` for that
   document.
3. ``graph.incremental.GraphMerger.merge`` → folds the per-document
   graph into a persistent global KG (entity dedup, version bump,
   history entry).

Everything above is intra-document or "merge two graphs side by side".
Nothing in that pipeline answers questions like *"Is Apple connected
to Jimmy Iovine via Beats?"* once Apple/Beats and Jimmy/Beats are
extracted from **different documents**. The
:class:`MultiDocumentReasoner` here fills that gap.

What it deliberately does **not** do
====================================

- No LLM calls. All inference is rule-based, deterministic, and
  schema-agnostic — there's no temperature, no prompt, no hallucination
  risk.
- No mutation of extraction code. The pipeline keeps working unchanged.
- No new dependencies. Pure stdlib + the existing graph/utils modules.
- No fabrication. Every inferred edge requires at least one extracted
  ``EvidenceLink`` and is dropped when its endpoints don't already
  exist in the graph.

Public surface
==============

- :class:`MultiDocumentReasoner` — apply rules to a graph in place.
- :func:`reason_over_graph` — one-call convenience.
- :class:`ReasoningConfig` — knobs (confidence floor, per-bridge caps,
  per-rule disable list, etc.).
- :class:`InferenceRule` — base class for custom rules.
- :class:`InferredEdge`, :class:`EvidenceLink`, :class:`InferenceReport` —
  data model for rule outputs and run reports.
- :func:`default_rules` — the bundled built-ins.

Built-in rules
==============

- :class:`PathBridgeRule` — the multi-document workhorse. Detects
  shared "bridge" entities mentioned in edges from **different
  documents** and emits a ``connected_via_<bridge>`` edge between the
  outer endpoints.
- :class:`InverseRule` — emits the missing inverse direction for stable
  inverse pairs (``founded`` ↔ ``founded_by``, ``owns`` ↔ ``owned_by``,
  …).
- :class:`SymmetricRule` — back-edge for symmetric predicates
  (``works_with``, ``collaborates_with``, …).
- :class:`TransitiveRule` — one hop of transitive closure for
  whitelisted predicates (``part_of``, ``subclass_of``, ``located_in``,
  …).
- :class:`CompositionRule` — ``owns(A,B) + located_in(B,L)`` →
  ``operates_in(A,L)`` (graph-only complement to the existing
  text-based extract-time heuristic).

Quick example
=============

::

    from drg.graph import EnhancedKG, GraphMerger
    from drg.graph.builders import build_enhanced_kg
    from drg.reasoning import MultiDocumentReasoner

    base = EnhancedKG()
    for doc_id, text, entities, triples in your_documents:
        new_kg = build_enhanced_kg(
            entities_typed=entities,
            triples=triples,
            source_text=text,
            document_id=doc_id,
        )
        GraphMerger().merge(base, new_kg, document_id=doc_id)

    report = MultiDocumentReasoner().reason(base)
    print(report.summary())
    # {'added_edges': 3, 'per_rule_counts': {'path_bridge': 1, ...}, ...}

See ``docs/multi_document_reasoning.md`` for the architecture, rule
catalog, provenance schema and tips on authoring custom rules.
"""

from __future__ import annotations

from ._engine import MultiDocumentReasoner, reason_over_graph
from ._rules import (
    INVERSE_RELATION_PAIRS,
    LOCATION_RELATIONS_COMPOSITION,
    OWNERSHIP_RELATIONS_COMPOSITION,
    SYMMETRIC_RELATIONS,
    TRANSITIVE_RELATIONS,
    CompositionRule,
    InverseRule,
    PathBridgeRule,
    SymmetricRule,
    TransitiveRule,
    default_rules,
)
from ._types import (
    EvidenceLink,
    InferenceReport,
    InferenceRule,
    InferredEdge,
    ReasoningConfig,
)

__all__ = [
    "INVERSE_RELATION_PAIRS",
    "LOCATION_RELATIONS_COMPOSITION",
    "OWNERSHIP_RELATIONS_COMPOSITION",
    "SYMMETRIC_RELATIONS",
    "TRANSITIVE_RELATIONS",
    "CompositionRule",
    "EvidenceLink",
    "InferenceReport",
    "InferenceRule",
    "InferredEdge",
    "InverseRule",
    "MultiDocumentReasoner",
    "PathBridgeRule",
    "ReasoningConfig",
    "SymmetricRule",
    "TransitiveRule",
    "default_rules",
    "reason_over_graph",
]
