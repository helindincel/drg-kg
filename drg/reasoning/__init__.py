"""Multi-document reasoning for DRG knowledge graphs.

This package implements a rule-based inference engine that derives new
relationships from patterns already present in an
:class:`~drg.graph.kg_core.EnhancedKG`.

Quick start::

    from drg.reasoning import MultiDocumentReasoner, ReasoningConfig

    cfg = ReasoningConfig(min_confidence=0.4, disabled_rules=frozenset({"path_bridge"}))
    report = MultiDocumentReasoner(config=cfg).reason(kg, document_id="doc-1")
    print(f"Inferred {report.edges_added} new edges")

CLI::

    drg extract input.txt --schema schema.json --infer --infer-min-confidence 0.4

Built-in rules
--------------
* ``path_bridge``  — cross-document bridge inference
* ``inverse``      — relation inverse inference
* ``symmetric``    — symmetric relation counterparts
* ``transitive``   — transitive closure
* ``composition``  — relation composition (A owns B, B located_in C ⟹ A operates_in C)
"""

from ._engine import MultiDocumentReasoner, ReasoningConfig
from ._rules import (
    CompositionRule,
    InverseRule,
    PathBridgeRule,
    SymmetricRule,
    TransitiveRule,
)
from ._types import EvidenceLink, InferenceReport, InferenceRule, InferredEdge

__all__ = [
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
]
