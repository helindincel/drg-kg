"""Confidence scoring framework for DRG.

This package centralises confidence-score computation for the knowledge
graph pipeline. The data model (``KGNode`` / ``KGEdge`` /
``EnrichedRelationship``) carries the scalar; this package decides *how*
the scalar is produced.

Public API
----------
- :class:`ConfidenceStrategy` — pluggable scorer contract.
- :class:`DefaultConfidenceStrategy` — heuristic placeholder used when no
  strategy is supplied. Schema-aware, deterministic, side-effect-free.
- :class:`ConfidenceScore` — value + signal breakdown returned by
  strategies. The graph data model only stores the scalar; the breakdown
  is consumed by callers that want to audit the scoring.
- :func:`clamp_confidence` — utility that mirrors the model-layer bounds
  check (``[0.0, 1.0]``).

Why a separate package?
-----------------------
Strategies are pure transformations from extraction signals to scalar
scores. Keeping them out of ``drg.extract`` and ``drg.graph`` means:

- The graph layer doesn't depend on extraction internals.
- Calibrated/learned scorers can be swapped in without touching the
  pipeline's public APIs (``extract_typed``, ``build_enhanced_kg``).
- Tests can exercise scoring logic without spinning up DSPy/embeddings.

See ``docs/confidence_scoring.md`` for the design overview, the upgrade
roadmap (LLM self-rating, ensemble disagreement, calibration), and
guidance for authoring custom strategies.
"""

from __future__ import annotations

from ._default import DefaultConfidenceStrategy
from ._strategy import ConfidenceStrategy, EntityScoreMap, RelationScoreMap
from ._types import ConfidenceScore, clamp_confidence

__all__ = [
    "ConfidenceScore",
    "ConfidenceStrategy",
    "DefaultConfidenceStrategy",
    "EntityScoreMap",
    "RelationScoreMap",
    "clamp_confidence",
]
