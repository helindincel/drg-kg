"""Confidence framework — value types.

Lightweight, dependency-free dataclasses used by the confidence strategy
layer. They are intentionally separate from the strategy/default
implementations so callers can import the *types* without pulling in
heavier scoring logic.

Design notes
------------
- ``ConfidenceScore`` carries both the final scalar and a breakdown of the
  signals that produced it. This is useful for debugging and for downstream
  systems that may want to re-aggregate signals with their own weights.
- Scores are clamped to ``[0.0, 1.0]`` at construction time. The clamp is
  defensive: heuristic scorers may overshoot when combining multiple boosts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ConfidenceScore", "clamp_confidence"]


def clamp_confidence(value: float) -> float:
    """Clamp a raw score into the canonical ``[0.0, 1.0]`` range.

    Mirrors the validation the data-model layer (``KGNode``/``KGEdge``)
    enforces. Centralising it here keeps behaviour consistent across all
    strategies and avoids per-strategy ``min/max`` boilerplate.
    """
    if value != value:  # NaN check (NaN != NaN)
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


@dataclass(frozen=True)
class ConfidenceScore:
    """Final confidence score plus the signals that produced it.

    Attributes:
        value: Final confidence in ``[0.0, 1.0]``.
        signals: Optional ``signal_name -> contribution`` map. Strategies
            that compute the score from multiple cues (schema validation,
            keyword match, heuristic inference) populate this so that
            downstream consumers can audit *why* a value is what it is.
        method: Name of the strategy that produced the score (default
            "unknown"). Useful for telemetry / mixed-strategy graphs.
    """

    value: float
    signals: dict[str, float] = field(default_factory=dict)
    method: str = "unknown"

    def __post_init__(self) -> None:
        # ``frozen=True`` forbids ``self.value = ...``; use object.__setattr__.
        object.__setattr__(self, "value", clamp_confidence(self.value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "signals": dict(self.signals),
            "method": self.method,
        }
