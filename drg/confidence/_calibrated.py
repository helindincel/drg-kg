"""Calibrated confidence strategy.

This strategy wraps the existing deterministic default scorer with optional
labelled calibration points. It is intentionally lightweight: callers can pass
observed ``(predicted, actual)`` pairs from an evaluation set, and the strategy
learns a monotonic reliability curve by binning those observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._default import DefaultConfidenceStrategy
from ._strategy import EntityScoreMap, RelationScoreMap
from ._types import ConfidenceScore, clamp_confidence

__all__ = ["CalibratedConfidenceStrategy", "CalibrationPoint"]


@dataclass(frozen=True)
class CalibrationPoint:
    """One labelled confidence observation."""

    predicted: float
    actual: bool | float

    @property
    def target(self) -> float:
        if isinstance(self.actual, bool):
            return 1.0 if self.actual else 0.0
        return clamp_confidence(float(self.actual))


class CalibratedConfidenceStrategy(DefaultConfidenceStrategy):
    """Default scorer plus a labelled reliability correction.

    When no calibration points are supplied this behaves like the default
    strategy but marks scores as ``calibrated_missing_data``. With points, raw
    scores are mapped through nearest non-empty reliability bins.
    """

    name = "calibrated"

    def __init__(
        self,
        calibration_points: list[CalibrationPoint | tuple[float, bool | float]] | None = None,
        *,
        bins: int = 10,
    ) -> None:
        self.bins = max(2, bins)
        self._bin_reliability = self._fit(calibration_points or [])

    def score_entities(
        self,
        entities: list[tuple[str, str]],
        *,
        context: dict[str, Any] | None = None,
    ) -> EntityScoreMap:
        raw = super().score_entities(entities, context=context)
        return {key: self._calibrate(score) for key, score in raw.items()}

    def score_relations(
        self,
        relations: list[tuple[str, str, str]],
        *,
        enriched_relations: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> RelationScoreMap:
        raw = super().score_relations(
            relations,
            enriched_relations=enriched_relations,
            context=context,
        )
        return {key: self._calibrate(score) for key, score in raw.items()}

    def _fit(
        self,
        calibration_points: list[CalibrationPoint | tuple[float, bool | float]],
    ) -> dict[int, float]:
        buckets: dict[int, list[float]] = {}
        for item in calibration_points:
            point = item if isinstance(item, CalibrationPoint) else CalibrationPoint(*item)
            predicted = clamp_confidence(float(point.predicted))
            bucket = min(self.bins - 1, int(predicted * self.bins))
            buckets.setdefault(bucket, []).append(point.target)
        return {bucket: sum(values) / len(values) for bucket, values in buckets.items()}

    def _calibrate(self, score: ConfidenceScore) -> ConfidenceScore:
        raw = clamp_confidence(score.value)
        if not self._bin_reliability:
            signals = dict(score.signals)
            signals["calibrated_missing_data"] = 0.0
            return ConfidenceScore(value=raw, signals=signals, method=self.name)

        bucket = min(self.bins - 1, int(raw * self.bins))
        if bucket in self._bin_reliability:
            calibrated = self._bin_reliability[bucket]
        else:
            nearest = min(self._bin_reliability, key=lambda idx: abs(idx - bucket))
            calibrated = self._bin_reliability[nearest]

        signals = dict(score.signals)
        signals["raw_confidence"] = raw
        signals["calibration_delta"] = calibrated - raw
        return ConfidenceScore(value=calibrated, signals=signals, method=self.name)
