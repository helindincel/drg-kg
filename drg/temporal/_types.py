"""Temporal metadata types for entities, relationships, and events.

Supports partial ISO 8601 dates (year, month, day, instant) so facts like
``valid_from = 2014`` or ``valid_from = 2014-06`` round-trip without lossy
normalisation.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "TemporalPrecision",
    "PartialDate",
    "TemporalScope",
    "temporal_from_edge_fields",
    "temporal_to_edge_fields",
]

TemporalPrecision = Literal["year", "month", "day", "instant"]

_YEAR_RE = re.compile(r"^\d{4}$")
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class PartialDate:
    """A calendar date with explicit precision.

    ``value`` is stored exactly as provided (``2014``, ``2014-06``,
    ``2014-06-15``, or a full ISO instant). ``precision`` records the
    granularity so consumers can render the original intent.
    """

    value: str
    precision: TemporalPrecision = "day"

    def __post_init__(self) -> None:
        if not self.value or not str(self.value).strip():
            raise ValueError("PartialDate.value cannot be empty")
        valid = ("year", "month", "day", "instant")
        if self.precision not in valid:
            raise ValueError(
                f"PartialDate.precision must be one of {valid}, got {self.precision!r}"
            )

    @classmethod
    def parse(cls, raw: str | None) -> PartialDate | None:
        """Infer precision from an ISO-like string."""
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        if _YEAR_RE.match(s):
            return cls(value=s, precision="year")
        if _MONTH_RE.match(s):
            return cls(value=s, precision="month")
        if _DAY_RE.match(s):
            return cls(value=s, precision="day")
        return cls(value=s, precision="instant")

    def interval_start(self) -> str:
        """Inclusive lower bound as ``YYYY-MM-DD`` (or full instant)."""
        if self.precision == "year":
            return f"{self.value}-01-01"
        if self.precision == "month":
            return f"{self.value}-01"
        if self.precision == "day":
            return self.value
        return self.value

    def interval_end(self) -> str:
        """Inclusive upper bound as ``YYYY-MM-DD`` (or full instant)."""
        if self.precision == "year":
            return f"{self.value}-12-31"
        if self.precision == "month":
            year_s, month_s = self.value.split("-", 1)
            year, month = int(year_s), int(month_s)
            last_day = calendar.monthrange(year, month)[1]
            return f"{year}-{month:02d}-{last_day}"
        if self.precision == "day":
            return self.value
        return self.value

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "precision": self.precision}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PartialDate:
        return cls(
            value=str(data["value"]),
            precision=data.get("precision", "day"),
        )


@dataclass
class TemporalScope:
    """Validity window and provenance timestamps for a graph fact.

    ``valid_from`` / ``valid_to`` describe when the fact held in the world.
    ``created_at`` / ``updated_at`` describe extraction or merge timestamps.
    All fields are optional — graphs without temporal data remain valid.
    """

    valid_from: str | None = None
    valid_to: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    precision_from: TemporalPrecision | None = None
    precision_to: TemporalPrecision | None = None
    raw_text: str | None = None

    def is_empty(self) -> bool:
        return not any(
            (
                self.valid_from,
                self.valid_to,
                self.created_at,
                self.updated_at,
                self.raw_text,
            )
        )

    def with_precision_defaults(self) -> TemporalScope:
        """Fill missing precision flags by parsing date strings."""
        pf = self.precision_from
        pt = self.precision_to
        if pf is None and self.valid_from:
            parsed = PartialDate.parse(self.valid_from)
            pf = parsed.precision if parsed else None
        if pt is None and self.valid_to:
            parsed = PartialDate.parse(self.valid_to)
            pt = parsed.precision if parsed else None
        return TemporalScope(
            valid_from=self.valid_from,
            valid_to=self.valid_to,
            created_at=self.created_at,
            updated_at=self.updated_at,
            precision_from=pf,
            precision_to=pt,
            raw_text=self.raw_text,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.valid_from is not None:
            out["valid_from"] = self.valid_from
        if self.valid_to is not None:
            out["valid_to"] = self.valid_to
        if self.created_at is not None:
            out["created_at"] = self.created_at
        if self.updated_at is not None:
            out["updated_at"] = self.updated_at
        if self.precision_from is not None:
            out["precision_from"] = self.precision_from
        if self.precision_to is not None:
            out["precision_to"] = self.precision_to
        if self.raw_text is not None:
            out["raw_text"] = self.raw_text
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TemporalScope | None:
        if not data or not isinstance(data, dict):
            return None
        scope = cls(
            valid_from=data.get("valid_from") or data.get("start"),
            valid_to=data.get("valid_to") or data.get("end"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            precision_from=data.get("precision_from") or data.get("precision"),
            precision_to=data.get("precision_to"),
            raw_text=data.get("raw_text"),
        )
        return None if scope.is_empty() else scope

    @classmethod
    def from_legacy_temporal(cls, temporal: dict[str, Any] | None) -> TemporalScope | None:
        """Convert extraction ``{"start": ..., "end": ...}`` shape."""
        if not temporal or not isinstance(temporal, dict):
            return None
        return cls.from_dict(temporal)


def temporal_from_edge_fields(
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TemporalScope | None:
    """Build a :class:`TemporalScope` from :class:`KGEdge` fields."""
    meta = metadata or {}
    nested = meta.get("temporal")
    if isinstance(nested, dict):
        scope = TemporalScope.from_dict(nested)
        if scope is not None:
            return scope.with_precision_defaults()

    scope = TemporalScope(
        valid_from=start_time,
        valid_to=end_time,
        created_at=created_at or meta.get("created_at"),
        updated_at=updated_at or meta.get("updated_at"),
    )
    if scope.is_empty():
        return None
    return scope.with_precision_defaults()


def temporal_to_edge_fields(scope: TemporalScope | None) -> dict[str, Any]:
    """Map a scope onto legacy ``start_time`` / ``end_time`` edge fields."""
    if scope is None or scope.is_empty():
        return {}
    out: dict[str, Any] = {
        "start_time": scope.valid_from,
        "end_time": scope.valid_to,
        "created_at": scope.created_at,
        "updated_at": scope.updated_at,
    }
    temporal_meta = scope.to_dict()
    if temporal_meta:
        out["metadata_temporal"] = temporal_meta
    return out
