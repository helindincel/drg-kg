"""Event data model — first-class citizens of the knowledge graph.

These dataclasses represent the canonical, in-memory shape of an event.
They are intentionally decoupled from the graph storage layer
(:mod:`drg.graph.kg_core`); the bridging between an :class:`Event` and a
``KGNode`` / ``KGEdge`` lives in :mod:`drg.events._graph_mapping`.

Design notes
------------
- All dataclasses are validated in ``__post_init__`` and raise plain
  :class:`ValueError` for shape contracts (consistent with
  :class:`drg.graph.kg_core.KGNode` / :class:`KGEdge`).
- ``to_dict`` / ``from_dict`` are pure-data round-trips so events can be
  persisted as JSON without bespoke serialisers in every consumer.
- Timestamps follow ISO 8601 strings with an explicit ``precision`` flag —
  this keeps the public API uniform with how :mod:`drg.graph.kg_core`
  already represents temporal data on edges (``start_time`` / ``end_time``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "Event",
    "EventProvenance",
    "EventRole",
    "EventTimestamp",
    "EventTypeDefinition",
    "TextSpan",
    "TimestampPrecision",
]


TimestampPrecision = Literal["year", "month", "day", "instant"]


@dataclass(frozen=True)
class EventRole:
    """Typed role describing how an entity participates in an event.

    A role is the contract between an event type (``Acquisition``) and
    the entities that fill it (``acquirer``, ``acquired``). ``entity_types``
    is a whitelist; an empty tuple means "any entity type is allowed",
    which keeps the system extensible for free-form event types.
    """

    name: str
    description: str = ""
    entity_types: tuple[str, ...] = ()
    cardinality: Literal["one", "many"] = "one"
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("EventRole.name cannot be empty")
        if self.cardinality not in ("one", "many"):
            raise ValueError(
                f"EventRole.cardinality must be 'one' or 'many', got {self.cardinality!r}"
            )

    def accepts(self, entity_type: str | None) -> bool:
        """Return True if an entity of the given type can fill this role."""
        if not self.entity_types:
            return True
        return entity_type in self.entity_types

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "entity_types": list(self.entity_types),
            "cardinality": self.cardinality,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventRole:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            entity_types=tuple(data.get("entity_types", []) or []),
            cardinality=data.get("cardinality", "one"),
            required=bool(data.get("required", True)),
        )


@dataclass
class EventTypeDefinition:
    """Declarative definition of one event type (e.g. ``Acquisition``).

    The set of these definitions inside an :class:`EventTypeRegistry` is
    what the extractor sees — there is no hard-coded event taxonomy
    anywhere in the pipeline. Adding a new event type is a pure-data
    change, no code edits required.
    """

    name: str
    description: str
    roles: list[EventRole] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("EventTypeDefinition.name cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError("EventTypeDefinition.description cannot be empty")
        seen: set[str] = set()
        for role in self.roles:
            if role.name in seen:
                raise ValueError(
                    f"Duplicate role '{role.name}' in event type '{self.name}'"
                )
            seen.add(role.name)

    def get_role(self, name: str) -> EventRole | None:
        for role in self.roles:
            if role.name == name:
                return role
        return None

    def required_roles(self) -> list[EventRole]:
        return [r for r in self.roles if r.required]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "roles": [r.to_dict() for r in self.roles],
            "properties": dict(self.properties),
            "examples": list(self.examples),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventTypeDefinition:
        return cls(
            name=data["name"],
            description=data["description"],
            roles=[EventRole.from_dict(r) for r in data.get("roles", []) or []],
            properties=dict(data.get("properties", {}) or {}),
            examples=list(data.get("examples", []) or []),
        )


@dataclass(frozen=True)
class EventTimestamp:
    """ISO 8601 temporal scope of an event.

    ``start`` / ``end`` are strings in ISO 8601 form (``YYYY``,
    ``YYYY-MM``, ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM:SSZ`` are all
    valid). ``precision`` records the granularity actually present in
    the source text — consumers can use it to render appropriate UIs
    without re-parsing the string.

    ``raw_text`` preserves the surface form ("late 2014", "Q3 2023")
    so downstream tooling can show exactly what the document said.
    """

    start: str | None = None
    end: str | None = None
    precision: TimestampPrecision = "year"
    raw_text: str | None = None

    def __post_init__(self) -> None:
        valid = ("year", "month", "day", "instant")
        if self.precision not in valid:
            raise ValueError(
                f"EventTimestamp.precision must be one of {valid}, got {self.precision!r}"
            )

    def is_empty(self) -> bool:
        return not self.start and not self.end and not self.raw_text

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"precision": self.precision}
        if self.start is not None:
            out["start"] = self.start
        if self.end is not None:
            out["end"] = self.end
        if self.raw_text is not None:
            out["raw_text"] = self.raw_text
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventTimestamp:
        return cls(
            start=data.get("start"),
            end=data.get("end"),
            precision=data.get("precision", "year"),
            raw_text=data.get("raw_text"),
        )


@dataclass(frozen=True)
class TextSpan:
    """A supporting text span used as evidence for an event.

    ``start`` / ``end`` are character offsets when known; they are
    optional because LLM-extracted evidence often arrives as a free
    snippet without exact offsets.
    """

    text: str
    chunk_id: str | None = None
    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("TextSpan.text cannot be empty")
        if self.start is not None and self.start < 0:
            raise ValueError("TextSpan.start must be >= 0")
        if self.end is not None and self.start is not None and self.end < self.start:
            raise ValueError("TextSpan.end must be >= start")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"text": self.text}
        if self.chunk_id is not None:
            out["chunk_id"] = self.chunk_id
        if self.start is not None:
            out["start"] = self.start
        if self.end is not None:
            out["end"] = self.end
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextSpan:
        return cls(
            text=data["text"],
            chunk_id=data.get("chunk_id"),
            start=data.get("start"),
            end=data.get("end"),
        )


@dataclass
class EventProvenance:
    """Where an event came from and how confident we are about it.

    Keeping provenance as its own dataclass (instead of stuffing it in a
    metadata bag) makes auditability cheap: callers can read
    ``event.provenance.confidence`` directly without parsing nested dicts.
    """

    document_id: str | None = None
    chunk_ids: list[str] = field(default_factory=list)
    text_spans: list[TextSpan] = field(default_factory=list)
    extracted_at: str | None = None
    extractor_version: str | None = None
    extraction_method: Literal["llm", "rule", "merged", "manual"] = "llm"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"EventProvenance.confidence must be in [0, 1], got {self.confidence}"
            )
        valid_methods = ("llm", "rule", "merged", "manual")
        if self.extraction_method not in valid_methods:
            raise ValueError(
                f"EventProvenance.extraction_method must be one of {valid_methods}, "
                f"got {self.extraction_method!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_ids": list(self.chunk_ids),
            "text_spans": [s.to_dict() for s in self.text_spans],
            "extracted_at": self.extracted_at,
            "extractor_version": self.extractor_version,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventProvenance:
        return cls(
            document_id=data.get("document_id"),
            chunk_ids=list(data.get("chunk_ids", []) or []),
            text_spans=[TextSpan.from_dict(s) for s in data.get("text_spans", []) or []],
            extracted_at=data.get("extracted_at"),
            extractor_version=data.get("extractor_version"),
            extraction_method=data.get("extraction_method", "llm"),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class Event:
    """Canonical in-memory representation of an event.

    An :class:`Event` is the primary unit of "something happened" in a
    knowledge graph: an n-ary fact with typed participants, a temporal
    scope, an optional location, free-form properties, and full
    provenance.

    Mapping to the storage layer
    -----------------------------
    An :class:`Event` is *not* the persisted form. Persistence happens
    through :mod:`drg.events._graph_mapping`, which projects each event
    onto a single ``KGNode`` (with ``type='Event:<event_type>'``) plus
    one ``KGEdge`` per ``(role, participant)`` pair. This keeps the
    storage layer dependency-free of the event package and lets all
    existing consumers (Neo4j, visualization, reasoning, merger) work
    on event-augmented graphs without modification.
    """

    id: str
    event_type: str
    participants: dict[str, list[str]] = field(default_factory=dict)
    timestamp: EventTimestamp | None = None
    location: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    provenance: EventProvenance = field(default_factory=EventProvenance)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Event.id cannot be empty")
        if not self.event_type or not self.event_type.strip():
            raise ValueError("Event.event_type cannot be empty")
        for role_name, entities in self.participants.items():
            if not isinstance(entities, list):
                raise ValueError(
                    f"Event.participants[{role_name!r}] must be a list, "
                    f"got {type(entities).__name__}"
                )

    @property
    def confidence(self) -> float:
        """Convenience accessor for ``provenance.confidence``."""
        return self.provenance.confidence

    def participant_entities(self) -> list[str]:
        """Flat, de-duplicated list of every entity referenced as a participant."""
        seen: set[str] = set()
        out: list[str] = []
        for entities in self.participants.values():
            for e in entities:
                if e not in seen:
                    seen.add(e)
                    out.append(e)
        return out

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "event_type": self.event_type,
            "participants": {k: list(v) for k, v in self.participants.items()},
            "properties": dict(self.properties),
            "provenance": self.provenance.to_dict(),
            "metadata": dict(self.metadata),
        }
        if self.timestamp is not None and not self.timestamp.is_empty():
            out["timestamp"] = self.timestamp.to_dict()
        if self.location is not None:
            out["location"] = self.location
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        ts_data = data.get("timestamp")
        timestamp = EventTimestamp.from_dict(ts_data) if isinstance(ts_data, dict) else None
        prov_data = data.get("provenance") or {}
        provenance = (
            EventProvenance.from_dict(prov_data)
            if isinstance(prov_data, dict)
            else EventProvenance()
        )
        return cls(
            id=data["id"],
            event_type=data["event_type"],
            participants={
                k: list(v) for k, v in (data.get("participants") or {}).items()
            },
            timestamp=timestamp,
            location=data.get("location"),
            properties=dict(data.get("properties", {}) or {}),
            provenance=provenance,
            metadata=dict(data.get("metadata", {}) or {}),
        )
