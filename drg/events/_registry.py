"""Event-type registry — mutable, dataset-agnostic catalogue of event types.

The registry is the single point through which the rest of the pipeline
discovers which event types exist. There is no hard-coded taxonomy: the
default factory returns an *empty* registry, and callers wire in either
the bundled examples (:func:`example_event_registry`) or their own
domain-specific definitions.

Why empty by default?
---------------------
Hard-coding a "standard 10" set into the default would push opinions into
every consumer of the module. Keeping the default empty preserves the
"avoid hardcoding event logic for specific domains" constraint while
still giving users a one-liner (``example_event_registry()``) for quick
starts and demos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._types import EventRole, EventTypeDefinition

__all__ = [
    "EventTypeRegistry",
    "default_event_registry",
    "example_event_registry",
]


class EventTypeRegistry:
    """Ordered, name-indexed collection of :class:`EventTypeDefinition`.

    The ordering matters because it is reflected in the prompt sent to
    the LLM during extraction; deterministic ordering keeps prompts
    stable across runs and makes optimizer training reproducible.
    """

    def __init__(self, types: list[EventTypeDefinition] | None = None) -> None:
        self._types: dict[str, EventTypeDefinition] = {}
        for t in types or []:
            self.register(t)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._types

    def __len__(self) -> int:
        return len(self._types)

    def __iter__(self):
        return iter(self._types.values())

    def names(self) -> list[str]:
        return list(self._types.keys())

    def get(self, name: str) -> EventTypeDefinition | None:
        return self._types.get(name)

    def register(self, definition: EventTypeDefinition, *, overwrite: bool = False) -> None:
        """Register a new event type. Raises if name collides unless ``overwrite``."""
        if definition.name in self._types and not overwrite:
            raise ValueError(
                f"Event type '{definition.name}' already registered "
                "(pass overwrite=True to replace)"
            )
        self._types[definition.name] = definition

    def remove(self, name: str) -> None:
        if name not in self._types:
            raise ValueError(f"Event type '{name}' not registered")
        del self._types[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "event_types": [t.to_dict() for t in self._types.values()],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventTypeRegistry:
        types_data = data.get("event_types", []) or []
        types = [EventTypeDefinition.from_dict(t) for t in types_data]
        return cls(types=types)

    @classmethod
    def from_json(cls, filepath: str | Path) -> EventTypeRegistry:
        path = Path(filepath)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def default_event_registry() -> EventTypeRegistry:
    """Empty registry — the canonical starting point.

    Returning an empty registry by default keeps the system free of any
    pre-baked domain knowledge. Use :func:`example_event_registry` for
    a curated set of common business/political event types.
    """
    return EventTypeRegistry()


def example_event_registry() -> EventTypeRegistry:
    """Curated example registry covering common business / political events.

    These are *examples*, not defaults. The set covers the 10 event
    types mentioned in the project brief and is intended for
    demonstrations, tests, and quick starts. Production users are
    expected to either use this as a base (and ``register`` more types)
    or build their own registry from scratch.
    """
    types: list[EventTypeDefinition] = [
        EventTypeDefinition(
            name="Acquisition",
            description="One organization acquires another organization or asset.",
            roles=[
                EventRole(
                    name="acquirer",
                    description="Organization performing the acquisition",
                    entity_types=("Organization", "Company"),
                    cardinality="one",
                    required=True,
                ),
                EventRole(
                    name="acquired",
                    description="Organization or asset being acquired",
                    entity_types=("Organization", "Company", "Product", "Asset"),
                    cardinality="one",
                    required=True,
                ),
            ],
            properties={"deal_value": "monetary amount", "currency": "ISO currency"},
        ),
        EventTypeDefinition(
            name="Merger",
            description="Two organizations combine into a single entity.",
            roles=[
                EventRole(
                    name="parties",
                    description="Organizations participating in the merger",
                    entity_types=("Organization", "Company"),
                    cardinality="many",
                    required=True,
                ),
                EventRole(
                    name="resulting_entity",
                    description="Combined organization (if a new one is formed)",
                    entity_types=("Organization", "Company"),
                    cardinality="one",
                    required=False,
                ),
            ],
            properties={"deal_value": "monetary amount"},
        ),
        EventTypeDefinition(
            name="Funding",
            description="An organization raises investment capital.",
            roles=[
                EventRole(
                    name="recipient",
                    description="Organization receiving funding",
                    entity_types=("Organization", "Company"),
                    cardinality="one",
                    required=True,
                ),
                EventRole(
                    name="investors",
                    description="Investors providing the capital",
                    entity_types=("Organization", "Person", "Company"),
                    cardinality="many",
                    required=False,
                ),
            ],
            properties={
                "round": "Series A/B/Seed/etc.",
                "amount": "monetary amount",
                "currency": "ISO currency",
            },
        ),
        EventTypeDefinition(
            name="ProductLaunch",
            description="An organization releases a new product or service.",
            roles=[
                EventRole(
                    name="launcher",
                    description="Organization launching the product",
                    entity_types=("Organization", "Company"),
                    cardinality="one",
                    required=True,
                ),
                EventRole(
                    name="product",
                    description="Product or service being launched",
                    entity_types=("Product",),
                    cardinality="one",
                    required=True,
                ),
            ],
            properties={"market": "geographic market", "category": "product category"},
        ),
        EventTypeDefinition(
            name="Partnership",
            description="Two or more organizations form a strategic partnership.",
            roles=[
                EventRole(
                    name="parties",
                    description="Partners",
                    entity_types=("Organization", "Company"),
                    cardinality="many",
                    required=True,
                ),
            ],
            properties={"scope": "scope/area of cooperation"},
        ),
        EventTypeDefinition(
            name="LeadershipChange",
            description="A person assumes or leaves a leadership role at an organization.",
            roles=[
                EventRole(
                    name="organization",
                    description="Organization where the change occurs",
                    entity_types=("Organization", "Company"),
                    cardinality="one",
                    required=True,
                ),
                EventRole(
                    name="successor",
                    description="Person taking the role",
                    entity_types=("Person",),
                    cardinality="one",
                    required=False,
                ),
                EventRole(
                    name="predecessor",
                    description="Person leaving the role",
                    entity_types=("Person",),
                    cardinality="one",
                    required=False,
                ),
            ],
            properties={"role_title": "the leadership role (CEO, Chair, ...)"},
        ),
        EventTypeDefinition(
            name="Lawsuit",
            description="A legal action between parties.",
            roles=[
                EventRole(
                    name="plaintiff",
                    description="Party initiating the lawsuit",
                    entity_types=("Person", "Organization", "Company"),
                    cardinality="one",
                    required=True,
                ),
                EventRole(
                    name="defendant",
                    description="Party being sued",
                    entity_types=("Person", "Organization", "Company"),
                    cardinality="one",
                    required=True,
                ),
                EventRole(
                    name="venue",
                    description="Court or jurisdiction",
                    entity_types=("Location", "Organization"),
                    cardinality="one",
                    required=False,
                ),
            ],
            properties={"cause_of_action": "brief reason", "outcome": "resolution"},
        ),
        EventTypeDefinition(
            name="Election",
            description="A political or organizational election.",
            roles=[
                EventRole(
                    name="winner",
                    description="Winning candidate",
                    entity_types=("Person",),
                    cardinality="one",
                    required=False,
                ),
                EventRole(
                    name="candidates",
                    description="All candidates",
                    entity_types=("Person",),
                    cardinality="many",
                    required=False,
                ),
                EventRole(
                    name="position",
                    description="Office or position contested",
                    entity_types=("Position", "Role"),
                    cardinality="one",
                    required=False,
                ),
            ],
            properties={"jurisdiction": "geographic scope"},
        ),
        EventTypeDefinition(
            name="Employment",
            description="A person is hired by, or starts working at, an organization.",
            roles=[
                EventRole(
                    name="employee",
                    description="Person being employed",
                    entity_types=("Person",),
                    cardinality="one",
                    required=True,
                ),
                EventRole(
                    name="employer",
                    description="Hiring organization",
                    entity_types=("Organization", "Company"),
                    cardinality="one",
                    required=True,
                ),
            ],
            properties={"role_title": "job title"},
        ),
        EventTypeDefinition(
            name="Investment",
            description="A party invests in an asset or organization (non-equity-round).",
            roles=[
                EventRole(
                    name="investor",
                    description="Party making the investment",
                    entity_types=("Person", "Organization", "Company"),
                    cardinality="one",
                    required=True,
                ),
                EventRole(
                    name="target",
                    description="Asset or organization receiving the investment",
                    entity_types=("Organization", "Company", "Asset", "Product"),
                    cardinality="one",
                    required=True,
                ),
            ],
            properties={"amount": "monetary amount", "currency": "ISO currency"},
        ),
    ]
    return EventTypeRegistry(types=types)
