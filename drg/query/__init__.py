"""Query & reasoning layer for :class:`drg.graph.kg_core.EnhancedKG`.

This package provides an evidence-first, deterministic query interface on top
of the existing graph model. It is deliberately separate from graph generation
(``drg.extract``, ``drg.graph.builders``) and from multi-document inference
(``drg.reasoning``).

Public surface
==============

- :class:`GraphQuery` — main facade (``neighbors``, ``find_paths``, ``explain``, …)
- :class:`InMemoryBackend` — default backend over ``EnhancedKG``
- :class:`QueryBackend` — protocol for future Neo4j integration
- Result types: :class:`EntityView`, :class:`EdgeView`, :class:`GraphPath`,
  :class:`Explanation`, :class:`EvidenceBundle`, …

Quick example
=============

::

    from drg.query import GraphQuery

    gq = GraphQuery.from_json("outputs/global_kg.json")
    print(gq.explain("Apple", "Jimmy Iovine").summary)
    for company in gq.related_entities("Apple", entity_type="Company"):
        print(company.entity.id, company.score)

See ``docs/query_layer.md`` for the full API catalog.
"""

from __future__ import annotations

from ._backend import QueryBackend
from ._engine import GraphQuery
from ._memory import InMemoryBackend
from ._types import (
    CommunityView,
    EdgeView,
    EntityMatch,
    EntityView,
    EventView,
    EvidenceBundle,
    EvidenceItem,
    Explanation,
    GraphMetricScore,
    GraphPath,
    NeighborhoodView,
    Provenance,
    QueryAnswer,
    QueryError,
    RelatedEntityMatch,
)

__all__ = [
    "CommunityView",
    "EdgeView",
    "EntityMatch",
    "EntityView",
    "EventView",
    "EvidenceBundle",
    "EvidenceItem",
    "Explanation",
    "GraphMetricScore",
    "GraphPath",
    "GraphQuery",
    "InMemoryBackend",
    "NeighborhoodView",
    "Provenance",
    "QueryAnswer",
    "QueryBackend",
    "QueryError",
    "RelatedEntityMatch",
]
