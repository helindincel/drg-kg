"""Thin wrapper around ``drg.graph.query_engine.execute_query``.

Kept in a separate module so ``_search.py`` does not create a circular import
with the graph package's public surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph.kg_core import EnhancedKG
    from ..graph.query_engine import QueryResult


def execute_query_compat(
    kg: EnhancedKG,
    query: str,
    *,
    k_entities: int = 10,
    k_edges: int = 40,
) -> QueryResult:
    from ..graph.query_engine import execute_query

    return execute_query(kg, query, k_entities=k_entities, k_edges=k_edges)
