"""
Legacy KG class — preserved for backward compatibility.

.. deprecated::
    Use :class:`drg.graph.kg_core.EnhancedKG` instead.
    The :class:`KG` class will be removed in a future major version.

Migration guide
---------------
Before::

    from drg.graph import KG
    kg = KG.from_typed(entities_typed, triples)

After::

    from drg.graph import EnhancedKG, KGNode, KGEdge
    kg = EnhancedKG()
    for name, etype in entities_typed:
        kg.add_node(KGNode(id=name, type=etype))
    for s, r, o in triples:
        kg.add_edge(KGEdge(source=s, target=o, relationship_type=r, relationship_detail=r))
"""

import json
import warnings
from typing import Any


class KG:
    """Simple Knowledge Graph class.

    .. deprecated::
        Use :class:`~drg.graph.kg_core.EnhancedKG` instead.
        This class will be removed in a future major version.
    """

    def __init__(self) -> None:
        warnings.warn(
            "KG is deprecated and will be removed in a future major version. "
            "Use EnhancedKG from drg.graph instead. "
            "See drg/graph/_legacy.py for a migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []

    @classmethod
    def from_typed(
        cls,
        entities_typed: list[tuple[str, str]],
        triples: list[tuple[str, str, str]],
    ) -> "KG":
        kg = cls()
        for name, etype in entities_typed:
            kg.nodes.setdefault(name, {"type": etype})
        for s, r, o in triples:
            kg.nodes.setdefault(s, {"type": None})
            kg.nodes.setdefault(o, {"type": None})
            kg.edges.append((s, r, o))
        return kg

    @classmethod
    def from_triples(cls, triples: list[tuple[str, str, str]]) -> "KG":
        kg = cls()
        for s, r, o in triples:
            kg.nodes.setdefault(s, {"type": None})
            kg.nodes.setdefault(o, {"type": None})
            kg.edges.append((s, r, o))
        return kg

    def to_json(self, indent: int = 2) -> str:
        data = {
            "nodes": [{"id": n, **attr} for n, attr in self.nodes.items()],
            "edges": [{"source": s, "type": r, "target": o} for s, r, o in self.edges],
        }
        return json.dumps(data, indent=indent)
