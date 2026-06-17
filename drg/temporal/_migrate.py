"""Migration helpers for existing graphs without temporal metadata."""

from __future__ import annotations

from typing import Any

from ._types import TemporalScope

__all__ = [
    "migrate_edge_dict",
    "migrate_node_dict",
]


def migrate_edge_dict(edge: dict[str, Any]) -> dict[str, Any]:
    """Normalise legacy ``start_time`` / ``end_time`` into temporal scope.

    Idempotent — safe to run on already-migrated edges.
    """
    out = dict(edge)
    start = out.get("start_time") or out.get("valid_from")
    end = out.get("end_time") or out.get("valid_to")
    created = out.get("created_at")
    updated = out.get("updated_at")

    meta = dict(out.get("metadata") or {})
    if "temporal" not in meta and (start or end or created or updated):
        scope = TemporalScope(
            valid_from=start,
            valid_to=end,
            created_at=created,
            updated_at=updated,
        )
        if not scope.is_empty():
            meta["temporal"] = scope.to_dict()
            out["metadata"] = meta

    if start and "valid_from" not in out:
        out["valid_from"] = start
    if end and "valid_to" not in out:
        out["valid_to"] = end

    return out


def migrate_node_dict(node: dict[str, Any]) -> dict[str, Any]:
    """Ensure node metadata can carry a ``temporal`` block."""
    out = dict(node)
    meta = dict(out.get("metadata") or {})
    if "temporal" in meta:
        out["metadata"] = meta
        return out

    props = out.get("properties") or {}
    vf = props.pop("valid_from", None) if isinstance(props, dict) else None
    vt = props.pop("valid_to", None) if isinstance(props, dict) else None
    if vf or vt:
        scope = TemporalScope(valid_from=vf, valid_to=vt)
        meta["temporal"] = scope.to_dict()
        out["metadata"] = meta
        if props:
            out["properties"] = props
    return out
