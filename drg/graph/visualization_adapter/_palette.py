"""Static color palettes for KG and provenance visualization.

Pure functions — no I/O, no state. Centralizing them here means front-end
contributors have a single place to tweak visual theming without touching the
adapter dispatcher.
"""

from __future__ import annotations

__all__ = [
    "COMMUNITY_COLORS",
    "EDGE_COLORS",
    "NODE_COLORS",
    "PROVENANCE_COLORS",
    "get_community_color",
    "get_edge_color",
    "get_node_color",
    "get_provenance_color",
]


NODE_COLORS: dict[str, str] = {
    "Person": "#FF6B6B",
    "Location": "#4ECDC4",
    "Event": "#FFE66D",
    "Organization": "#95E1D3",
    "Product": "#F38181",
    "Company": "#95E1D3",
    "default": "#A8A8A8",
}

EDGE_COLORS: dict[str, str] = {
    "influences": "#FF6B6B",
    "caused_by": "#4ECDC4",
    "located_at": "#95E1D3",
    "collaborates_with": "#FFE66D",
    "works_with": "#FFE66D",
    "default": "#CCCCCC",
}

# Cycle through this palette by community index for stable, distinguishable colors.
COMMUNITY_COLORS: tuple[str, ...] = (
    "#FF6B6B",
    "#4ECDC4",
    "#FFE66D",
    "#95E1D3",
    "#F38181",
    "#A8E6CF",
    "#FFD3B6",
    "#FFAAA5",
    "#FF8B94",
    "#C7CEEA",
)

PROVENANCE_COLORS: dict[str, str] = {
    "query": "#FF6B6B",
    "chunk": "#4ECDC4",
    "community": "#FFE66D",
    "summary": "#95E1D3",
    "answer": "#F38181",
}


def get_node_color(node_type: str | None) -> str:
    """Return the color string for ``node_type``. Falls back to the default grey."""
    if node_type is None:
        return NODE_COLORS["default"]
    return NODE_COLORS.get(node_type, NODE_COLORS["default"])


def get_edge_color(relationship_type: str) -> str:
    """Return the color string for ``relationship_type``. Falls back to neutral grey."""
    return EDGE_COLORS.get(relationship_type, EDGE_COLORS["default"])


def get_community_color(index: int) -> str:
    """Cycle through :data:`COMMUNITY_COLORS` so each community has a stable color."""
    return COMMUNITY_COLORS[index % len(COMMUNITY_COLORS)]


def get_provenance_color(node_type: str) -> str:
    """Return color for a provenance node type. Defaults to ``#A8A8A8`` grey."""
    return PROVENANCE_COLORS.get(node_type, "#A8A8A8")
