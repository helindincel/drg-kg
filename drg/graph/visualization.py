"""
Knowledge Graph Visualization Module

Provides visualization exports in multiple formats:
- Mermaid text format
- PyVis HTML format
- Configurable node colors, edge styles, and tooltips
"""

from pathlib import Path

from .kg_core import EnhancedKG, KGEdge, KGNode

# Default color mapping for entity types
DEFAULT_NODE_COLORS = {
    "Person": "#FF6B6B",
    "Location": "#4ECDC4",
    "Event": "#FFE66D",
    "Organization": "#95E1D3",
    "Product": "#F38181",
    "default": "#A8A8A8",
}

# Default relationship type colors
DEFAULT_EDGE_COLORS = {
    "influences": "#FF6B6B",
    "caused_by": "#4ECDC4",
    "located_at": "#95E1D3",
    "collaborates_with": "#FFE66D",
    "default": "#CCCCCC",
}


class KGVisualizer:
    """Knowledge Graph Visualizer for multiple export formats."""

    def __init__(
        self,
        kg: EnhancedKG,
        node_colors: dict[str, str] | None = None,
        edge_colors: dict[str, str] | None = None,
    ):
        self.kg = kg
        self.node_colors = node_colors or DEFAULT_NODE_COLORS.copy()
        self.edge_colors = edge_colors or DEFAULT_EDGE_COLORS.copy()

    def get_node_color(self, node: KGNode) -> str:
        """Get color for a node based on its type."""
        node_type = node.type or "default"
        return self.node_colors.get(node_type, self.node_colors.get("default", "#A8A8A8"))

    def get_edge_color(self, edge: KGEdge) -> str:
        """Get color for an edge based on its relationship type."""
        return self.edge_colors.get(
            edge.relationship_type, self.edge_colors.get("default", "#CCCCCC")
        )

    def to_mermaid(self, direction: str = "TD") -> str:
        """Export to Mermaid diagram format."""
        lines = [f"graph {direction}"]

        node_id_map = {node_id: f"N{i}" for i, node_id in enumerate(self.kg.nodes.keys())}

        # Add nodes with styling
        for node in self.kg.nodes.values():
            safe_id = node_id_map[node.id]
            node_type = node.type or "Entity"
            color = self.get_node_color(node)
            label = f"{node.id}<br/>({node_type})".replace('"', "'")
            lines.append(f'    {safe_id}["{label}"]')
            lines.append(f"    classDef style_{safe_id} fill:{color},stroke:#333,stroke-width:2px")
            lines.append(f"    class {safe_id} style_{safe_id}")

        # Add edges with labels and tooltips
        for edge in self.kg.edges:
            source_safe = node_id_map.get(edge.source, edge.source)
            target_safe = node_id_map.get(edge.target, edge.target)
            rel_type = edge.relationship_type.replace('"', "'")[:20]
            label = rel_type
            if len(edge.relationship_detail) <= 50:
                label = f"{rel_type}\\n({edge.relationship_detail[:50]})"
            lines.append(f'    {source_safe} -->|"{label}"| {target_safe}')

        return "\n".join(lines)

    def save_mermaid(self, filepath: str, direction: str = "TD") -> None:
        """Save Mermaid diagram to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_mermaid(direction=direction), encoding="utf-8")

    def to_pyvis_html(
        self,
        height: str = "800px",
        width: str = "100%",
        show_physics: bool = True,
    ) -> str:
        """Export to PyVis HTML format."""
        try:
            import pyvis.network as net
        except ImportError as err:
            raise ImportError("PyVis is required. Install with: pip install pyvis") from err

        g = net.Network(height=height, width=width, directed=True, notebook=False)

        if show_physics:
            g.set_options("""
            {
                "physics": {
                    "enabled": true,
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 200,
                        "springConstant": 0.08
                    }
                }
            }
            """)

        # Add nodes with colors and tooltips
        for node in self.kg.nodes.values():
            node_type = node.type or "Entity"
            color = self.get_node_color(node)
            tooltip_parts = [f"Type: {node_type}", f"ID: {node.id}"]
            if node.properties:
                tooltip_parts.append(f"Properties: {len(node.properties)}")
            if node.metadata:
                if "confidence" in node.metadata:
                    tooltip_parts.append(f"Confidence: {node.metadata['confidence']}")
            tooltip = "\\n".join(tooltip_parts)

            title = f"{node.id} ({node_type})"
            if node.properties:
                title += f"\\nProperties: {', '.join(list(node.properties.keys())[:5])}"

            g.add_node(
                node.id,
                label=f"{node.id}\\n({node_type})",
                color=color,
                title=title,
                size=20,
            )

        # Add edges with colors, labels, and tooltips
        for edge in self.kg.edges:
            color = self.get_edge_color(edge)
            tooltip = f"Type: {edge.relationship_type}\\nDetail: {edge.relationship_detail}"
            if edge.metadata:
                if "confidence" in edge.metadata:
                    tooltip += f"\\nConfidence: {edge.metadata['confidence']}"
                if "source_ref" in edge.metadata:
                    tooltip += f"\\nSource: {edge.metadata['source_ref']}"

            title = f"{edge.relationship_type}\\n{edge.relationship_detail}"
            label = edge.relationship_type[:15]

            g.add_edge(
                edge.source,
                edge.target,
                label=label,
                color=color,
                title=title,
                width=2,
            )

        return g.generate_html()

    def save_pyvis_html(self, filepath: str, height: str = "800px", width: str = "100%") -> None:
        """Save PyVis HTML visualization to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        html_content = self.to_pyvis_html(height=height, width=width)
        path.write_text(html_content, encoding="utf-8")
