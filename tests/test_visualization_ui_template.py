"""Regression tests for the bundled DRG visualization UI template.

These tests intentionally avoid a browser runtime. The UI is shipped as a
single static template, so the most stable contract is that the controls,
client-side behavior hooks, and existing API endpoint calls remain present.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATE = Path("drg/api/templates/index.html")


def _template_text() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def test_empty_state_guides_user_to_load_graph():
    html = _template_text()

    assert "No graph loaded" in html
    assert "Use Load Full Graph" in html
    assert "python examples/api_server_example.py" in html


def test_entity_search_autocomplete_and_zoom_controls_exist():
    html = _template_text()

    assert 'id="entity-search"' in html
    assert 'list="entity-options"' in html
    assert 'id="entity-options"' in html
    assert "function updateEntitySearch()" in html
    assert "function zoomToSelectedEntity()" in html
    assert "handleEntitySearchKeydown(event)" in html


def test_details_panels_include_node_edge_provenance_fields():
    html = _template_text()

    for expected in (
        "renderNodeRelations(node)",
        "Name:",
        "<strong>Incoming</strong>",
        "<strong>Outgoing</strong>",
        "document_id",
        "sentence_id",
        "snippet",
        "extracted_at",
        "Source:",
        "Target:",
        "Confidence:",
        "relationship_description",
    ):
        assert expected in html


def test_readability_controls_are_client_side_only():
    html = _template_text()

    assert "function prepareGraphElements(elements = [])" in html
    assert "'width': 'mapData(connection_count, 0, 12, 16, 52)'" in html
    assert "function renderLegend()" in html
    assert 'id="legend-content"' in html

    # Existing visualization API calls should stay unchanged apart from the
    # already-supported hub_split query params.
    assert "/api/visualization/cytoscape" in html
    assert "/api/visualization/communities/cytoscape" in html
    assert "/api/graph/stats" in html
