"""Tests for projecting Events onto KGNode/KGEdge."""

from __future__ import annotations

import json

from drg.events import (
    EVENT_LOCATION_EDGE_TYPE,
    EVENT_NODE_TYPE_PREFIX,
    EVENT_ROLE_EDGE_PREFIX,
    Event,
    EventProvenance,
    EventTimestamp,
    TextSpan,
    event_from_kg_node,
    event_role_from_edge,
    event_to_kg_node,
    event_to_role_edges,
    events_to_kg_nodes_and_edges,
    is_event_node,
    is_event_role_edge,
)
from drg.graph.builders import build_enhanced_kg
from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode


def _sample_event(eid: str = "event:Acquisition:abc123") -> Event:
    return Event(
        id=eid,
        event_type="Acquisition",
        participants={"acquirer": ["Apple"], "acquired": ["Beats"]},
        timestamp=EventTimestamp(start="2014-05", end="2014-05", precision="month"),
        location="California",
        properties={"deal_value": "$3B"},
        provenance=EventProvenance(
            document_id="doc1",
            chunk_ids=["c1"],
            text_spans=[TextSpan(text="Apple acquired Beats", chunk_id="c1")],
            confidence=0.85,
        ),
    )


class TestEventToKgNode:
    def test_basic_projection(self):
        ev = _sample_event()
        node = event_to_kg_node(ev)
        assert node.id == ev.id
        assert node.type == f"{EVENT_NODE_TYPE_PREFIX}Acquisition"
        assert node.confidence == 0.85
        assert node.metadata["is_event"] is True
        assert node.metadata["event_type"] == "Acquisition"
        assert node.metadata["participants"] == {
            "acquirer": ["Apple"],
            "acquired": ["Beats"],
        }
        assert node.metadata["timestamp"]["start"] == "2014-05"
        assert node.metadata["location"] == "California"
        assert node.metadata["provenance"]["confidence"] == 0.85

    def test_is_event_node_detection(self):
        ev = _sample_event()
        node = event_to_kg_node(ev)
        assert is_event_node(node)

    def test_normal_node_is_not_event(self):
        node = KGNode(id="Apple", type="Company")
        assert not is_event_node(node)

    def test_event_node_without_prefix_but_with_flag(self):
        node = KGNode(id="x", type="Custom", metadata={"is_event": True})
        assert is_event_node(node)


class TestEventToRoleEdges:
    def test_role_edges_created(self):
        ev = _sample_event()
        edges = event_to_role_edges(ev)
        assert len(edges) == 3  # acquirer, acquired, occurred_at(California)
        role_edges = [e for e in edges if e.relationship_type.startswith(EVENT_ROLE_EDGE_PREFIX)]
        assert {e.relationship_type for e in role_edges} == {
            "role:acquirer",
            "role:acquired",
        }
        for e in role_edges:
            assert e.source == ev.id
            assert e.metadata["is_event_role"] is True
            assert e.metadata["event_id"] == ev.id

    def test_location_edge_created(self):
        ev = _sample_event()
        edges = event_to_role_edges(ev)
        loc_edges = [e for e in edges if e.relationship_type == EVENT_LOCATION_EDGE_TYPE]
        assert len(loc_edges) == 1
        assert loc_edges[0].target == "California"

    def test_no_location_no_location_edge(self):
        ev = Event(
            id="e1",
            event_type="X",
            participants={"r": ["Apple"]},
        )
        edges = event_to_role_edges(ev)
        assert len(edges) == 1
        assert edges[0].relationship_type == "role:r"

    def test_skips_self_loop_targets(self):
        ev = Event(
            id="e1",
            event_type="X",
            participants={"r": ["e1"]},  # would be a self-loop
        )
        edges = event_to_role_edges(ev)
        assert edges == []

    def test_role_edges_carry_timestamp(self):
        ev = _sample_event()
        edges = event_to_role_edges(ev)
        for e in edges:
            assert e.start_time == "2014-05"
            assert e.end_time == "2014-05"

    def test_event_role_helpers(self):
        ev = _sample_event()
        edges = event_to_role_edges(ev)
        for e in edges:
            if e.relationship_type.startswith("role:"):
                assert is_event_role_edge(e)
                assert event_role_from_edge(e) is not None
        normal = KGEdge(
            source="A", target="B", relationship_type="produces", relationship_detail="x"
        )
        assert not is_event_role_edge(normal)
        assert event_role_from_edge(normal) is None


class TestEventFromKgNode:
    def test_roundtrip_through_kg_node(self):
        ev = _sample_event()
        node = event_to_kg_node(ev)
        out = event_from_kg_node(node)
        assert out is not None
        assert out.id == ev.id
        assert out.event_type == ev.event_type
        assert out.participants == ev.participants
        assert out.timestamp is not None
        assert out.timestamp.start == ev.timestamp.start
        assert out.location == ev.location
        assert out.provenance.confidence == ev.provenance.confidence

    def test_returns_none_for_normal_node(self):
        node = KGNode(id="Apple", type="Company")
        assert event_from_kg_node(node) is None

    def test_handles_node_with_only_type_prefix(self):
        node = KGNode(
            id="ev1",
            type="Event:CustomEvent",
            metadata={"is_event": True},
        )
        out = event_from_kg_node(node)
        assert out is not None
        assert out.event_type == "CustomEvent"


class TestEventsToKgNodesAndEdges:
    def test_multiple_events(self):
        ev1 = _sample_event("event:Acquisition:e1")
        ev2 = Event(
            id="event:Funding:f1",
            event_type="Funding",
            participants={"recipient": ["Anthropic"], "investors": ["Amazon"]},
        )
        nodes, edges = events_to_kg_nodes_and_edges([ev1, ev2])
        assert len(nodes) == 2
        assert {n.id for n in nodes} == {"event:Acquisition:e1", "event:Funding:f1"}
        # ev1 has 2 role + 1 location, ev2 has 2 role
        assert len(edges) == 5


class TestBuilderIntegration:
    def test_build_enhanced_kg_with_events(self):
        ev = _sample_event()
        kg = build_enhanced_kg(
            entities_typed=[
                ("Apple", "Company"),
                ("Beats", "Company"),
                ("California", "Location"),
            ],
            triples=[],
            events=[ev],
        )
        assert ev.id in kg.nodes
        assert kg.nodes[ev.id].type.startswith("Event:")
        # Apple, Beats, California already exist; event adds 3 edges
        edges_from_event = [e for e in kg.edges if e.source == ev.id]
        assert len(edges_from_event) == 3

    def test_build_enhanced_kg_auto_creates_missing_participants(self):
        ev = Event(
            id="ev1",
            event_type="Funding",
            participants={"recipient": ["Anthropic"]},
        )
        kg = build_enhanced_kg(
            entities_typed=[],
            triples=[],
            events=[ev],
        )
        assert "Anthropic" in kg.nodes
        assert "ev1" in kg.nodes

    def test_build_enhanced_kg_without_events_unchanged(self):
        # Regression: behavior without events param identical to legacy.
        kg_old = build_enhanced_kg(
            entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
            triples=[("Apple", "produces", "iPhone")],
        )
        kg_new = build_enhanced_kg(
            entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
            triples=[("Apple", "produces", "iPhone")],
            events=None,
        )
        assert set(kg_old.nodes.keys()) == set(kg_new.nodes.keys())
        assert len(kg_old.edges) == len(kg_new.edges)


class TestSerialization:
    def test_to_json_includes_events_when_present(self):
        ev = _sample_event()
        kg = build_enhanced_kg(
            entities_typed=[("Apple", "Company"), ("Beats", "Company")],
            triples=[],
            events=[ev],
        )
        data = json.loads(kg.to_json())
        assert "events" in data
        assert len(data["events"]) == 1
        assert data["events"][0]["event_type"] == "Acquisition"

    def test_to_json_omits_events_when_absent(self):
        # Regression: legacy graphs serialize byte-identically.
        kg = build_enhanced_kg(
            entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
            triples=[("Apple", "produces", "iPhone")],
        )
        data = json.loads(kg.to_json())
        assert "events" not in data
        assert set(data.keys()) == {"nodes", "edges", "clusters"}

    def test_to_enriched_format_includes_events(self):
        ev = _sample_event()
        kg = build_enhanced_kg(
            entities_typed=[("Apple", "Company"), ("Beats", "Company")],
            triples=[],
            events=[ev],
        )
        data = json.loads(kg.to_enriched_format())
        assert "events" in data

    def test_to_json_ld_includes_events(self):
        ev = _sample_event()
        kg = build_enhanced_kg(
            entities_typed=[("Apple", "Company"), ("Beats", "Company")],
            triples=[],
            events=[ev],
        )
        data = json.loads(kg.to_json_ld())
        assert "events" in data
        assert data["events"][0]["@type"] == "Event"
        assert data["events"][0]["startDate"] == "2014-05"

    def test_save_load_roundtrip(self, tmp_path):
        ev = _sample_event()
        kg = build_enhanced_kg(
            entities_typed=[("Apple", "Company"), ("Beats", "Company")],
            triples=[],
            events=[ev],
        )
        path = tmp_path / "kg.json"
        kg.save_json(str(path))
        loaded = EnhancedKG.load_json(str(path))
        assert ev.id in loaded.nodes
        # The loaded node still has the event metadata intact
        assert loaded.nodes[ev.id].metadata.get("is_event") is True
