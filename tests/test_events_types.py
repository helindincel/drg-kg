"""Unit tests for drg.events._types data model.

Pure-data tests — no LLM, no IO, fully deterministic.
"""

from __future__ import annotations

import pytest

from drg.events import (
    Event,
    EventProvenance,
    EventRole,
    EventTimestamp,
    EventTypeDefinition,
    TextSpan,
)


class TestEventRole:
    def test_basic_role(self):
        role = EventRole(name="acquirer", description="who acquires")
        assert role.name == "acquirer"
        assert role.cardinality == "one"
        assert role.required is True
        assert role.entity_types == ()

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            EventRole(name="")
        with pytest.raises(ValueError, match="name cannot be empty"):
            EventRole(name="   ")

    def test_invalid_cardinality_raises(self):
        with pytest.raises(ValueError, match="cardinality"):
            EventRole(name="x", cardinality="bogus")  # type: ignore[arg-type]

    def test_accepts_when_no_types_means_any(self):
        role = EventRole(name="x")
        assert role.accepts("Person")
        assert role.accepts("Anything")
        assert role.accepts(None)

    def test_accepts_typed(self):
        role = EventRole(name="x", entity_types=("Person", "Organization"))
        assert role.accepts("Person")
        assert role.accepts("Organization")
        assert not role.accepts("Product")

    def test_to_from_dict_roundtrip(self):
        role = EventRole(
            name="r",
            description="d",
            entity_types=("A", "B"),
            cardinality="many",
            required=False,
        )
        out = EventRole.from_dict(role.to_dict())
        assert out == role


class TestEventTypeDefinition:
    def test_basic_definition(self):
        td = EventTypeDefinition(name="Acquisition", description="A acquires B")
        assert td.name == "Acquisition"
        assert td.roles == []
        assert td.required_roles() == []

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            EventTypeDefinition(name="", description="d")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            EventTypeDefinition(name="X", description="")

    def test_duplicate_role_raises(self):
        with pytest.raises(ValueError, match="Duplicate role"):
            EventTypeDefinition(
                name="X",
                description="d",
                roles=[EventRole(name="r"), EventRole(name="r")],
            )

    def test_get_role(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[EventRole(name="a"), EventRole(name="b", required=False)],
        )
        assert td.get_role("a") is not None
        assert td.get_role("missing") is None

    def test_required_roles(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[
                EventRole(name="a", required=True),
                EventRole(name="b", required=False),
                EventRole(name="c", required=True),
            ],
        )
        assert {r.name for r in td.required_roles()} == {"a", "c"}

    def test_to_from_dict_roundtrip(self):
        td = EventTypeDefinition(
            name="Acq",
            description="d",
            roles=[EventRole(name="acquirer", entity_types=("Org",))],
            properties={"deal_value": "amount"},
            examples=["foo bought bar"],
        )
        out = EventTypeDefinition.from_dict(td.to_dict())
        assert out.name == td.name
        assert out.description == td.description
        assert len(out.roles) == 1
        assert out.roles[0].name == "acquirer"
        assert out.properties == {"deal_value": "amount"}
        assert out.examples == ["foo bought bar"]


class TestEventTimestamp:
    def test_basic(self):
        ts = EventTimestamp(start="2014", end="2014", precision="year", raw_text="2014")
        assert ts.start == "2014"
        assert not ts.is_empty()

    def test_empty_timestamp(self):
        ts = EventTimestamp()
        assert ts.is_empty()

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match="precision"):
            EventTimestamp(precision="century")  # type: ignore[arg-type]

    def test_to_from_dict_roundtrip(self):
        ts = EventTimestamp(start="2014-05", end="2014-05", precision="month", raw_text="May 2014")
        out = EventTimestamp.from_dict(ts.to_dict())
        assert out == ts

    def test_to_dict_omits_none_fields(self):
        ts = EventTimestamp(start="2014")
        d = ts.to_dict()
        assert "start" in d
        assert "end" not in d
        assert "raw_text" not in d


class TestTextSpan:
    def test_basic(self):
        span = TextSpan(text="hello world", chunk_id="c1", start=0, end=11)
        assert span.text == "hello world"

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            TextSpan(text="")
        with pytest.raises(ValueError, match="text"):
            TextSpan(text="   ")

    def test_negative_start_raises(self):
        with pytest.raises(ValueError, match="start"):
            TextSpan(text="x", start=-1)

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="end"):
            TextSpan(text="x", start=10, end=5)

    def test_optional_offsets(self):
        span = TextSpan(text="x")
        assert span.start is None
        assert span.end is None

    def test_to_from_dict_roundtrip(self):
        span = TextSpan(text="hello", chunk_id="c", start=0, end=5)
        out = TextSpan.from_dict(span.to_dict())
        assert out == span


class TestEventProvenance:
    def test_basic(self):
        prov = EventProvenance(document_id="d1", confidence=0.8)
        assert prov.confidence == 0.8

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            EventProvenance(confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            EventProvenance(confidence=-0.1)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="extraction_method"):
            EventProvenance(extraction_method="bogus")  # type: ignore[arg-type]

    def test_to_from_dict_roundtrip(self):
        prov = EventProvenance(
            document_id="doc1",
            chunk_ids=["c1", "c2"],
            text_spans=[TextSpan(text="evidence")],
            extracted_at="2024-01-01T00:00:00Z",
            extractor_version="1.0",
            extraction_method="llm",
            confidence=0.7,
        )
        out = EventProvenance.from_dict(prov.to_dict())
        assert out.document_id == "doc1"
        assert out.chunk_ids == ["c1", "c2"]
        assert len(out.text_spans) == 1
        assert out.text_spans[0].text == "evidence"
        assert out.confidence == 0.7


class TestEvent:
    def test_basic_event(self):
        ev = Event(
            id="e1",
            event_type="Acquisition",
            participants={"acquirer": ["Apple"], "acquired": ["Beats"]},
        )
        assert ev.id == "e1"
        assert ev.confidence == 1.0
        assert ev.participant_entities() == ["Apple", "Beats"]

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="id"):
            Event(id="", event_type="X")
        with pytest.raises(ValueError, match="id"):
            Event(id="   ", event_type="X")

    def test_empty_event_type_raises(self):
        with pytest.raises(ValueError, match="event_type"):
            Event(id="e1", event_type="")

    def test_participants_must_be_lists(self):
        with pytest.raises(ValueError, match="must be a list"):
            Event(
                id="e1",
                event_type="X",
                participants={"role": "not-a-list"},  # type: ignore[dict-item]
            )

    def test_participant_entities_dedup(self):
        ev = Event(
            id="e1",
            event_type="X",
            participants={"a": ["Apple", "Apple"], "b": ["Apple", "Beats"]},
        )
        assert ev.participant_entities() == ["Apple", "Beats"]

    def test_to_dict_omits_optional_fields_when_unset(self):
        ev = Event(id="e1", event_type="X")
        d = ev.to_dict()
        assert "timestamp" not in d
        assert "location" not in d

    def test_to_dict_includes_full_payload(self):
        ev = Event(
            id="e1",
            event_type="Acquisition",
            participants={"acquirer": ["Apple"]},
            timestamp=EventTimestamp(start="2014", precision="year"),
            location="California",
            properties={"deal_value": "$3B"},
            provenance=EventProvenance(document_id="doc1", confidence=0.9),
            metadata={"custom": "tag"},
        )
        d = ev.to_dict()
        assert d["id"] == "e1"
        assert d["event_type"] == "Acquisition"
        assert d["participants"] == {"acquirer": ["Apple"]}
        assert d["timestamp"]["start"] == "2014"
        assert d["location"] == "California"
        assert d["properties"] == {"deal_value": "$3B"}
        assert d["provenance"]["confidence"] == 0.9
        assert d["metadata"] == {"custom": "tag"}

    def test_to_from_dict_roundtrip(self):
        ev = Event(
            id="e1",
            event_type="Acquisition",
            participants={"acquirer": ["Apple"], "acquired": ["Beats"]},
            timestamp=EventTimestamp(start="2014-05", end="2014-05", precision="month"),
            location="California",
            properties={"deal_value": "$3B"},
            provenance=EventProvenance(
                document_id="doc1",
                text_spans=[TextSpan(text="Apple acquired Beats")],
                confidence=0.85,
            ),
        )
        out = Event.from_dict(ev.to_dict())
        assert out.id == ev.id
        assert out.event_type == ev.event_type
        assert out.participants == ev.participants
        assert out.timestamp == ev.timestamp
        assert out.location == ev.location
        assert out.properties == ev.properties
        assert out.provenance.confidence == 0.85
        assert len(out.provenance.text_spans) == 1
