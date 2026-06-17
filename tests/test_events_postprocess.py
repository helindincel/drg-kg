"""Unit tests for drg.events._postprocess."""

from __future__ import annotations

import pytest

from drg.events import (
    EventRole,
    EventTypeDefinition,
    EventTimestamp,
    build_event,
    find_evidence_span,
    has_all_required_roles,
    make_event_id,
    normalize_participants,
    parse_timestamp,
    required_role_coverage,
)


class TestParseTimestamp:
    def test_none_or_empty(self):
        assert parse_timestamp(None) is None
        assert parse_timestamp("") is None
        assert parse_timestamp("   ") is None

    def test_year_only(self):
        ts = parse_timestamp("2014")
        assert ts is not None
        assert ts.start == "2014"
        assert ts.precision == "year"

    def test_iso_full_date(self):
        ts = parse_timestamp("2014-05-28")
        assert ts is not None
        assert ts.start == "2014-05-28"
        assert ts.precision == "day"

    def test_iso_year_month(self):
        ts = parse_timestamp("2014-05")
        assert ts is not None
        assert ts.start == "2014-05"
        assert ts.precision == "month"

    def test_month_day_year(self):
        ts = parse_timestamp("May 28, 2014")
        assert ts is not None
        assert ts.start == "2014-05-28"
        assert ts.precision == "day"

    def test_day_month_year(self):
        ts = parse_timestamp("28 May 2014")
        assert ts is not None
        assert ts.start == "2014-05-28"
        assert ts.precision == "day"

    def test_month_year(self):
        ts = parse_timestamp("May 2014")
        assert ts is not None
        assert ts.start == "2014-05"
        assert ts.precision == "month"

    def test_year_range(self):
        ts = parse_timestamp("2014-2018")
        assert ts is not None
        assert ts.start == "2014"
        assert ts.end == "2018"
        assert ts.precision == "year"

    def test_unparseable_returns_none(self):
        assert parse_timestamp("yesterday") is None
        assert parse_timestamp("some time") is None

    def test_year_in_sentence_is_extracted(self):
        ts = parse_timestamp("In late 2014, Apple acquired Beats")
        assert ts is not None
        assert ts.start == "2014"

    def test_raw_text_preserved(self):
        ts = parse_timestamp("late 2014")
        assert ts is not None
        assert ts.raw_text == "late 2014"


class TestNormalizeParticipants:
    def test_empty_input(self):
        assert normalize_participants(None, type_def=None, entity_type_map={}) == {}
        assert normalize_participants({}, type_def=None, entity_type_map={}) == {}

    def test_string_value_wrapped_to_list(self):
        out = normalize_participants(
            {"role": "Apple"},
            type_def=None,
            entity_type_map={"Apple": "Company"},
        )
        assert out == {"role": ["Apple"]}

    def test_list_value(self):
        out = normalize_participants(
            {"parties": ["Apple", "Beats"]},
            type_def=None,
            entity_type_map={"Apple": "Company", "Beats": "Company"},
        )
        assert out == {"parties": ["Apple", "Beats"]}

    def test_canonicalizes_case(self):
        out = normalize_participants(
            {"role": ["apple"]},
            type_def=None,
            entity_type_map={"Apple": "Company"},
        )
        assert out == {"role": ["Apple"]}

    def test_drops_unknown_role_when_type_def_provided(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[EventRole(name="known")],
        )
        out = normalize_participants(
            {"known": ["A"], "unknown": ["B"]},
            type_def=td,
            entity_type_map={"A": "T", "B": "T"},
        )
        assert "known" in out
        assert "unknown" not in out

    def test_dedups_same_canonical_form(self):
        out = normalize_participants(
            {"role": ["Apple", "apple", "APPLE"]},
            type_def=None,
            entity_type_map={"Apple": "Company"},
        )
        assert out == {"role": ["Apple"]}

    def test_drops_empty_role_lists(self):
        out = normalize_participants(
            {"role": []},
            type_def=None,
            entity_type_map={"Apple": "Company"},
        )
        assert out == {}


class TestRequiredRoleCoverage:
    def test_no_type_def_returns_one(self):
        assert required_role_coverage(None, {}) == 1.0

    def test_no_required_roles_returns_one(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[EventRole(name="r", required=False)],
        )
        assert required_role_coverage(td, {}) == 1.0

    def test_partial_coverage(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[
                EventRole(name="a", required=True),
                EventRole(name="b", required=True),
            ],
        )
        assert required_role_coverage(td, {"a": ["x"]}) == pytest.approx(0.5)

    def test_full_coverage(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[
                EventRole(name="a", required=True),
                EventRole(name="b", required=True),
            ],
        )
        assert required_role_coverage(td, {"a": ["x"], "b": ["y"]}) == 1.0


class TestHasAllRequiredRoles:
    def test_no_type_def(self):
        assert has_all_required_roles(None, {}) is True

    def test_missing_required(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[EventRole(name="a", required=True)],
        )
        assert has_all_required_roles(td, {}) is False
        assert has_all_required_roles(td, {"a": []}) is False
        assert has_all_required_roles(td, {"a": ["x"]}) is True


class TestFindEvidenceSpan:
    def test_none_inputs(self):
        assert find_evidence_span("", ["a"]) is None
        assert find_evidence_span("text", []) is None

    def test_no_match(self):
        assert find_evidence_span("hello world", ["nothing"]) is None

    def test_single_entity(self):
        text = "Apple was founded in 1976. Many things happened later."
        span = find_evidence_span(text, ["Apple"])
        assert span is not None
        assert "Apple" in span.text

    def test_two_entities_picks_close_window(self):
        text = (
            "Apple acquired Beats in 2014. "
            + ("filler " * 200)
            + " Apple again, Beats again."
        )
        span = find_evidence_span(text, ["Apple", "Beats"])
        assert span is not None
        assert "Apple" in span.text
        assert "Beats" in span.text

    def test_case_insensitive(self):
        span = find_evidence_span("APPLE bought BEATS", ["apple", "beats"])
        assert span is not None


class TestMakeEventId:
    def test_deterministic(self):
        a = make_event_id(
            "Acquisition",
            {"acquirer": ["Apple"], "acquired": ["Beats"]},
            EventTimestamp(start="2014"),
            "Apple acquired Beats",
        )
        b = make_event_id(
            "Acquisition",
            {"acquirer": ["Apple"], "acquired": ["Beats"]},
            EventTimestamp(start="2014"),
            "Apple acquired Beats",
        )
        assert a == b
        assert a.startswith("event:Acquisition:")

    def test_different_inputs_yield_different_ids(self):
        a = make_event_id(
            "Acquisition", {"acquirer": ["Apple"]}, None, "x"
        )
        b = make_event_id(
            "Acquisition", {"acquirer": ["Google"]}, None, "x"
        )
        assert a != b

    def test_role_order_does_not_matter(self):
        a = make_event_id(
            "X",
            {"a": ["A"], "b": ["B"]},
            None,
            None,
        )
        b = make_event_id(
            "X",
            {"b": ["B"], "a": ["A"]},
            None,
            None,
        )
        assert a == b

    def test_safe_type_segment(self):
        eid = make_event_id("Custom Type!@#", {"r": ["x"]}, None, None)
        assert eid.startswith("event:CustomType:")


class TestBuildEvent:
    def test_basic_event_built(self):
        td = EventTypeDefinition(
            name="Acquisition",
            description="d",
            roles=[
                EventRole(name="acquirer", required=True),
                EventRole(name="acquired", required=True),
            ],
        )
        ev = build_event(
            event_type="Acquisition",
            raw_participants={"acquirer": "Apple", "acquired": "Beats"},
            raw_timestamp="May 2014",
            location=None,
            properties={"deal_value": "$3B"},
            evidence_text="Apple acquired Beats",
            type_def=td,
            entity_type_map={"Apple": "Company", "Beats": "Company"},
            source_text="Apple acquired Beats in May 2014.",
            document_id="doc1",
        )
        assert ev is not None
        assert ev.event_type == "Acquisition"
        assert ev.participants == {"acquirer": ["Apple"], "acquired": ["Beats"]}
        assert ev.timestamp is not None
        assert ev.timestamp.start == "2014-05"
        assert ev.properties == {"deal_value": "$3B"}
        assert ev.provenance.document_id == "doc1"
        assert len(ev.provenance.text_spans) == 1

    def test_drops_when_required_role_missing(self):
        td = EventTypeDefinition(
            name="Acquisition",
            description="d",
            roles=[
                EventRole(name="acquirer", required=True),
                EventRole(name="acquired", required=True),
            ],
        )
        ev = build_event(
            event_type="Acquisition",
            raw_participants={"acquirer": "Apple"},
            raw_timestamp=None,
            location=None,
            properties=None,
            evidence_text=None,
            type_def=td,
            entity_type_map={"Apple": "Company"},
        )
        assert ev is None

    def test_drops_when_no_participants_resolved(self):
        td = EventTypeDefinition(
            name="X", description="d", roles=[EventRole(name="r", required=False)]
        )
        ev = build_event(
            event_type="X",
            raw_participants={"unknown_role": "Z"},
            raw_timestamp=None,
            location=None,
            properties=None,
            evidence_text=None,
            type_def=td,
            entity_type_map={"Z": "T"},
        )
        assert ev is None

    def test_clamps_invalid_llm_confidence(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[EventRole(name="r", required=False)],
        )
        ev = build_event(
            event_type="X",
            raw_participants={"r": "Z"},
            raw_timestamp=None,
            location=None,
            properties=None,
            evidence_text=None,
            type_def=td,
            entity_type_map={"Z": "T"},
            llm_confidence=2.5,
        )
        assert ev is not None
        assert ev.confidence == 1.0

    def test_falls_back_when_llm_confidence_invalid(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[EventRole(name="r", required=False)],
        )
        ev = build_event(
            event_type="X",
            raw_participants={"r": "Z"},
            raw_timestamp=None,
            location=None,
            properties=None,
            evidence_text=None,
            type_def=td,
            entity_type_map={"Z": "T"},
            llm_confidence="not-a-float",  # type: ignore[arg-type]
        )
        assert ev is not None
        assert 0.0 <= ev.confidence <= 1.0

    def test_evidence_falls_back_to_source_text_search(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[EventRole(name="r", required=False)],
        )
        ev = build_event(
            event_type="X",
            raw_participants={"r": "Apple"},
            raw_timestamp=None,
            location=None,
            properties=None,
            evidence_text=None,  # nothing explicit
            type_def=td,
            entity_type_map={"Apple": "Company"},
            source_text="Yesterday Apple announced something interesting.",
        )
        assert ev is not None
        assert len(ev.provenance.text_spans) >= 1
        assert "Apple" in ev.provenance.text_spans[0].text

    def test_id_is_deterministic(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[EventRole(name="r", required=False)],
        )
        kwargs = dict(
            event_type="X",
            raw_participants={"r": "Apple"},
            raw_timestamp="2014",
            location=None,
            properties=None,
            evidence_text="Apple did something",
            type_def=td,
            entity_type_map={"Apple": "Company"},
        )
        a = build_event(**kwargs)  # type: ignore[arg-type]
        b = build_event(**kwargs)  # type: ignore[arg-type]
        assert a is not None and b is not None
        assert a.id == b.id

    def test_canonicalizes_location(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[EventRole(name="r", required=False)],
        )
        ev = build_event(
            event_type="X",
            raw_participants={"r": "Apple"},
            raw_timestamp=None,
            location="california",
            properties=None,
            evidence_text=None,
            type_def=td,
            entity_type_map={"Apple": "Company", "California": "Location"},
        )
        assert ev is not None
        assert ev.location == "California"

    def test_cleaned_properties_drop_empty_values(self):
        td = EventTypeDefinition(
            name="X",
            description="d",
            roles=[EventRole(name="r", required=False)],
        )
        ev = build_event(
            event_type="X",
            raw_participants={"r": "Apple"},
            raw_timestamp=None,
            location=None,
            properties={"deal_value": "$3B", "currency": "", "extras": [], "note": None},
            evidence_text=None,
            type_def=td,
            entity_type_map={"Apple": "Company"},
        )
        assert ev is not None
        assert ev.properties == {"deal_value": "$3B"}
