"""Tests for drg.events._extraction (mocked DSPy)."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from drg.events import (
    EventRole,
    EventTypeDefinition,
    EventTypeRegistry,
    extract_events,
)


@pytest.fixture
def acquisition_registry():
    return EventTypeRegistry(
        types=[
            EventTypeDefinition(
                name="Acquisition",
                description="A acquires B",
                roles=[
                    EventRole(name="acquirer", required=True, entity_types=("Company",)),
                    EventRole(name="acquired", required=True, entity_types=("Company",)),
                ],
                properties={"deal_value": "amount"},
            )
        ]
    )


@pytest.fixture
def empty_registry():
    return EventTypeRegistry()


def _make_predictor_mock(events_payload):
    """Return a mocked predictor whose call returns an object with .events."""
    pred = Mock()
    result = Mock()
    result.events = events_payload
    pred.return_value = result
    return pred


class TestExtractEvents:
    def test_empty_text_returns_empty(self, acquisition_registry):
        assert extract_events("", [], acquisition_registry) == []
        assert extract_events("   ", [], acquisition_registry) == []

    def test_empty_registry_returns_empty(self, empty_registry):
        assert extract_events("Apple acquired Beats.", [], empty_registry) == []

    def test_no_lm_returns_empty(self, acquisition_registry):
        # The autouse `disable_lm_config` fixture in conftest already
        # ensures no LM is configured. The function should short-circuit
        # to an empty list rather than crashing.
        out = extract_events(
            "Apple acquired Beats in 2014.",
            [("Apple", "Company"), ("Beats", "Company")],
            acquisition_registry,
        )
        assert out == []

    def test_lm_required_raises_when_set(
        self, acquisition_registry, monkeypatch
    ):
        monkeypatch.setenv("DRG_REQUIRE_LM", "1")
        from drg.errors import ExtractionError

        with pytest.raises(ExtractionError):
            extract_events(
                "Apple acquired Beats.",
                [("Apple", "Company"), ("Beats", "Company")],
                acquisition_registry,
            )

    def test_extracts_event_from_typed_predictor_output(
        self, acquisition_registry
    ):
        from drg.events._extraction import RawEvent, RawEventList

        events_obj = RawEventList(
            events=[
                RawEvent(
                    event_type="Acquisition",
                    participants={"acquirer": "Apple", "acquired": "Beats"},
                    timestamp="May 2014",
                    location=None,
                    properties={"deal_value": "$3B"},
                    evidence_text="Apple acquired Beats in May 2014",
                    confidence=0.92,
                )
            ]
        )
        with patch("drg.events._extraction.dspy") as mock_dspy:
            mock_dspy.TypedPredictor = Mock(
                return_value=Mock(return_value=events_obj)
            )
            mock_dspy.Predict = Mock()
            mock_dspy.Signature = type("Signature", (), {})
            mock_dspy.InputField = Mock(return_value="in")
            mock_dspy.OutputField = Mock(return_value="out")
            mock_dspy.context = None
            mock_dspy.settings = Mock()
            mock_dspy.settings.lm = Mock()  # pretend LM is configured

            out = extract_events(
                "Apple acquired Beats in May 2014 for $3B.",
                [("Apple", "Company"), ("Beats", "Company")],
                acquisition_registry,
                document_id="doc1",
            )
        assert len(out) == 1
        ev = out[0]
        assert ev.event_type == "Acquisition"
        assert ev.participants["acquirer"] == ["Apple"]
        assert ev.participants["acquired"] == ["Beats"]
        assert ev.timestamp is not None
        assert ev.timestamp.start == "2014-05"
        assert ev.properties["deal_value"] == "$3B"
        assert ev.provenance.document_id == "doc1"
        assert 0.0 <= ev.confidence <= 1.0

    def test_drops_event_with_unknown_type(self, acquisition_registry):
        from drg.events._extraction import RawEvent, RawEventList

        events_obj = RawEventList(
            events=[
                RawEvent(
                    event_type="UnregisteredType",
                    participants={"acquirer": "Apple", "acquired": "Beats"},
                ),
                RawEvent(
                    event_type="Acquisition",
                    participants={"acquirer": "Apple", "acquired": "Beats"},
                ),
            ]
        )
        with patch("drg.events._extraction.dspy") as mock_dspy:
            mock_dspy.TypedPredictor = Mock(
                return_value=Mock(return_value=events_obj)
            )
            mock_dspy.Predict = Mock()
            mock_dspy.Signature = type("Signature", (), {})
            mock_dspy.InputField = Mock(return_value="in")
            mock_dspy.OutputField = Mock(return_value="out")
            mock_dspy.context = None
            mock_dspy.settings = Mock()
            mock_dspy.settings.lm = Mock()
            out = extract_events(
                "Apple acquired Beats.",
                [("Apple", "Company"), ("Beats", "Company")],
                acquisition_registry,
            )
        assert len(out) == 1
        assert out[0].event_type == "Acquisition"

    def test_drops_event_with_missing_required_role(self, acquisition_registry):
        from drg.events._extraction import RawEvent, RawEventList

        events_obj = RawEventList(
            events=[
                RawEvent(
                    event_type="Acquisition",
                    participants={"acquirer": "Apple"},  # missing 'acquired'
                ),
            ]
        )
        with patch("drg.events._extraction.dspy") as mock_dspy:
            mock_dspy.TypedPredictor = Mock(
                return_value=Mock(return_value=events_obj)
            )
            mock_dspy.Predict = Mock()
            mock_dspy.Signature = type("Signature", (), {})
            mock_dspy.InputField = Mock(return_value="in")
            mock_dspy.OutputField = Mock(return_value="out")
            mock_dspy.context = None
            mock_dspy.settings = Mock()
            mock_dspy.settings.lm = Mock()
            out = extract_events(
                "Apple acquired something.",
                [("Apple", "Company")],
                acquisition_registry,
            )
        assert out == []

    def test_dedups_identical_events(self, acquisition_registry):
        from drg.events._extraction import RawEvent, RawEventList

        events_obj = RawEventList(
            events=[
                RawEvent(
                    event_type="Acquisition",
                    participants={"acquirer": "Apple", "acquired": "Beats"},
                    timestamp="2014",
                ),
                RawEvent(
                    event_type="Acquisition",
                    participants={"acquirer": "Apple", "acquired": "Beats"},
                    timestamp="2014",
                ),
            ]
        )
        with patch("drg.events._extraction.dspy") as mock_dspy:
            mock_dspy.TypedPredictor = Mock(
                return_value=Mock(return_value=events_obj)
            )
            mock_dspy.Predict = Mock()
            mock_dspy.Signature = type("Signature", (), {})
            mock_dspy.InputField = Mock(return_value="in")
            mock_dspy.OutputField = Mock(return_value="out")
            mock_dspy.context = None
            mock_dspy.settings = Mock()
            mock_dspy.settings.lm = Mock()
            out = extract_events(
                "Apple acquired Beats in 2014. Apple acquired Beats in 2014.",
                [("Apple", "Company"), ("Beats", "Company")],
                acquisition_registry,
            )
        assert len(out) == 1
