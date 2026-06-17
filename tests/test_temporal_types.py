"""Unit tests for drg.temporal types and comparison."""

from __future__ import annotations

import pytest

from drg.graph.kg_core import KGEdge, KGNode
from drg.temporal import (
    PartialDate,
    TemporalScope,
    compare_partial_dates,
    intervals_overlap,
    is_active_at,
    migrate_edge_dict,
    migrate_node_dict,
    temporal_from_edge_fields,
)


def test_partial_date_year_precision():
    pd = PartialDate.parse("2014")
    assert pd is not None
    assert pd.precision == "year"
    assert pd.interval_start() == "2014-01-01"
    assert pd.interval_end() == "2014-12-31"


def test_partial_date_month_precision():
    pd = PartialDate.parse("2014-06")
    assert pd is not None
    assert pd.precision == "month"
    assert pd.interval_start() == "2014-06-01"
    assert pd.interval_end() == "2014-06-30"


def test_temporal_scope_active_at_year():
    scope = TemporalScope(valid_from="2011", valid_to=None)
    assert is_active_at(scope, "2011")
    assert is_active_at(scope, "2011-06-15")
    assert not is_active_at(scope, "2010")


def test_temporal_scope_active_at_bounded():
    jobs = TemporalScope(valid_from="1997", valid_to="2011")
    cook = TemporalScope(valid_from="2011", valid_to=None)
    assert is_active_at(jobs, "2008")
    assert not is_active_at(jobs, "2012")
    assert is_active_at(cook, "2012")
    assert not is_active_at(cook, "2010")


def test_intervals_overlap_open_ended():
    a = TemporalScope(valid_from="1997", valid_to="2010")
    b = TemporalScope(valid_from="2011", valid_to=None)
    assert not intervals_overlap(a, b)


def test_compare_partial_dates_ordering():
    assert compare_partial_dates("2010", "2011") < 0
    assert compare_partial_dates("2014-06", "2014-06-15") == 0


def test_kgedge_valid_from_aliases():
    edge = KGEdge(
        source="Tim Cook",
        target="Apple",
        relationship_type="CEO_OF",
        relationship_detail="Tim Cook CEO of Apple",
        start_time="2011",
    )
    assert edge.valid_from == "2011"
    edge.valid_to = "2020"
    assert edge.end_time == "2020"


def test_kgedge_temporal_scope_roundtrip():
    scope = TemporalScope(
        valid_from="2011",
        valid_to=None,
        created_at="2026-06-07T12:00:00Z",
        precision_from="year",
    )
    edge = KGEdge(
        source="Tim Cook",
        target="Apple",
        relationship_type="CEO_OF",
        relationship_detail="detail",
    )
    edge.apply_temporal_scope(scope)
    edge.created_at = "2026-06-07T12:00:00Z"
    restored = edge.get_temporal_scope()
    assert restored is not None
    assert restored.valid_from == "2011"
    assert restored.precision_from == "year"


def test_kgnode_temporal_metadata():
    node = KGNode(id="Apple", type="Company")
    node.apply_temporal_scope(TemporalScope(valid_from="1976", valid_to=None))
    d = node.to_dict()
    assert "temporal" in d
    loaded = KGNode.from_dict(d)
    scope = loaded.get_temporal_scope()
    assert scope is not None
    assert scope.valid_from == "1976"


def test_temporal_from_edge_fields_legacy_start_end():
    scope = temporal_from_edge_fields(start_time="2020", end_time="2021")
    assert scope is not None
    assert scope.valid_from == "2020"
    assert scope.valid_to == "2021"


def test_migrate_edge_dict_idempotent():
    raw = {
        "source": "A",
        "target": "B",
        "relationship_type": "owns",
        "relationship_detail": "A owns B",
        "start_time": "2014",
        "end_time": None,
    }
    migrated = migrate_edge_dict(raw)
    assert migrated["valid_from"] == "2014"
    assert "temporal" in migrated["metadata"]


def test_migrate_node_dict_from_properties():
    raw = {
        "id": "X",
        "properties": {"valid_from": "2010", "valid_to": "2015"},
    }
    migrated = migrate_node_dict(raw)
    assert migrated["metadata"]["temporal"]["valid_from"] == "2010"


def test_partial_date_rejects_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        PartialDate(value="", precision="year")
