"""Unit tests for drg.events._registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drg.events import (
    EventRole,
    EventTypeDefinition,
    EventTypeRegistry,
    default_event_registry,
    example_event_registry,
)


class TestEventTypeRegistry:
    def test_default_is_empty(self):
        reg = default_event_registry()
        assert len(reg) == 0
        assert reg.names() == []

    def test_register_and_lookup(self):
        reg = EventTypeRegistry()
        td = EventTypeDefinition(name="X", description="d")
        reg.register(td)
        assert "X" in reg
        assert reg.get("X") is td
        assert reg.names() == ["X"]
        assert len(reg) == 1

    def test_iteration_is_ordered(self):
        reg = EventTypeRegistry()
        for name in ["First", "Second", "Third"]:
            reg.register(EventTypeDefinition(name=name, description="d"))
        assert [t.name for t in reg] == ["First", "Second", "Third"]

    def test_register_duplicate_raises(self):
        reg = EventTypeRegistry()
        reg.register(EventTypeDefinition(name="X", description="d"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(EventTypeDefinition(name="X", description="d2"))

    def test_register_overwrite_allowed(self):
        reg = EventTypeRegistry()
        reg.register(EventTypeDefinition(name="X", description="d"))
        new_td = EventTypeDefinition(name="X", description="new")
        reg.register(new_td, overwrite=True)
        assert reg.get("X").description == "new"

    def test_remove(self):
        reg = EventTypeRegistry()
        reg.register(EventTypeDefinition(name="X", description="d"))
        reg.remove("X")
        assert "X" not in reg
        with pytest.raises(ValueError, match="not registered"):
            reg.remove("X")

    def test_to_from_dict_roundtrip(self):
        reg = EventTypeRegistry(
            types=[
                EventTypeDefinition(
                    name="A",
                    description="d",
                    roles=[EventRole(name="r1", entity_types=("X",))],
                ),
                EventTypeDefinition(name="B", description="d2"),
            ]
        )
        out = EventTypeRegistry.from_dict(reg.to_dict())
        assert out.names() == ["A", "B"]
        assert out.get("A").roles[0].name == "r1"

    def test_save_and_load_json(self, tmp_path: Path):
        reg = EventTypeRegistry(
            types=[EventTypeDefinition(name="X", description="d")]
        )
        path = tmp_path / "registry.json"
        reg.save_json(path)
        assert path.exists()
        # Verify JSON is valid + structured
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == "1.0"
        # Roundtrip
        loaded = EventTypeRegistry.from_json(path)
        assert loaded.names() == ["X"]


class TestExampleEventRegistry:
    def test_contains_expected_types(self):
        reg = example_event_registry()
        expected = {
            "Acquisition",
            "Merger",
            "Funding",
            "ProductLaunch",
            "Partnership",
            "LeadershipChange",
            "Lawsuit",
            "Election",
            "Employment",
            "Investment",
        }
        assert set(reg.names()) == expected

    def test_acquisition_has_required_roles(self):
        reg = example_event_registry()
        td = reg.get("Acquisition")
        assert td is not None
        role_names = {r.name for r in td.required_roles()}
        assert "acquirer" in role_names
        assert "acquired" in role_names

    def test_employment_roles_typed(self):
        reg = example_event_registry()
        td = reg.get("Employment")
        assert td is not None
        emp_role = td.get_role("employee")
        assert emp_role is not None
        assert "Person" in emp_role.entity_types

    def test_serialization_roundtrip(self):
        reg = example_event_registry()
        out = EventTypeRegistry.from_dict(reg.to_dict())
        assert out.names() == reg.names()
