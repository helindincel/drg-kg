"""Tests for Gemini-friendly DSPy adapter helpers."""

from __future__ import annotations

from drg.extract._adapters import (
    _salvage_wrapped_field,
    _use_json_adapter,
    run_predict,
)


def test_use_json_adapter_disabled_for_gemini(monkeypatch):
    monkeypatch.setenv("DRG_MODEL", "openrouter/google/gemini-2.5-flash")
    assert _use_json_adapter() is False


def test_use_json_adapter_honors_force_flag(monkeypatch):
    monkeypatch.setenv("DRG_MODEL", "openrouter/google/gemini-2.5-flash")
    monkeypatch.setenv("DRG_USE_JSON_ADAPTER", "1")
    assert _use_json_adapter() is True


def test_salvage_singleton_relation():
    error = Exception(
        "Adapter JSONAdapter failed to parse the LM response.\n\n"
        'LM Response: {"source": "A", "relation": "knows", "target": "B", '
        '"confidence": 1.0, "evidence": "A knows B", "is_negated": false, '
        '"temporal": null}\n\n'
        "Expected to find output fields in the LM response: [relations]"
    )
    salvaged = _salvage_wrapped_field(error, "relations")
    assert isinstance(salvaged, list)
    assert salvaged[0]["source"] == "A"


def test_salvage_wrapped_entities_list():
    error = Exception(
        'LM Response: {"entities": [{"name": "Marie", "type": "Person"}]} '
        "Expected to find output fields in the LM response: [entities]"
    )
    salvaged = _salvage_wrapped_field(error, "entities")
    assert salvaged == [{"name": "Marie", "type": "Person"}]


def test_run_predict_salvages_on_failure():
    class _BrokenPredictor:
        def __call__(self, **_kwargs):
            raise RuntimeError(
                'LM Response: {"source": "A", "relation": "knows", "target": "B"} '
                "Expected to find output fields in the LM response: [relations]"
            )

    result = run_predict(_BrokenPredictor(), salvage_fields=("relations",))
    assert result.relations[0]["target"] == "B"
