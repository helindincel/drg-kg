"""Unit tests for drg.extract._parsing — no LLM / DSPy required."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

# drg/extract/__init__.py imports dspy at module level.
# Stub it so these pure-Python tests don't need dspy installed.
sys.modules.setdefault("dspy", MagicMock())

from drg.extract._parsing import _parse_json_output  # noqa: E402


# ---------------------------------------------------------------------------
# Happy-path: strict JSON
# ---------------------------------------------------------------------------


class TestStrictJSON:
    def test_parses_plain_array(self):
        raw = json.dumps([["Apple", "Company"], ["iPhone", "Product"]])
        result = _parse_json_output(raw, expected_format="array")
        assert result == [["Apple", "Company"], ["iPhone", "Product"]]

    def test_parses_plain_object(self):
        raw = json.dumps({"key": "value", "num": 42})
        result = _parse_json_output(raw, expected_format="object")
        assert result == {"key": "value", "num": 42}

    def test_parses_empty_array(self):
        assert _parse_json_output("[]", expected_format="array") == []

    def test_parses_empty_object(self):
        assert _parse_json_output("{}", expected_format="object") == {}

    def test_parses_nested_structure(self):
        raw = json.dumps([{"a": [1, 2, 3]}, {"b": {"c": True}}])
        result = _parse_json_output(raw, expected_format="array")
        assert result[0] == {"a": [1, 2, 3]}

    def test_handles_leading_trailing_whitespace(self):
        raw = "  " + json.dumps([1, 2, 3]) + "  "
        assert _parse_json_output(raw, expected_format="array") == [1, 2, 3]


# ---------------------------------------------------------------------------
# Markdown code-fence stripping
# ---------------------------------------------------------------------------


class TestMarkdownFenceStripping:
    def test_strips_json_fence(self):
        raw = "```json\n[1, 2, 3]\n```"
        assert _parse_json_output(raw, expected_format="array") == [1, 2, 3]

    def test_strips_plain_fence(self):
        raw = "```\n[1, 2, 3]\n```"
        assert _parse_json_output(raw, expected_format="array") == [1, 2, 3]

    def test_strips_fence_with_whitespace(self):
        raw = "```json\n  {\"x\": 1}\n```"
        assert _parse_json_output(raw, expected_format="object") == {"x": 1}

    def test_strips_fence_no_trailing_newline(self):
        raw = "```json\n[\"a\",\"b\"]```"
        assert _parse_json_output(raw, expected_format="array") == ["a", "b"]


# ---------------------------------------------------------------------------
# Python-literal fallback (ast.literal_eval)
# ---------------------------------------------------------------------------


class TestPythonLiteralFallback:
    def test_parses_python_list_syntax(self):
        # Python uses True/False, JSON uses true/false — ast.literal_eval handles this
        raw = "['Apple', 'Google']"
        result = _parse_json_output(raw, expected_format="array")
        assert result == ["Apple", "Google"]

    def test_parses_python_dict_syntax(self):
        raw = "{'key': 'value'}"
        result = _parse_json_output(raw, expected_format="object")
        assert result == {"key": "value"}

    def test_parses_python_list_with_tuples_inside_fence(self):
        # ast.literal_eval can handle nested Python literals
        raw = "```\n['a', 'b', 'c']\n```"
        assert _parse_json_output(raw, expected_format="array") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Format validation errors
# ---------------------------------------------------------------------------


class TestFormatValidation:
    def test_array_expected_but_got_object(self):
        raw = json.dumps({"key": "value"})
        with pytest.raises(ValueError, match="Expected JSON array"):
            _parse_json_output(raw, expected_format="array")

    def test_object_expected_but_got_array(self):
        raw = json.dumps([1, 2, 3])
        with pytest.raises(ValueError, match="Expected JSON object"):
            _parse_json_output(raw, expected_format="object")

    def test_array_expected_but_got_string(self):
        # A bare JSON string is valid JSON but not an array
        raw = json.dumps("hello")
        with pytest.raises(ValueError, match="Expected JSON array"):
            _parse_json_output(raw, expected_format="array")

    def test_object_expected_but_got_number(self):
        raw = "42"
        with pytest.raises(ValueError, match="Expected JSON object"):
            _parse_json_output(raw, expected_format="object")


# ---------------------------------------------------------------------------
# Hard failure cases
# ---------------------------------------------------------------------------


class TestHardFailures:
    def test_raises_on_non_string_input(self):
        with pytest.raises(ValueError, match="Expected string"):
            _parse_json_output(123, expected_format="array")  # type: ignore[arg-type]

    def test_raises_on_none_input(self):
        with pytest.raises(ValueError, match="Expected string"):
            _parse_json_output(None, expected_format="array")  # type: ignore[arg-type]

    def test_raises_on_completely_invalid_json(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            _parse_json_output("not json at all !!!", expected_format="array")

    def test_raises_on_invalid_json_after_fence_strip(self):
        raw = "```json\nnot valid json\n```"
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            _parse_json_output(raw, expected_format="array")

    def test_error_message_includes_input_preview(self):
        raw = "definitely not json"
        with pytest.raises(ValueError) as exc_info:
            _parse_json_output(raw, expected_format="array")
        assert "definitely not json" in str(exc_info.value)

    def test_raises_on_list_input(self):
        with pytest.raises(ValueError, match="Expected string"):
            _parse_json_output([], expected_format="array")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Default expected_format behaviour
# ---------------------------------------------------------------------------


class TestDefaultFormat:
    def test_default_format_is_array(self):
        raw = json.dumps([1, 2])
        # Should not raise — default is "array"
        assert _parse_json_output(raw) == [1, 2]

    def test_default_format_rejects_object(self):
        raw = json.dumps({"x": 1})
        with pytest.raises(ValueError, match="Expected JSON array"):
            _parse_json_output(raw)
