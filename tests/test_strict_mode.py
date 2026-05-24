"""Tests for DRG_STRICT mode behavior.

These tests verify that:
1. `is_strict()` reads the DRG_STRICT env var correctly.
2. Best-effort subsystems (entity resolution, coreference) re-raise in strict
   mode instead of being silently downgraded to warnings.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from drg.extract import extract_typed
from drg.schema import DRGSchema, Entity, Relation
from drg.utils.strict import is_strict


class TestIsStrict:
    """Sanity tests for the strict-mode env reader."""

    def test_default_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DRG_STRICT", raising=False)
        assert is_strict() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Y", "on"])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("DRG_STRICT", value)
        assert is_strict() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "off", "anything-else"])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("DRG_STRICT", value)
        assert is_strict() is False


class TestExtractTypedStrictBehavior:
    """`extract_typed` should re-raise resolver errors when DRG_STRICT=1."""

    @staticmethod
    def _build_extractor_mock() -> Mock:
        mock_extractor = Mock()
        result = Mock()
        result.entities = [("John", "Person")]
        result.relations = []
        result.enriched_relations = None
        mock_extractor.return_value = result
        return mock_extractor

    def _basic_schema(self) -> DRGSchema:
        return DRGSchema(
            entities=[Entity("Person")],
            relations=[Relation("knows", "Person", "Person")],
        )

    @patch("drg.extract._get_extractor")
    @patch("drg.extract.resolve_entities_and_relations")
    def test_resolver_failure_is_swallowed_by_default(
        self,
        mock_resolve: Mock,
        mock_get_extractor: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("DRG_STRICT", raising=False)
        mock_get_extractor.return_value = self._build_extractor_mock()
        mock_resolve.side_effect = RuntimeError("simulated resolver crash")

        entities, relations = extract_typed(
            "John knows nobody", self._basic_schema(), enable_entity_resolution=True
        )
        # Pipeline degrades gracefully and returns the pre-resolution entities/relations.
        assert entities == [("John", "Person")]
        assert relations == []

    @patch("drg.extract._get_extractor")
    @patch("drg.extract.resolve_entities_and_relations")
    def test_resolver_failure_raises_in_strict_mode(
        self,
        mock_resolve: Mock,
        mock_get_extractor: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DRG_STRICT", "1")
        mock_get_extractor.return_value = self._build_extractor_mock()
        mock_resolve.side_effect = RuntimeError("simulated resolver crash")

        with pytest.raises(RuntimeError, match="simulated resolver crash"):
            extract_typed(
                "John knows nobody",
                self._basic_schema(),
                enable_entity_resolution=True,
            )
