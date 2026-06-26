"""Defensive JSON parser for DSPy outputs that arrive as strings.

Used in two places:

1. ``KGExtractor`` fallback path when the LLM returns a JSON string instead of
   parsed structured fields.
2. ``generate_schema_from_text`` — the schema-generation Signature always
   accepts a legacy JSON string in its ``generated_schema`` field, so the schema
   generator keeps this parser for saved examples and mocked tests.

The parser tolerates a few common LLM quirks (markdown code fences,
python-literal syntax) but raises ``ValueError`` on hard failures so callers
can decide how to handle them.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _parse_json_output(json_str: str, expected_format: str = "array") -> Any:
    """Parse a JSON string emitted by a legacy DSPy program.

    This is a defensive parser for legacy DSPy outputs that may wrap JSON in
    markdown code blocks or use Python-literal syntax instead of strict JSON.

    Args:
        json_str: JSON string to parse (may include markdown code blocks).
        expected_format: Expected JSON format ("array" or "object").

    Returns:
        Parsed JSON data (list or dict).

    Raises:
        ValueError: If JSON parsing fails or the format does not match.
    """
    if not isinstance(json_str, str):
        error_msg = f"Expected string, got {type(json_str).__name__}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Try strict JSON parsing first.
    json_str = json_str.strip()
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Strip markdown code fences (common LLM wrapper).
        cleaned = json_str
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            parsed = json.loads(cleaned)
            logger.debug(
                "JSON parsing succeeded after markdown code block removal (legacy behavior)"
            )
        except json.JSONDecodeError as e:
            # Last resort: parse Python-literal dict/list strings safely (no eval).
            try:
                parsed = ast.literal_eval(cleaned)
                logger.debug("Parsed output via ast.literal_eval fallback (python-literal JSON)")
            except Exception:
                error_msg = (
                    f"Failed to parse JSON output even after markdown cleaning: {e!s}. "
                    f"Input: {json_str[:200]}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e

    if expected_format == "array" and not isinstance(parsed, list):
        error_msg = f"Expected JSON array, got {type(parsed).__name__}. Input: {json_str[:200]}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    if expected_format == "object" and not isinstance(parsed, dict):
        error_msg = f"Expected JSON object, got {type(parsed).__name__}. Input: {json_str[:200]}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return parsed
