"""DSPy adapter selection and Gemini-friendly output salvage."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock as _Mock

import dspy

logger = logging.getLogger(__name__)


def _active_lm_model_name() -> str:
    settings = getattr(dspy, "settings", None)
    lm = getattr(settings, "lm", None) if settings is not None else None
    if lm is None:
        return os.getenv("DRG_MODEL", "")
    for attr in ("model", "model_name"):
        value = getattr(lm, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return os.getenv("DRG_MODEL", "")


def _use_json_adapter() -> bool:
    """Whether to scope DSPy calls with JSONAdapter.

    Gemini models (direct or via OpenRouter) often emit singleton JSON objects
    instead of typed field wrappers, which JSONAdapter cannot parse reliably.
    """
    override = os.getenv("DRG_USE_JSON_ADAPTER", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if os.getenv("DRG_DISABLE_JSON_ADAPTER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    model = _active_lm_model_name().lower()
    if "gemini" in model:
        return False
    return True


def _maybe_json_adapter_context():
    """Use DSPy JSONAdapter when the active model supports it."""
    if not _use_json_adapter():
        return contextlib.nullcontext()

    json_adapter_cls = getattr(dspy, "JSONAdapter", None)
    if json_adapter_cls is None or isinstance(json_adapter_cls, _Mock):
        return contextlib.nullcontext()

    settings = getattr(dspy, "settings", None)
    if settings is not None and getattr(settings, "adapter", None) is not None:
        return contextlib.nullcontext()

    try:
        adapter = json_adapter_cls()
    except Exception:
        return contextlib.nullcontext()

    ctx_factory = getattr(dspy, "context", None)
    if ctx_factory is not None and not isinstance(ctx_factory, _Mock):
        try:
            return ctx_factory(adapter=adapter)
        except TypeError:
            pass

    if settings is not None:
        sub_ctx = getattr(settings, "context", None)
        if sub_ctx is not None and not isinstance(sub_ctx, _Mock):
            try:
                return sub_ctx(adapter=adapter)
            except TypeError:
                pass

    return contextlib.nullcontext()


def _looks_like_entity(d: dict[str, Any]) -> bool:
    name = d.get("name") or d.get("entity") or d.get("entity_name")
    etype = d.get("type") or d.get("entity_type")
    return bool(name and etype)


def _looks_like_relation(d: dict[str, Any]) -> bool:
    source = d.get("source") or d.get("src")
    relation = d.get("relation") or d.get("predicate") or d.get("type")
    target = d.get("target") or d.get("dst") or d.get("object")
    return bool(source and relation and target)


def _looks_like_entity_type(d: dict[str, Any]) -> bool:
    return bool(d.get("name") and d.get("description"))


def _looks_like_relation_group(d: dict[str, Any]) -> bool:
    return bool(d.get("name") and isinstance(d.get("relations"), list))


def _salvage_wrapped_field(error: Exception, field_name: str) -> Any | None:
    """Recover a typed output field from a JSONAdapter parse error message."""
    msg = str(error)
    if "LM Response:" not in msg:
        return None

    payload = msg.split("LM Response:", 1)[1]
    if "Expected to find output fields" in payload:
        payload = payload.split("Expected to find output fields", 1)[0]
    payload = payload.strip()

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, list):
        return parsed

    if not isinstance(parsed, dict):
        return None

    if field_name in parsed:
        value = parsed[field_name]
        return value if isinstance(value, list) else [value]

    singleton_checks = {
        "entities": _looks_like_entity,
        "relations": _looks_like_relation,
        "resolved_relations": _looks_like_relation,
        "entity_types": _looks_like_entity_type,
        "relation_groups": _looks_like_relation_group,
    }
    checker = singleton_checks.get(field_name)
    if checker and checker(parsed):
        return [parsed]

    return None


def run_predict(predictor: Any, *, salvage_fields: tuple[str, ...] = (), **kwargs: Any) -> Any:
    """Run a DSPy predictor and salvage Gemini-style singleton JSON on parse failure."""
    try:
        return predictor(**kwargs)
    except Exception as exc:
        for field in salvage_fields:
            salvaged = _salvage_wrapped_field(exc, field)
            if salvaged is not None:
                logger.warning(
                    "Salvaged %s output from adapter parse error (%d item(s))",
                    field,
                    len(salvaged) if isinstance(salvaged, list) else 1,
                )
                return SimpleNamespace(**{field: salvaged})
        raise
