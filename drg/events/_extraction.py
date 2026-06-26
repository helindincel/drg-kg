"""LLM-driven event extraction using DSPy.

Mirrors the architecture of :mod:`drg.extract` (entity / relation
extraction): a single dynamic ``dspy.Signature`` is built from the
event-type registry, the LLM is asked for structured output, and a
deterministic post-processor (:mod:`drg.events._postprocess`) turns
raw output into validated :class:`Event` instances.

The extractor is deliberately separate from ``KGExtractor`` so callers
who only want entities + relations pay no extra cost. Wiring is
opt-in (``extract_events`` is a dedicated entry point).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any
from unittest.mock import Mock as _Mock

import dspy
from pydantic import BaseModel
from pydantic.config import ConfigDict

from ..errors import ExtractionError
from ..extract._parsing import _parse_json_output
from ..utils.llm_throttle import throttle_llm_calls
from ..utils.strict import is_strict
from ._postprocess import build_event
from ._registry import EventTypeRegistry
from ._types import Event

logger = logging.getLogger(__name__)


class RawEvent(BaseModel):
    """Raw, LLM-shaped event payload before post-processing."""

    model_config = ConfigDict(extra="ignore")

    event_type: str
    participants: dict[str, Any] = {}
    timestamp: str | None = None
    location: str | None = None
    properties: dict[str, Any] = {}
    evidence_text: str | None = None
    confidence: float | None = None


class RawEventList(BaseModel):
    """Structured wrapper around the LLM event-extraction output."""

    model_config = ConfigDict(extra="ignore")
    events: list[RawEvent] = []


def _event_types_for_registry(registry: EventTypeRegistry) -> list[dict[str, Any]]:
    """Return event registry definitions as structured DSPy input data."""
    return [event_type.to_dict() for event_type in registry]


def _build_signature(registry: EventTypeRegistry) -> type:
    """Build a dynamic ``dspy.Signature`` from the registry.

    Event catalogue data is passed as a structured InputField at call time, so
    the signature does not hide registry data inside OutputField prose.
    """
    output_desc = (
        "Extracted events. Each item has: "
        "event_type (one of the registered types), "
        "participants (object mapping role name -> entity name string OR list of entity names), "
        "timestamp (free-form date string from the text or null), "
        "location (entity name string or null), "
        "properties (object of additional fields), "
        "evidence_text (short snippet from the text supporting the event, or null), "
        "confidence (float in [0,1] reflecting how confident you are, or null). "
        "Use ONLY the entity names provided in the entities input. Use ONLY registered event_types. "
        "Populate properties only from keys declared by the matching event type."
    )

    class EventExtraction(dspy.Signature):
        """Extract typed events from text given a list of entities and an event-type catalogue."""

        text: str = dspy.InputField(desc="Input text from which to extract events")
        entities: list[tuple[str, str]] = dspy.InputField(
            desc="Known entities as [(name, type), ...]"
        )
        event_types: list[dict[str, Any]] = dspy.InputField(
            desc=(
                "Registered event type definitions with name, description, roles, "
                "properties, and examples."
            )
        )
        events: list[RawEvent] = dspy.OutputField(desc=output_desc)

    EventExtraction._registry_size = len(registry)  # type: ignore[attr-defined]
    return EventExtraction


def _maybe_lm_context(lm: Any | None):
    """Best-effort dspy-context manager (mirrors drg.extract._maybe_lm_context)."""
    if lm is None:
        return contextlib.nullcontext()
    ctx_factory = getattr(dspy, "context", None)
    if ctx_factory is not None and not isinstance(ctx_factory, _Mock):
        try:
            return ctx_factory(lm=lm)
        except TypeError:
            pass
    settings = getattr(dspy, "settings", None)
    if settings is not None:
        sub_ctx = getattr(settings, "context", None)
        if sub_ctx is not None and not isinstance(sub_ctx, _Mock):
            try:
                return sub_ctx(lm=lm)
            except TypeError:
                pass
    return contextlib.nullcontext()


def _maybe_json_adapter_context():
    """Use DSPy 3's JSONAdapter for event structured outputs when available."""
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


def _coerce_raw_events(raw: Any) -> list[RawEvent]:
    """Coerce structured ``Predict`` output into a list of :class:`RawEvent`."""
    if isinstance(raw, RawEventList):
        return list(raw.events)
    if isinstance(raw, list):
        return [r if isinstance(r, RawEvent) else RawEvent(**r) for r in raw if r]
    if isinstance(raw, dict):
        events_field = raw.get("events", raw)
        if isinstance(events_field, list):
            return [RawEvent(**r) for r in events_field if isinstance(r, dict)]
        return []
    if isinstance(raw, str):
        try:
            parsed = _parse_json_output(raw, expected_format="object")
        except ValueError:
            return []
        if isinstance(parsed, dict) and "events" in parsed:
            parsed = parsed["events"]
        if isinstance(parsed, list):
            return [RawEvent(**r) for r in parsed if isinstance(r, dict)]
        return []
    return []


def extract_events(
    text: str,
    entities_typed: list[tuple[str, str]],
    registry: EventTypeRegistry,
    *,
    document_id: str | None = None,
    chunk_id: str | None = None,
    lm: Any | None = None,
    extraction_method: str = "llm",
) -> list[Event]:
    """Extract :class:`Event` instances from text.

    The function is opt-in: callers must explicitly invoke it. The
    existing ``extract_typed`` / ``extract_from_chunks`` paths are
    untouched.

    Mock-mode short-circuit: when no LM is configured (neither
    injected nor present in ``dspy.settings``), an empty list is
    returned instead of crashing — same convention as
    :func:`drg.extract.extract_typed`.
    """
    if not text or not text.strip():
        return []
    if len(registry) == 0:
        logger.debug("Empty event registry; nothing to extract")
        return []

    effective_lm = lm if lm is not None else getattr(getattr(dspy, "settings", None), "lm", None)
    if effective_lm is None:
        if os.getenv("DRG_REQUIRE_LM", "").lower() in {"1", "true", "yes"}:
            raise ExtractionError(
                "No DSPy LM configured and DRG_REQUIRE_LM is set; cannot extract events."
            )
        logger.warning("No DSPy LM configured; returning empty events (mock mode).")
        return []

    Signature = _build_signature(registry)
    event_types = _event_types_for_registry(registry)

    try:
        predictor = dspy.Predict(Signature)
    except Exception as exc:
        if is_strict():
            raise
        logger.warning("Predictor creation failed for events: %s", exc, exc_info=True)
        return []

    with contextlib.ExitStack() as stack:
        stack.enter_context(_maybe_lm_context(lm))
        stack.enter_context(_maybe_json_adapter_context())
        try:
            throttle_llm_calls()
            result = predictor(text=text, entities=entities_typed, event_types=event_types)
            raw_events = _coerce_raw_events(getattr(result, "events", []))
        except Exception as exc:
            if is_strict():
                raise
            logger.warning("Event extraction failed: %s", exc, exc_info=True)
            return []

    entity_type_map: dict[str, str] = {}
    for name, etype in entities_typed:
        if name and isinstance(name, str):
            entity_type_map[name] = etype

    events: list[Event] = []
    seen_ids: set[str] = set()
    for raw in raw_events:
        type_def = registry.get(raw.event_type)
        if type_def is None:
            logger.debug("Skipping event with unknown type %r", raw.event_type)
            continue
        event = build_event(
            event_type=raw.event_type,
            raw_participants=raw.participants,
            raw_timestamp=raw.timestamp,
            location=raw.location,
            properties=raw.properties,
            evidence_text=raw.evidence_text,
            type_def=type_def,
            entity_type_map=entity_type_map,
            source_text=text,
            document_id=document_id,
            chunk_id=chunk_id,
            llm_confidence=raw.confidence,
            extraction_method=extraction_method,
        )
        if event is None:
            continue
        if event.id in seen_ids:
            continue
        seen_ids.add(event.id)
        events.append(event)

    logger.info("Event extraction complete: %d event(s) extracted", len(events))
    return events
