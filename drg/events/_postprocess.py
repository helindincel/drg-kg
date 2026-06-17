"""Deterministic post-processing for raw extracted events.

These helpers turn the raw, LLM-shaped output (a dict with possibly-noisy
participant names and a free-form date string) into a fully-validated
:class:`Event` with a deterministic id, a normalized timestamp, and an
evidence span derived from the source text.

Everything here is pure (no LLM calls, no IO) and side-effect-free, so it
is cheap to unit-test and safe to call from any pipeline stage.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

from .._version import __version__ as _drg_version  # type: ignore[import-not-found]
from ._types import (
    Event,
    EventProvenance,
    EventTimestamp,
    EventTypeDefinition,
    TextSpan,
    TimestampPrecision,
)

logger = logging.getLogger(__name__)


_MONTHS_EN = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def parse_timestamp(raw: str | None) -> EventTimestamp | None:
    """Parse a free-form date string into an :class:`EventTimestamp`.

    Conservative: returns ``None`` when nothing recognizable is found
    rather than guessing. Recognized shapes (in order of preference):

    - ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM:SS`` (ISO 8601)
    - ``Month DD, YYYY`` / ``DD Month YYYY``
    - ``Month YYYY``
    - ``YYYY``
    - ``YYYY-YYYY`` ranges
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()

    iso_full = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?)?", text)
    if iso_full:
        y, m, d = iso_full.group(1), iso_full.group(2), iso_full.group(3)
        return EventTimestamp(
            start=f"{y}-{m}-{d}",
            end=f"{y}-{m}-{d}",
            precision="day",
            raw_text=text,
        )

    iso_ym = re.match(r"^(\d{4})-(\d{2})$", text)
    if iso_ym:
        y, m = iso_ym.group(1), iso_ym.group(2)
        return EventTimestamp(start=f"{y}-{m}", end=f"{y}-{m}", precision="month", raw_text=text)

    range_match = re.match(r"^(\d{4})\s*[-\u2013]\s*(\d{4})$", text)
    if range_match:
        y1, y2 = sorted((int(range_match.group(1)), int(range_match.group(2))))
        return EventTimestamp(start=str(y1), end=str(y2), precision="year", raw_text=text)

    month_alt = "|".join(_MONTHS_EN.keys())
    md_y = re.search(
        rf"\b({month_alt})\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
        text,
        flags=re.IGNORECASE,
    )
    if md_y:
        m = _MONTHS_EN[md_y.group(1).lower()]
        d = int(md_y.group(2))
        y = int(md_y.group(3))
        if 1 <= d <= 31:
            iso = f"{y:04d}-{m:02d}-{d:02d}"
            return EventTimestamp(start=iso, end=iso, precision="day", raw_text=text)

    d_m_y = re.search(
        rf"\b(\d{{1,2}})\s+({month_alt})\s+(\d{{4}})\b",
        text,
        flags=re.IGNORECASE,
    )
    if d_m_y:
        d = int(d_m_y.group(1))
        m = _MONTHS_EN[d_m_y.group(2).lower()]
        y = int(d_m_y.group(3))
        if 1 <= d <= 31:
            iso = f"{y:04d}-{m:02d}-{d:02d}"
            return EventTimestamp(start=iso, end=iso, precision="day", raw_text=text)

    m_y = re.search(rf"\b({month_alt})\s+(\d{{4}})\b", text, flags=re.IGNORECASE)
    if m_y:
        m = _MONTHS_EN[m_y.group(1).lower()]
        y = int(m_y.group(2))
        iso = f"{y:04d}-{m:02d}"
        return EventTimestamp(start=iso, end=iso, precision="month", raw_text=text)

    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if year_match:
        y = year_match.group(1)
        return EventTimestamp(start=y, end=y, precision="year", raw_text=text)

    return None


def normalize_participants(
    raw_participants: dict[str, Any] | None,
    *,
    type_def: EventTypeDefinition | None,
    entity_type_map: dict[str, str],
) -> dict[str, list[str]]:
    """Normalize raw participant data against a type definition.

    Accepts dict-of-string-or-list shapes (LLMs often emit either). Drops
    role names not declared in the type definition (when one is given).
    Resolves participant names case-insensitively against ``entity_type_map``
    so the LLM can return surface forms while we record canonical names.
    """
    if not isinstance(raw_participants, dict):
        return {}

    canonical_index = {name.lower(): name for name in entity_type_map.keys()}
    allowed_roles: set[str] | None = None
    if type_def is not None:
        allowed_roles = {r.name for r in type_def.roles}

    out: dict[str, list[str]] = {}
    for role_name, value in raw_participants.items():
        if not isinstance(role_name, str) or not role_name.strip():
            continue
        if allowed_roles is not None and role_name not in allowed_roles:
            continue
        if isinstance(value, str):
            entities = [value]
        elif isinstance(value, list):
            entities = [str(v) for v in value if v is not None]
        else:
            continue

        resolved: list[str] = []
        seen_lower: set[str] = set()
        for raw_name in entities:
            if not raw_name or not raw_name.strip():
                continue
            canon = canonical_index.get(raw_name.strip().lower(), raw_name.strip())
            if canon.lower() in seen_lower:
                continue
            seen_lower.add(canon.lower())
            resolved.append(canon)

        if resolved:
            out[role_name] = resolved
    return out


def required_role_coverage(
    type_def: EventTypeDefinition | None,
    participants: dict[str, list[str]],
) -> float:
    """Fraction of required roles that have at least one participant.

    Returns 1.0 when there are no required roles or when no type
    definition is available — i.e. we don't penalize free-form events.
    """
    if type_def is None:
        return 1.0
    required = type_def.required_roles()
    if not required:
        return 1.0
    covered = sum(1 for r in required if participants.get(r.name))
    return covered / len(required)


def has_all_required_roles(
    type_def: EventTypeDefinition | None,
    participants: dict[str, list[str]],
) -> bool:
    if type_def is None:
        return True
    return all(participants.get(r.name) for r in type_def.required_roles())


def find_evidence_span(
    text: str,
    entity_names: list[str],
    *,
    chunk_id: str | None = None,
    max_chars: int = 280,
) -> TextSpan | None:
    """Return a short text span containing as many participants as possible.

    Greedy: picks the smallest window that contains at least two
    participant mentions (or one, if only one is provided). Returns
    ``None`` when no participant is found in the text.
    """
    if not text or not entity_names:
        return None

    positions: list[tuple[int, str]] = []
    for name in entity_names:
        if not name:
            continue
        for match in re.finditer(
            rf"(?<!\w){re.escape(name)}(?!\w)", text, flags=re.IGNORECASE
        ):
            positions.append((match.start(), name))
    if not positions:
        return None
    positions.sort(key=lambda p: p[0])

    if len(positions) == 1 or len({n for _, n in positions}) == 1:
        center = positions[0][0]
        half = max_chars // 2
        start = max(0, center - half)
        end = min(len(text), start + max_chars)
        snippet = text[start:end].strip()
        return TextSpan(text=snippet, chunk_id=chunk_id, start=start, end=end) if snippet else None

    best_start = 0
    best_end = len(text)
    best_width = best_end - best_start
    distinct_target = min(len({n for _, n in positions}), 4)
    n = len(positions)
    left = 0
    seen_counter: dict[str, int] = {}
    for right in range(n):
        seen_counter[positions[right][1]] = seen_counter.get(positions[right][1], 0) + 1
        while len(seen_counter) >= distinct_target and left <= right:
            window = positions[right][0] - positions[left][0]
            if window < best_width:
                best_width = window
                best_start = positions[left][0]
                best_end = positions[right][0] + len(positions[right][1])
            name = positions[left][1]
            seen_counter[name] -= 1
            if seen_counter[name] == 0:
                del seen_counter[name]
            left += 1

    pad = max(0, (max_chars - (best_end - best_start)) // 2)
    start = max(0, best_start - pad)
    end = min(len(text), best_end + pad)
    if end - start > max_chars:
        end = start + max_chars
    snippet = text[start:end].strip()
    if not snippet:
        return None
    return TextSpan(text=snippet, chunk_id=chunk_id, start=start, end=end)


def make_event_id(
    event_type: str,
    participants: dict[str, list[str]],
    timestamp: EventTimestamp | None,
    evidence_text: str | None,
) -> str:
    """Deterministic id for an event.

    Two extractions of the same event from the same document text
    produce the same id, which is what enables idempotent merging in
    :class:`drg.graph.GraphMerger`.
    """
    role_part = ";".join(
        f"{role}={','.join(sorted(p.lower() for p in entities))}"
        for role, entities in sorted(participants.items())
    )
    ts_part = ""
    if timestamp is not None and timestamp.start:
        ts_part = timestamp.start
    elif timestamp is not None and timestamp.raw_text:
        ts_part = timestamp.raw_text

    evidence_part = ""
    if evidence_text:
        evidence_part = hashlib.sha1(
            evidence_text.lower().encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:8]

    payload = f"{event_type}|{role_part}|{ts_part}|{evidence_part}"
    digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    safe_type = re.sub(r"[^A-Za-z0-9]+", "", event_type) or "Event"
    return f"event:{safe_type}:{digest}"


def build_event(
    *,
    event_type: str,
    raw_participants: dict[str, Any] | None,
    raw_timestamp: str | None,
    location: str | None,
    properties: dict[str, Any] | None,
    evidence_text: str | None,
    type_def: EventTypeDefinition | None,
    entity_type_map: dict[str, str],
    source_text: str | None = None,
    document_id: str | None = None,
    chunk_id: str | None = None,
    llm_confidence: float | None = None,
    extraction_method: str = "llm",
) -> Event | None:
    """Assemble a fully-validated :class:`Event` from raw extractor output.

    Returns ``None`` when the event cannot be salvaged (e.g. no
    participants resolve to known entities, or required roles are
    completely unfilled).
    """
    participants = normalize_participants(
        raw_participants, type_def=type_def, entity_type_map=entity_type_map
    )
    if not participants:
        logger.debug("Dropping event: no participants resolved (type=%s)", event_type)
        return None

    if not has_all_required_roles(type_def, participants):
        logger.debug(
            "Dropping event: required roles missing (type=%s, got=%s)",
            event_type,
            list(participants.keys()),
        )
        return None

    timestamp = parse_timestamp(raw_timestamp)

    text_spans: list[TextSpan] = []
    chunk_ids = [chunk_id] if chunk_id else []
    if evidence_text and evidence_text.strip():
        text_spans.append(TextSpan(text=evidence_text.strip(), chunk_id=chunk_id))
    elif source_text:
        span = find_evidence_span(
            source_text,
            list({e for entities in participants.values() for e in entities}),
            chunk_id=chunk_id,
        )
        if span is not None:
            text_spans.append(span)

    coverage = required_role_coverage(type_def, participants)
    if llm_confidence is None:
        confidence = 0.5 + 0.5 * coverage
    else:
        try:
            confidence = max(0.0, min(1.0, float(llm_confidence)))
        except (TypeError, ValueError):
            confidence = 0.5 + 0.5 * coverage

    provenance = EventProvenance(
        document_id=document_id,
        chunk_ids=chunk_ids,
        text_spans=text_spans,
        extracted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        extractor_version=_drg_version,
        extraction_method=extraction_method,  # type: ignore[arg-type]
        confidence=confidence,
    )

    canonical_index = {name.lower(): name for name in entity_type_map.keys()}
    canonical_location: str | None = None
    if location and isinstance(location, str) and location.strip():
        canonical_location = canonical_index.get(
            location.strip().lower(), location.strip()
        )

    cleaned_props: dict[str, Any] = {}
    if isinstance(properties, dict):
        for k, v in properties.items():
            if v is None or v == "" or v == []:
                continue
            cleaned_props[str(k)] = v

    eid = make_event_id(
        event_type=event_type,
        participants=participants,
        timestamp=timestamp,
        evidence_text=text_spans[0].text if text_spans else None,
    )

    return Event(
        id=eid,
        event_type=event_type,
        participants=participants,
        timestamp=timestamp,
        location=canonical_location,
        properties=cleaned_props,
        provenance=provenance,
        metadata={},
    )


def _precision_value(p: TimestampPrecision) -> int:  # pragma: no cover - tiny helper
    return {"year": 1, "month": 2, "day": 3, "instant": 4}.get(p, 0)
