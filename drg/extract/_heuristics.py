"""Deterministic post-processing heuristics for extracted relations.

These helpers are intentionally minimal and conservative — they prefer
abstaining (None / False) over guessing. They are applied only to fill missing
optional relation metadata (negation, year-only temporal info).

This module is intentionally English-first and gated by `DRG_LANGUAGE` for
negation/temporal cues.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Negation / temporal metadata (per-relation)
# ---------------------------------------------------------------------------


def _infer_relation_metadata_heuristic(
    text: str,
    relations: list[tuple[str, str, str]],
) -> dict[str, Any]:
    """Heuristic relation metadata inference (English-first, conservative).

    Only used when the LLM did not provide temporal/negation metadata.
    Prefers abstaining (None/False) over guessing.
    """
    if not text or not relations:
        return {"temporal_info": None, "negations": None}

    lang = os.getenv("DRG_LANGUAGE", "en").lower()
    if lang not in {"en", "english"}:
        return {"temporal_info": None, "negations": None}

    temporal_info: list[dict[str, str | None] | None] = []
    negations: list[bool] = []

    for s, r, o in relations:
        window = _find_evidence_window(text, s, o, window_chars=220)
        negations.append(_detect_negation_in_window(window, relation_name=r))
        temporal_info.append(_extract_year_temporal(window))

    return {"temporal_info": temporal_info, "negations": negations}


def _find_evidence_window(text: str, a: str, b: str, window_chars: int = 200) -> str:
    """Return a short window around the closest mentions of `a` and `b`."""
    if not text or not a or not b:
        return ""

    def _positions(term: str) -> list[int]:
        pattern = rf"(?i)(?<!\w){re.escape(term)}(?!\w)"
        return [m.start() for m in re.finditer(pattern, text)]

    pos_a = _positions(a)
    pos_b = _positions(b)
    if not pos_a or not pos_b:
        return ""

    best_pair = None
    best_dist = None
    for pa in pos_a:
        for pb in pos_b:
            d = abs(pa - pb)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_pair = (pa, pb)

    if best_pair is None:
        return ""

    # Prefer the sentence containing the closest pair to limit negation bleed
    # from adjacent sentences.
    center = int((best_pair[0] + best_pair[1]) / 2)
    left_punct = max(
        text.rfind(".", 0, center),
        text.rfind("?", 0, center),
        text.rfind("!", 0, center),
    )
    sent_start = 0 if left_punct == -1 else left_punct + 1
    right_candidates = [
        p
        for p in (text.find(".", center), text.find("?", center), text.find("!", center))
        if p != -1
    ]
    sent_end = (min(right_candidates) + 1) if right_candidates else len(text)
    sent = text[sent_start:sent_end].strip()
    if sent:
        return sent

    left = max(0, min(best_pair) - window_chars)
    right = min(len(text), max(best_pair) + window_chars)
    return text[left:right]


def _detect_negation_in_window(window: str, relation_name: str) -> bool:
    """Detect obvious negation cues near a relation mention (conservative)."""
    if not window:
        return False

    w = window.lower()
    cues = [
        "no longer",
        "never",
        "did not",
        "does not",
        "do not",
        "is not",
        "was not",
        "are not",
        "were not",
        "cannot",
        "can't",
        "won't",
        "ceased to",
        "stopped",
        "discontinued",
        "discontinue",
    ]
    if not any(c in w for c in cues):
        return False

    stem = (relation_name or "").split("_")[0].lower()
    if stem and len(stem) >= 4 and stem in w:
        return True

    # Special-case common production verbs (allow inflections).
    if any(k in w for k in ["produc", "manufactur", "mak"]) and stem in {
        "produces",
        "produce",
        "produced",
        "manufactures",
        "manufacture",
        "makes",
        "make",
    }:
        return True

    return False


def _extract_year_temporal(window: str) -> dict[str, str | None] | None:
    """Extract simple year-only temporal metadata from a window.

    Returns:
        ``{"start": "YYYY", "end": "YYYY", "precision": "year"}`` for ranges,
        ``{"start": "YYYY", "end": None, "precision": "year"}`` for single years,
        or ``None`` if no year detected.
    """
    if not window:
        return None
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", window)]
    if not years:
        return None
    years = sorted(set(years))

    m = re.search(
        r"\b(19\d{2}|20\d{2})\b\s*(?:-|to|until|through)\s*\b(19\d{2}|20\d{2})\b",
        window,
        flags=re.IGNORECASE,
    )
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2))
        return {
            "start": str(min(y1, y2)),
            "end": str(max(y1, y2)),
            "precision": "year",
        }

    return {"start": str(years[0]), "end": None, "precision": "year"}
