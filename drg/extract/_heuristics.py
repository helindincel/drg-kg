"""Deterministic post-processing heuristics for extracted relations.

These helpers are intentionally minimal and conservative — they prefer
abstaining (None / False) over guessing. They are applied to LLM outputs to
attach optional metadata (negation, year-only temporal info) and to infer a
small set of schema-gated implicit relations from surface patterns.

This module is intentionally English-first (with a small Turkish carve-out for
possessive detection) and gated by `DRG_LANGUAGE` for negation/temporal cues.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from ..schema import DRGSchema, EnhancedDRGSchema
from ._relations import _normalize_schema

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


# ---------------------------------------------------------------------------
# Implicit relation inference (schema-gated, surface patterns)
# ---------------------------------------------------------------------------

# Conservative candidate relations for possessive inference.
# We intentionally avoid overly generic "has" and reverse-like "part_of" here;
# schema authors can still express composition via has_part.
_POSSESSIVE_CANDIDATE_RELS: list[str] = ["owns", "has_part"]

# English + Turkish genitive suffixes (for "A'nın B" style).
_TR_GEN_SUFFIXES: list[str] = ["nın", "nin", "nun", "nün", "ın", "in", "un", "ün"]

# Two-hop inference categories.
_OWNERSHIP_RELS: set[str] = {"owns", "has_part"}
_LOCATION_RELS: set[str] = {"located_in", "located_at", "hosts", "contains"}
_CANDIDATE_OWNER_LOCATION_RELS: list[str] = [
    "operates_in",
    "based_in",
    "headquartered_in",
]

# Operation cues that license two-hop owner-location inference. Conservative.
_OPERATION_CUE_PATTERNS: list[str] = [
    r"(?i)\boperat(?:e|es|ing|ed)\b",
    r"(?i)\brun(?:s|ning|ned)?\b",
    r"(?i)\bmanage(?:s|d|ment|ing)?\b",
    r"(?i)\bemploy(?:s|ed|ing)?\b",
    r"(?i)\bwork(?:s|ed|ing)?\b",
    r"(?i)\bfaaliyet\b",
    r"(?i)\bçalıştır(?:ıyor|di|mak|ma)?\b",
    r"(?i)\bçalış(?:ıyor|tı|mak|ma)?\b",
]


def _infer_implicit_relations(
    text: str,
    entities: list[tuple[str, str]],
    schema: DRGSchema | EnhancedDRGSchema,
    existing_triples: list[tuple[str, str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    """Infer a small set of implicit relations from surface patterns (schema-gated).

    Only emits relations that exist in the provided schema. This complements
    LLM extraction for cases like "Tesla's Gigafactory" / "Tesla'nın
    Gigafactory'si".
    """
    if not text or not entities:
        return []

    normalized = _normalize_schema(schema)
    legacy_rel_types = {(r.src, r.name, r.dst) for r in normalized.relations}

    def _allows(rel: str, s_type: str, o_type: str) -> bool:
        if isinstance(schema, EnhancedDRGSchema):
            return schema.is_valid_relation(rel, s_type, o_type)
        return (s_type, rel, o_type) in legacy_rel_types

    type_map: dict[str, str] = {name: etype for name, etype in entities if name and etype}
    entity_names = [name for name, _ in entities if name]
    if len(entity_names) < 2:
        return []

    text_l = text.lower()
    entity_names_sorted = sorted(entity_names, key=lambda s: len(s), reverse=True)

    inferred: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set(existing_triples or [])

    def _try_add(a: str, b: str) -> None:
        a_type = type_map.get(a)
        b_type = type_map.get(b)
        if not a_type or not b_type:
            return
        for rel in _POSSESSIVE_CANDIDATE_RELS:
            if _allows(rel, a_type, b_type):
                t = (a, rel, b)
                if t not in seen:
                    inferred.append(t)
                    seen.add(t)
                return

    def _has_possessive(a: str, b: str) -> bool:
        a_esc = re.escape(a)
        b_esc = re.escape(b)
        # English: "A's B" (apostrophe variants)
        en_pat = rf"(?i)(?<!\w){a_esc}(?:'s|’s)\s+{b_esc}(?!\w)"
        if re.search(en_pat, text):
            return True
        # Turkish: "A'nın B" with optional apostrophe and common genitive suffixes
        suf_alt = "|".join(re.escape(s) for s in _TR_GEN_SUFFIXES)
        tr_pat = rf"(?i)(?<!\w){a_esc}(?:'?\s*(?:{suf_alt}))\s+{b_esc}(?!\w)"
        return re.search(tr_pat, text) is not None

    for a in entity_names_sorted:
        for b in entity_names_sorted:
            if a == b:
                continue
            if _has_possessive(a, b):
                _try_add(a, b)

    # Two-hop inference (schema-gated, input-agnostic):
    # If (A owns/has_part B) and (B located_in L) then infer
    # (A operates_in/based_in/headquartered_in L) when schema allows.
    if existing_triples:
        has_operation_cue = any(re.search(p, text) for p in _OPERATION_CUE_PATTERNS)
        if not has_operation_cue:
            return inferred

        owner_to_asset: dict[str, set[str]] = {}
        asset_to_location: dict[str, set[str]] = {}

        combined = list(existing_triples) + inferred
        for s, r, o in combined:
            if r in _OWNERSHIP_RELS:
                owner_to_asset.setdefault(s, set()).add(o)
            if r in _LOCATION_RELS:
                asset_to_location.setdefault(s, set()).add(o)

        for owner, assets in owner_to_asset.items():
            owner_type = type_map.get(owner)
            if not owner_type:
                continue
            for asset in assets:
                # Safety: require the asset mention itself in text to avoid
                # accidental type-shaped inference from unrelated triples.
                if asset.lower() not in text_l:
                    continue
                for loc in asset_to_location.get(asset, set()):
                    loc_type = type_map.get(loc)
                    if not loc_type:
                        continue
                    for rel in _CANDIDATE_OWNER_LOCATION_RELS:
                        if _allows(rel, owner_type, loc_type):
                            t = (owner, rel, loc)
                            if t not in seen:
                                inferred.append(t)
                                seen.add(t)
                            break

    return inferred
