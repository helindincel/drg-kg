"""Deterministic cross-chunk context selection.

The helpers here only reuse text that has already been ingested as part of the
input document, using string-indexed selection based on entity mentions.
"""

from __future__ import annotations

import math
import re

_PRONOUN_LIKE: set[str] = {
    # English
    "he",
    "she",
    "it",
    "they",
    "him",
    "her",
    "them",
    "his",
    "hers",
    "its",
    "their",
    "theirs",
    "that",
    "this",
    # Turkish
    "o",
    "onlar",
    "ona",
    "onu",
    "onun",
    "onların",
    "bu",
    "şu",
}


def _build_cross_chunk_context_snippets(
    chunk_texts: list[str],
    entity_to_chunks: dict[str, list[int]],
    anchor_entities: list[str],
    current_chunk_index: int,
    max_chunks: int,
    snippet_chars: int,
    max_total_chars: int = 1200,
    min_anchor_len: int = 3,
) -> list[str]:
    """Build short evidence snippets from other chunks sharing anchor entities.

    Deterministic, string-indexed intra-document evidence selection.
    """
    if not chunk_texts or not anchor_entities or max_chunks <= 0:
        return []

    filtered_anchors: list[str] = []
    for ent in anchor_entities:
        if not ent:
            continue
        ent_s = ent.strip()
        if len(ent_s) < min_anchor_len:
            continue
        if ent_s.lower() in _PRONOUN_LIKE:
            continue
        if ent_s.isdigit():
            continue
        filtered_anchors.append(ent_s)
    if not filtered_anchors:
        return []

    def _contains_entity(text: str, entity: str) -> bool:
        # Word-boundary match to reduce substring collisions ("us" in "business").
        pattern = r"(?i)(?<!\w)" + re.escape(entity) + r"(?!\w)"
        return re.search(pattern, text) is not None

    candidates: set[int] = set()
    for ent in filtered_anchors:
        idxs = entity_to_chunks.get(ent.lower(), [])
        for j in idxs:
            if j != current_chunk_index:
                candidates.add(j)
    if not candidates:
        return []

    def _anchor_hits(j: int) -> int:
        t = chunk_texts[j] or ""
        return sum(1 for a in filtered_anchors if _contains_entity(t, a))

    ranked = sorted(
        candidates,
        key=lambda j: (-_anchor_hits(j), abs(j - current_chunk_index), j),
    )
    ranked = [j for j in ranked if _anchor_hits(j) > 0][:max_chunks]

    def _snippet_for(text: str, term: str) -> str:
        if not term:
            return text[:snippet_chars].strip()
        pattern = r"(?i)(?<!\w)" + re.escape(term) + r"(?!\w)"
        m = re.search(pattern, text)
        if not m:
            return text[:snippet_chars].strip()
        pos = m.start()
        start = max(0, pos - snippet_chars // 2)
        end = min(len(text), start + snippet_chars)
        return text[start:end].strip()

    snippets: list[str] = []
    for j in ranked:
        t = chunk_texts[j]
        if not t:
            continue
        chosen = None
        for ent in filtered_anchors:
            if _contains_entity(t, ent):
                chosen = ent
                break
        excerpt = _snippet_for(t, chosen) if chosen else t[:snippet_chars].strip()
        snippets.append(f"Chunk {j + 1} excerpt: {excerpt}")

    # Enforce total context budget.
    if max_total_chars > 0 and snippets:
        out: list[str] = []
        total = 0
        for s in snippets:
            if total + len(s) > max_total_chars:
                break
            out.append(s)
            total += len(s)
        return out
    return snippets


def _select_anchor_entities(
    chunk_text: str,
    chunk_entities: list[tuple[str, str]],
    entity_to_chunks: dict[str, list[int]],
    total_chunks: int,
    min_anchor_len: int,
    max_anchors: int,
) -> list[str]:
    """Select a small, robust set of anchor entities for cross-chunk evidence injection.

    Uses TF-IDF-like scoring:
        score = tf_in_current_chunk * (log((N+1)/(df+1)) + 1)

    Downweights entities that appear in many chunks (too generic) and prioritizes
    entities that are salient in the current chunk.
    """
    if not chunk_text or not chunk_entities or max_anchors <= 0:
        return []

    def _count_occurrences(term: str) -> int:
        if not term:
            return 0
        pattern = r"(?i)(?<!\w)" + re.escape(term) + r"(?!\w)"
        return len(re.findall(pattern, chunk_text))

    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for name, _ in chunk_entities:
        if not name:
            continue
        term = name.strip()
        if term.lower() in seen:
            continue
        seen.add(term.lower())
        if len(term) < min_anchor_len:
            continue
        if term.lower() in _PRONOUN_LIKE:
            continue
        if term.isdigit():
            continue
        tf = _count_occurrences(term)
        if tf <= 0:
            continue
        df = len(entity_to_chunks.get(term.lower(), []))
        idf = math.log((total_chunks + 1) / (df + 1)) + 1.0
        # Slight preference for longer names.
        score = tf * idf * (1.0 + min(len(term) / 20.0, 1.0) * 0.1)
        scored.append((score, term))

    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    return [t for _, t in scored[:max_anchors]]
