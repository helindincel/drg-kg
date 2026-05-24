"""Low-level similarity primitives used by entity-resolution strategies.

These functions are stateless and have no DRG-specific dependencies — they
can be unit-tested in isolation and reused elsewhere if needed.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np

__all__ = ["cosine_similarity", "similarity_score"]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity clamped to ``[0, 1]``.

    Raises:
        ValueError: when the two vectors have different dimensionality —
            indicates a provider mismatch the caller almost certainly wants
            to know about.
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions must match: {len(vec1)} != {len(vec2)}")

    v1 = np.asarray(vec1)
    v2 = np.asarray(vec2)
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = float(np.dot(v1, v2) / (norm1 * norm2))
    return max(0.0, min(1.0, similarity))


def _word_boundary_contains(short: str, long: str) -> bool:
    """True when ``short`` appears in ``long`` as a whole word (≥ 3 chars).

    Prevents "elena" from matching inside "selena" while still catching
    "Elena Vasquez" → "Elena".
    """
    if len(short) < 3:
        return False
    return re.search(rf"(?i)(?<!\w){re.escape(short)}(?!\w)", long) is not None


def similarity_score(str1: str, str2: str) -> float:
    """String similarity for entity names.

    Combines exact match, word-boundary containment and ``SequenceMatcher``
    edit distance. Has a deliberate safety rail: two **different**
    single-token names always return ``0.0`` (e.g. "Elena" vs "Selena") to
    avoid catastrophic merges driven by edit distance alone.

    Returns a score in ``[0, 1]``.
    """
    s1 = str1.lower().strip()
    s2 = str2.lower().strip()

    if s1 == s2:
        return 1.0

    # Safety rule: never merge two different single-token names by string
    # similarity alone — explicit aliases or embeddings must back that up.
    if (" " not in s1) and (" " not in s2) and s1 != s2:
        return 0.0

    if _word_boundary_contains(s1, s2) or _word_boundary_contains(s2, s1):
        shorter = min(len(s1), len(s2))
        longer = max(len(s1), len(s2))
        base_score = shorter / longer if longer else 0.0
        # Boundary-contained aliases get aggressive but bounded boosts.
        boosted_score = max(0.75, min(0.95, 0.75 + base_score * 0.25))
        seq_similarity = SequenceMatcher(None, s1, s2).ratio()
        return max(boosted_score, seq_similarity)

    return SequenceMatcher(None, s1, s2).ratio()
