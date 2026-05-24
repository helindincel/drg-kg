"""Scoring helpers for coreference resolution.

These functions are deliberately strategy-agnostic and side-effect free:
given an entity and a context window they return a numeric boost. The NLP
strategy plugs them in to disambiguate pronouns when multiple candidates
are plausible.

Behaviour is preserved from the legacy ``CoreferenceResolver._get_*`` helpers;
only the call surface has been flattened so they can be unit tested in isolation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def semantic_similarity_score(
    entity_name: str,
    pronoun_pos: int,
    doc: Any,
    embedding_provider: Any | None,
) -> float:
    """Cosine similarity between ``entity_name`` and the window around the pronoun.

    Returns ``0.0`` when no embedding provider is available, the window is empty,
    or any error occurs (the resolver should degrade gracefully, never crash).
    """
    if embedding_provider is None:
        return 0.0

    try:
        context_window = doc[max(0, pronoun_pos - 10) : pronoun_pos + 10].text
        if not context_window.strip():
            return 0.0

        entity_emb = embedding_provider.embed(entity_name)
        context_emb = embedding_provider.embed(context_window)

        try:
            import numpy as np

            dot_product = np.dot(entity_emb, context_emb)
            norm_product = float(np.linalg.norm(entity_emb)) * float(np.linalg.norm(context_emb))
        except ImportError:
            logger.debug("numpy not available, using basic cosine similarity")
            dot_product = sum(a * b for a, b in zip(entity_emb, context_emb, strict=False))
            norm_a = sum(a * a for a in entity_emb) ** 0.5
            norm_b = sum(b * b for b in context_emb) ** 0.5
            norm_product = norm_a * norm_b

        if norm_product == 0:
            return 0.0
        similarity = dot_product / norm_product
        return max(0.0, float(similarity))
    except Exception as e:
        logger.debug(f"Semantic similarity calculation failed: {e}")
        return 0.0


_ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "spoke": ("about", "regarding", "on", "concerning"),
    "discussed": ("about", "regarding"),
    "mentioned": ("about",),
    "talked": ("about", "on"),
    "wrote": ("about", "on"),
    "created": ("by",),
    "designed": ("by",),
    "developed": ("by",),
}


def action_based_score(entity_name: str, context: str, entity_type: str) -> float:
    """Boost score when the pronoun sits in an action/topic context.

    Patterns like ``"spoke about iPhone"`` make ``Person`` entities more salient
    than other types. The result is intentionally small (≤ 0.5) so it can't
    overpower the structural distance signal.
    """
    context_lower = context.lower()
    score = 0.0
    for action, prepositions in _ACTION_KEYWORDS.items():
        if action not in context_lower:
            continue
        for prep in prepositions:
            if re.search(f"{action}.*{prep}", context_lower):
                if entity_name.lower() in context_lower:
                    score = 0.5
                if entity_type == "Person":
                    score = 0.3
    return score


def matches_svo_pattern(entity_name: str, prev_sent: Any, pronoun_context: str) -> bool:
    """Return True when ``entity_name`` is the subject of ``prev_sent`` and the
    pronoun context contains an object-style preposition.

    Heuristic; neural coref does this better. Failing silently (``False``) is
    correct because the caller treats SVO match as a small bonus only.
    """
    try:
        prev_sent_lower = prev_sent.text.lower()  # noqa: F841 - kept for parity
        pronoun_context_lower = pronoun_context.lower()
        entity_lower = entity_name.lower()

        entity_is_subject = False
        for token in prev_sent:
            if token.text.lower() == entity_lower and token.dep_ in {"nsubj", "nsubjpass"}:
                entity_is_subject = True
                break

        if entity_is_subject:
            object_keywords = ("about", "regarding", "concerning", "on", "for", "with")
            if any(keyword in pronoun_context_lower for keyword in object_keywords):
                return True
    except Exception:
        pass
    return False
