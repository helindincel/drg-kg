"""Similarity strategies for entity resolution.

The previous monolithic ``EntityResolver._calculate_similarity`` mixed three
concerns:

1. Name normalization (kept in :mod:`_normalize`).
2. String similarity scoring (kept in :mod:`_similarity`).
3. Optional embedding-based scoring + caching.

This module exposes a small :class:`SimilarityStrategy` ABC so the resolver
can be wired with either pure-string similarity (no extra deps, fast) or a
hybrid string+embedding strategy (more accurate for surface variations).
Adding a new backend — e.g. an LLM-based judge — is a matter of subclassing
:class:`SimilarityStrategy`; the resolver doesn't need to change.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ._normalize import normalize_entity_name
from ._similarity import cosine_similarity, similarity_score

logger = logging.getLogger(__name__)

__all__ = [
    "HybridSimilarity",
    "SimilarityStrategy",
    "StringSimilarity",
]


class SimilarityStrategy(ABC):
    """Compute a similarity score in ``[0, 1]`` between two entity names."""

    @abstractmethod
    def score(self, name1: str, name2: str) -> float:
        """Return the similarity of two entity names."""

    # Convenience flag the resolver consults when reporting which path is active.
    @property
    def uses_embeddings(self) -> bool:
        return False


class StringSimilarity(SimilarityStrategy):
    """Pure string similarity (no embedding provider needed).

    Cheap, deterministic, and a safe default. Falls back here automatically
    when ``HybridSimilarity``'s embedding call fails.
    """

    def __init__(self, *, use_normalization: bool = True):
        self.use_normalization = use_normalization

    def score(self, name1: str, name2: str) -> float:
        norm1, norm2 = self._normalize_pair(name1, name2)
        return similarity_score(norm1, norm2)

    def _normalize_pair(self, name1: str, name2: str) -> tuple[str, str]:
        if not self.use_normalization:
            return name1.lower().strip(), name2.lower().strip()
        return normalize_entity_name(name1), normalize_entity_name(name2)


class HybridSimilarity(StringSimilarity):
    """Combine string similarity with embedding cosine similarity.

    The two signals are blended via ``embedding_weight``:

        combined = w * embedding_sim + (1 - w) * string_sim

    For substring matches the combined score is floored at ``0.75`` because
    embedding models can miss exact substring alignments that the string
    signal catches. Failures in the embedding path degrade gracefully back
    to the string score — never raised, only logged.

    The cache is keyed on the normalized name so it survives variant
    casing / honorifics.
    """

    def __init__(
        self,
        embedding_provider: Any,
        *,
        embedding_weight: float = 0.7,
        use_normalization: bool = True,
    ):
        super().__init__(use_normalization=use_normalization)
        self.embedding_provider = embedding_provider
        self.embedding_weight = embedding_weight
        self._embedding_cache: dict[str, list[float]] = {}

    @property
    def uses_embeddings(self) -> bool:
        return self.embedding_provider is not None

    def score(self, name1: str, name2: str) -> float:
        norm1, norm2 = self._normalize_pair(name1, name2)
        string_sim = similarity_score(norm1, norm2)

        if not self.uses_embeddings:
            return string_sim

        try:
            emb1 = self._get_embedding(name1, norm1)
            emb2 = self._get_embedding(name2, norm2)
            if not emb1 or not emb2:
                return string_sim
            embedding_sim = cosine_similarity(emb1, emb2)
        except Exception as e:
            logger.debug("Embedding similarity failed: %s; using string similarity only", e)
            return string_sim

        combined = (
            self.embedding_weight * embedding_sim + (1.0 - self.embedding_weight) * string_sim
        )

        # Substring evidence: floor at 0.75 if both length and ratio look like an alias.
        if norm1 in norm2 or norm2 in norm1:
            min_len = min(len(norm1), len(norm2))
            max_len = max(len(norm1), len(norm2))
            if min_len >= 3 and max_len > 0 and (min_len / max_len) > 0.2:
                combined = max(combined, 0.75)

        return combined

    def _get_embedding(self, original_name: str, cache_key: str) -> list[float] | None:
        """Embedding lookup with normalize-key cache. ``cache_key`` is the
        normalized form so cosmetic variants share an entry."""
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        if not self.embedding_provider:
            return None
        try:
            embedding = self.embedding_provider.embed(original_name)
        except Exception as e:
            logger.debug("Embedding fetch failed for %r: %s", original_name, e)
            return None
        self._embedding_cache[cache_key] = embedding
        return embedding
