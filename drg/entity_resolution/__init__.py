"""Entity Resolution package.

Merges entity mentions that refer to the same real-world thing but appear
with different surface forms. Domain-agnostic: works for any entity type
(Person, Company, Product, Location, …) and any language.

Public surface (kept stable for backward compatibility)
-------------------------------------------------------

- :func:`resolve_entities_and_relations` — top-level helper used by the
  extraction pipeline.
- :class:`EntityResolver` — class entry point. The constructor accepts the
  same keyword arguments as the legacy version (``similarity_threshold``,
  ``use_normalization``, ``adaptive_threshold``, ``embedding_provider``,
  ``use_embedding``, ``embedding_weight``, ``min_merge_margin``) and selects
  an appropriate :class:`SimilarityStrategy` under the hood.
- Low-level utilities (``normalize_entity_name``, ``similarity_score``,
  ``cosine_similarity``) remain importable from this package — examples and
  research scripts use them directly.

Architecture
============

::

    drg/entity_resolution/
        __init__.py        # Public API + EntityResolver factory
        _normalize.py      # normalize_entity_name + title/suffix patterns
        _similarity.py     # cosine_similarity + similarity_score
        _strategy.py       # SimilarityStrategy ABC + String / Hybrid impls
        _resolver.py       # EntityResolver (strategy DI'd in)

Pipeline ordering
=================

Coreference resolution (``drg.coreference_resolution``) should always run
*before* entity resolution: it rewrites pronouns into explicit mentions,
which entity resolution can then merge.
"""

from __future__ import annotations

import logging
from typing import Any

from ._normalize import normalize_entity_name
from ._resolver import EntityResolver as _BaseResolver
from ._similarity import cosine_similarity, similarity_score
from ._strategy import HybridSimilarity, SimilarityStrategy, StringSimilarity

logger = logging.getLogger(__name__)

__all__ = [
    "EntityResolver",
    "HybridSimilarity",
    # Strategies (re-exported for advanced users / tests)
    "SimilarityStrategy",
    "StringSimilarity",
    "cosine_similarity",
    # Low-level helpers (legacy public surface)
    "normalize_entity_name",
    "resolve_entities_and_relations",
    "similarity_score",
]


# ---------------------------------------------------------------------------
# Backward-compatible EntityResolver constructor
# ---------------------------------------------------------------------------


class EntityResolver(_BaseResolver):
    """User-facing :class:`EntityResolver`.

    Accepts the historic keyword set and wires up a :class:`SimilarityStrategy`
    for you. Pass ``similarity_strategy=<custom>`` to bypass the auto-wiring
    entirely (preferred for tests and bespoke backends).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        use_normalization: bool = True,
        adaptive_threshold: bool = True,
        embedding_provider: Any = None,
        use_embedding: bool = True,
        embedding_weight: float = 0.7,
        min_merge_margin: float = 0.08,
        *,
        similarity_strategy: SimilarityStrategy | None = None,
    ):
        strategy = similarity_strategy or _build_similarity_strategy(
            use_normalization=use_normalization,
            use_embedding=use_embedding,
            embedding_provider=embedding_provider,
            embedding_weight=embedding_weight,
        )
        super().__init__(
            similarity_strategy=strategy,
            similarity_threshold=similarity_threshold,
            adaptive_threshold=adaptive_threshold,
            use_normalization=use_normalization,
            min_merge_margin=min_merge_margin,
        )


def _build_similarity_strategy(
    *,
    use_normalization: bool,
    use_embedding: bool,
    embedding_provider: Any,
    embedding_weight: float,
) -> SimilarityStrategy:
    """Select the appropriate strategy based on legacy kwargs.

    Tries (in order):
        1. The caller's explicit ``embedding_provider``.
        2. The default ``local`` provider, if importable.
        3. String-only similarity.
    """
    if not use_embedding:
        return StringSimilarity(use_normalization=use_normalization)

    if embedding_provider is not None:
        provider_cls = _get_embedding_provider_class()
        if provider_cls is not None and not isinstance(embedding_provider, provider_cls):
            logger.warning(
                "Provided embedding_provider is not an EmbeddingProvider; "
                "falling back to string similarity"
            )
        else:
            return HybridSimilarity(
                embedding_provider=embedding_provider,
                embedding_weight=embedding_weight,
                use_normalization=use_normalization,
            )

    default_provider = _try_default_embedding_provider()
    if default_provider is not None:
        return HybridSimilarity(
            embedding_provider=default_provider,
            embedding_weight=embedding_weight,
            use_normalization=use_normalization,
        )

    return StringSimilarity(use_normalization=use_normalization)


def _get_embedding_provider_class():
    """Resolve the :class:`EmbeddingProvider` ABC lazily; absent in some installs."""
    try:
        from ..embedding import EmbeddingProvider

        return EmbeddingProvider
    except ImportError:
        return None


def _try_default_embedding_provider():
    """Best-effort: instantiate a local embedding provider if extras are installed."""
    try:
        from ..embedding import create_embedding_provider

        try:
            provider = create_embedding_provider("local")
            logger.info(
                "Using default local embedding provider: %s",
                provider.get_model_name(),
            )
            return provider
        except (ImportError, ValueError):
            logger.debug(
                "Local embedding provider not available; entity resolution will use "
                "string similarity only"
            )
    except ImportError:
        logger.debug(
            "Embedding module not available; entity resolution will use string similarity only"
        )
    return None


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def resolve_entities_and_relations(
    entities: list[tuple[str, str]],
    relations: list[tuple[str, str, str]],
    similarity_threshold: float = 0.65,
    adaptive_threshold: bool = True,
    embedding_provider: Any | None = None,
    use_embedding: bool = True,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Resolve both entities and the relations referring to them.

    Convenience wrapper that builds an :class:`EntityResolver` with the
    legacy keyword set and applies it in one call. For finer control (custom
    strategies, embedding caching shared across runs), instantiate
    :class:`EntityResolver` directly.
    """
    resolver = EntityResolver(
        similarity_threshold=similarity_threshold,
        adaptive_threshold=adaptive_threshold,
        embedding_provider=embedding_provider,
        use_embedding=use_embedding,
    )
    resolved_entities, name_mapping = resolver.resolve(entities)
    resolved_relations = resolver.resolve_relations(relations, name_mapping)
    return resolved_entities, resolved_relations
