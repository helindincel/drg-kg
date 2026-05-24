"""Coreference Resolution package.

Handles pronoun and reference resolution to link pronouns (he, she, it, they)
and definite noun phrases (the company, this person) to their antecedent
entities. Complements entity resolution, which merges duplicate explicit
mentions.

Pipeline ordering (important):

1. Extract entities + relations (LLM extraction)
2. **Coreference resolution** (pronouns → explicit entities) — this module
3. Entity resolution (merge duplicate mentions)
4. Final KG construction

Public surface (kept stable for backward compatibility):

- :func:`resolve_coreferences` — top-level helper used by the extraction pipeline.
- :class:`CoreferenceResolver` — class-style entry point that picks the right
  strategy under the hood.

Architecture
============

The previous monolithic ``coreference_resolution.py`` has been split into a
strategy-based package. New strategies (LLM coref, transformer coref) plug in
by implementing :class:`CoreferenceStrategy` and registering with the resolver.

::

    drg/coreference_resolution/
        __init__.py             # public API + dispatcher
        _strategy.py            # CoreferenceStrategy ABC + ResolverConfig
        _pronouns.py            # language → pronoun/gender table
        _scoring.py             # pluggable scoring helpers (semantic, action, SVO)
        _heuristic_strategy.py  # regex-only fallback (no NLP dep)
        _nlp_strategy.py        # spaCy + neural coref overlay
"""

from __future__ import annotations

from typing import Any

from ..utils.logging import get_logger
from ._heuristic_strategy import HeuristicCoreferenceStrategy
from ._nlp_strategy import NLPCoreferenceStrategy
from ._strategy import CoreferenceStrategy, ResolverConfig

logger = get_logger(__name__)

__all__ = [
    "CoreferenceResolver",
    "CoreferenceStrategy",
    "HeuristicCoreferenceStrategy",
    "NLPCoreferenceStrategy",
    "ResolverConfig",
    "resolve_coreferences",
]


class CoreferenceResolver:
    """Pronoun and reference resolution dispatcher.

    Selects an :class:`NLPCoreferenceStrategy` when spaCy is available and the
    caller hasn't opted out (``use_nlp=False``); otherwise falls back to
    :class:`HeuristicCoreferenceStrategy`.

    Backward compatibility
    ----------------------
    The constructor signature and the :meth:`resolve` contract mirror the
    legacy ``CoreferenceResolver`` class so existing call sites don't change.
    Attributes such as ``use_nlp``, ``nlp``, ``neural_coref``, ``language``,
    ``embedding_provider`` and the gating knobs remain accessible for
    introspection / tests.
    """

    def __init__(
        self,
        use_nlp: bool = True,
        use_neural_coref: bool = True,
        embedding_provider: Any | None = None,
        language: str = "en",
    ):
        self.use_nlp = use_nlp
        self.use_neural_coref = use_neural_coref
        self.embedding_provider = embedding_provider
        self.language = (language or "en").lower()

        self.min_resolution_score = 0.85
        self.min_resolution_margin = 0.15

        self._config = ResolverConfig(
            language=self.language,
            embedding_provider=embedding_provider,
            min_resolution_score=self.min_resolution_score,
            min_resolution_margin=self.min_resolution_margin,
        )

        self._nlp_strategy: NLPCoreferenceStrategy | None = None
        self._heuristic_strategy = HeuristicCoreferenceStrategy(self._config)

        if use_nlp:
            self._nlp_strategy = NLPCoreferenceStrategy(
                self._config, use_neural_coref=use_neural_coref
            )
            # If spaCy/model wasn't installable, mark NLP off so introspection
            # via ``resolver.use_nlp`` reflects reality (parity with legacy).
            if not self._nlp_strategy.is_available():
                self.use_nlp = False

    # --- Legacy attribute mirrors (introspection / tests may read these) ---

    @property
    def nlp(self):
        return self._nlp_strategy.nlp if self._nlp_strategy else None

    @property
    def neural_coref(self) -> str | None:
        return self._nlp_strategy.neural_coref if self._nlp_strategy else None

    # --- Public API ---

    def resolve(
        self,
        text: str,
        entities: list[tuple[str, str]],
        relations: list[tuple[str, str, str]],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
        """Resolve pronouns in ``relations`` to explicit entity names.

        Args:
            text: Original text the entities/relations were extracted from.
            entities: ``(name, type)`` tuples.
            relations: ``(src, predicate, dst)`` tuples — predicates left intact;
                pronouns in ``src``/``dst`` may be rewritten.

        Returns:
            ``(entities, resolved_relations)``. Entities are returned unchanged
            because this module does not introduce new mentions, only rewrites
            existing pronoun references.
        """
        if not entities or not text:
            return entities, relations

        strategy = self._pick_strategy()
        return strategy.resolve(text, entities, relations)

    def _pick_strategy(self) -> CoreferenceStrategy:
        if self.use_nlp and self._nlp_strategy and self._nlp_strategy.is_available():
            return self._nlp_strategy
        return self._heuristic_strategy


def resolve_coreferences(
    text: str,
    entities: list[tuple[str, str]],
    relations: list[tuple[str, str, str]],
    use_nlp: bool = True,
    use_neural_coref: bool = True,
    embedding_provider: Any | None = None,
    language: str = "en",
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Convenience wrapper for :class:`CoreferenceResolver`.

    Strategies in order of preference:

    1. Neural coreference (``coreferee`` or ``neuralcoref``) if installed.
    2. spaCy structural scoring (with optional semantic / action boosts when an
       ``embedding_provider`` is passed).
    3. Pure-Python regex heuristics (no external deps).

    Domain-agnostic — works for any entity types. Conservative by default: when
    the top candidate isn't sufficiently better than the runner-up, the resolver
    abstains rather than guess.

    Args:
        text: Original text.
        entities: ``(name, type)`` tuples.
        relations: ``(src, predicate, dst)`` tuples.
        use_nlp: Use the spaCy strategy when available. Falls back to heuristics.
        use_neural_coref: Try to attach a neural coref backend on top of spaCy.
        embedding_provider: Optional embedding source for semantic disambiguation.
        language: ``"en"`` (default) or ``"tr"`` for Turkish pronoun support.

    Returns:
        ``(entities, resolved_relations)``.
    """
    resolver = CoreferenceResolver(
        use_nlp=use_nlp,
        use_neural_coref=use_neural_coref,
        embedding_provider=embedding_provider,
        language=language,
    )
    return resolver.resolve(text, entities, relations)
