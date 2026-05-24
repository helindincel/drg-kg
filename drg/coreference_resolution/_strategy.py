"""Coreference resolution strategy interface.

The legacy ``CoreferenceResolver`` was a single class with two private
``_resolve_with_*`` methods. We extract those methods into strategies that
share this small contract. New backends (LLM coref, transformer-based, etc.)
plug in by subclassing this ABC — no edits to the dispatcher required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResolverConfig:
    """Strategy-shared configuration.

    Kept tiny on purpose: anything that's strategy-specific (e.g. spaCy model
    name) stays inside the strategy class.
    """

    language: str = "en"
    embedding_provider: Any | None = None
    # Conservative gating — abstain when we're not confident enough.
    min_resolution_score: float = 0.85
    min_resolution_margin: float = 0.15


class CoreferenceStrategy(ABC):
    """Abstract base for coreference resolution strategies.

    Implementations consume ``(text, entities, relations)`` and return the same
    relations with any resolvable pronouns rewritten. They must NEVER raise on
    expected failure modes (missing model, empty text, ambiguous pronoun);
    instead they should return the input untouched and log a warning.
    """

    def __init__(self, config: ResolverConfig):
        self.config = config

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when this strategy can actually run (e.g. model loaded)."""

    @abstractmethod
    def resolve(
        self,
        text: str,
        entities: list[tuple[str, str]],
        relations: list[tuple[str, str, str]],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
        """Resolve pronouns; return ``(entities, resolved_relations)``."""
