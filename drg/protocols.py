"""Central protocol definitions for DRG components.

Interface-first design (per ``.cursorrules``): every pluggable component in DRG
implements a small, explicit contract defined here. Using :class:`typing.Protocol`
lets implementations duck-type the interface — they don't have to inherit a
specific ABC — which keeps existing concrete classes (``EmbeddingProvider``,
``ClusteringAlgorithm``, etc.) unchanged while still giving us a single source
of truth for type checking and dependency injection.

Decorating each protocol with :func:`runtime_checkable` allows ``isinstance``
checks at runtime, which is handy for clear error messages when a wrong object
is wired up. Keep these protocols **minimal** — only the methods the rest of
DRG actually calls. Anything richer belongs on the concrete class.

Public protocols
================

- :class:`EmbeddingProviderProtocol` — text → vector providers.
- :class:`ClusteringAlgorithmProtocol` — graph → communities.
- :class:`KGExtractorProtocol` — text → ``(entities, relations)``.
- :class:`LLMProtocol` — narrow DSPy-compatible LM surface, used for DI.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ClusteringAlgorithmProtocol",
    "EmbeddingProviderProtocol",
    "KGExtractorProtocol",
    "LLMProtocol",
]


@runtime_checkable
class EmbeddingProviderProtocol(Protocol):
    """Structural interface for embedding providers.

    Matches ``drg.embedding.providers.EmbeddingProvider`` so any concrete
    provider already satisfies this protocol without inheriting it.
    """

    def embed(self, text: str) -> list[float]:
        """Embed a single text into a dense vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; order matches the input list."""
        ...

    def get_dimension(self) -> int:
        """Return the dimensionality of the produced vectors."""
        ...

    def get_model_name(self) -> str:
        """Return a human-readable identifier for the underlying model."""
        ...


@runtime_checkable
class ClusteringAlgorithmProtocol(Protocol):
    """Structural interface for graph clustering algorithms.

    ``graph`` is intentionally typed as ``Any`` to avoid a hard dependency on a
    specific KG representation here — clustering algorithms only need a
    NetworkX-compatible view, which the KG class exposes.
    """

    def cluster(self, graph: Any) -> list[Any]:
        """Partition ``graph`` into a list of community/cluster objects."""
        ...


@runtime_checkable
class KGExtractorProtocol(Protocol):
    """Structural interface for KG extractors.

    Mirrors ``KGExtractor.forward`` so alternate implementations (rule-based,
    cached, mocked) can be substituted via dependency injection.

    The return type is intentionally ``Any``: real implementations return
    either an ``ExtractionResult`` or a ``dspy.Prediction`` depending on the
    DSPy environment. The pipeline reads ``.entities``/``.relations`` via
    attribute access, which both shapes support.
    """

    def forward(
        self,
        text: str,
        context_entities: list[tuple[str, str]] | None = None,
    ) -> Any:
        """Extract entities and relations from ``text``."""
        ...


@runtime_checkable
class LLMProtocol(Protocol):
    """Narrow LM surface for dependency-injected DSPy language models.

    Any object with a ``__call__`` returning string completions (DSPy ``dspy.LM``
    is the canonical example) satisfies this protocol. We deliberately don't
    require the full DSPy ``LM`` API here — we only ever need to plug it into
    ``dspy.settings`` or a per-extractor context manager.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
