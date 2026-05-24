"""Embedding abstraction layer for semantic representations."""

from .providers import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    create_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "GeminiEmbeddingProvider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "OpenRouterEmbeddingProvider",
    "create_embedding_provider",
]
