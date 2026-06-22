"""Embedding provider implementations."""

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Model name string
        """
        pass

    def get_cost_per_token(self) -> float:
        """Get cost per token (optional, for cost tracking).

        Returns:
            Cost per token in USD (default: 0.0)
        """
        return 0.0


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        """Initialize OpenAI embedding provider.

        Args:
            model: Model name (text-embedding-3-small or text-embedding-3-large)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
        """
        try:
            import openai
        except ImportError as err:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from err

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self.client = openai.OpenAI(api_key=self.api_key)

        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

    def embed(self, text: str) -> list[float]:
        """Embed single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        # OpenAI supports batch in single call
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimensions.get(self.model, 1536)

    def get_model_name(self) -> str:
        """Get model name."""
        return f"openai/{self.model}"

    def get_cost_per_token(self) -> float:
        """Get cost per token."""
        # Approximate costs (as of 2024)
        costs = {
            "text-embedding-3-small": 0.02 / 1_000_000,  # $0.02 per 1M tokens
            "text-embedding-3-large": 0.13 / 1_000_000,  # $0.13 per 1M tokens
        }
        return costs.get(self.model, 0.0)


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embedding provider."""

    def __init__(
        self,
        model: str = "models/embedding-001",
        api_key: str | None = None,
    ):
        """Initialize Gemini embedding provider.

        Args:
            model: Model name (default: models/embedding-001)
            api_key: Gemini API key (default: from GEMINI_API_KEY env var)
        """
        try:
            import google.generativeai as genai
        except ImportError as err:
            raise ImportError(
                "google-generativeai package is required. Install with: pip install google-generativeai"
            ) from err

        self.model_name = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY env var or pass api_key."
            )

        genai.configure(api_key=self.api_key)
        self.genai_module = genai

    def embed(self, text: str) -> list[float]:
        """Embed single text."""
        result = self.genai_module.embed_content(
            model=self.model_name, content=text, task_type="semantic_similarity"
        )
        return result["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        # Gemini embedding API
        embeddings = []
        for text in texts:
            result = self.genai_module.embed_content(
                model=self.model_name, content=text, task_type="semantic_similarity"
            )
            embeddings.append(result["embedding"])
        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        # Gemini embedding-001 is typically 768 dimensions
        return 768

    def get_model_name(self) -> str:
        """Get model name."""
        return f"gemini/{self.model_name}"

    def get_cost_per_token(self) -> float:
        """Get cost per token."""
        # Gemini embedding is free (as of 2024)
        return 0.0


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """OpenRouter embedding provider (unified API)."""

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """Initialize OpenRouter embedding provider.

        Args:
            model: Model name (e.g., "openai/text-embedding-3-small")
            api_key: OpenRouter API key (default: from OPENROUTER_API_KEY env var)
            base_url: OpenRouter API base URL
        """
        try:
            import openai
        except ImportError as err:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from err

        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY env var or pass api_key."
            )

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def embed(self, text: str) -> list[float]:
        """Embed single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        # Default to 1536 for OpenAI models, may vary
        if "large" in self.model:
            return 3072
        return 1536

    def get_model_name(self) -> str:
        """Get model name."""
        return f"openrouter/{self.model}"

    def get_cost_per_token(self) -> float:
        """Get cost per token."""
        # OpenRouter costs vary by model and are subject to change
        # For accurate cost calculation, consult OpenRouter pricing documentation
        # or implement model-specific cost lookup
        return 0.0


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize local embedding provider.

        Args:
            model_name: HuggingFace model name
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            ) from err

        self.model_name = model_name
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Embed single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def get_model_name(self) -> str:
        """Get model name."""
        return f"local/{self.model_name}"

    def get_cost_per_token(self) -> float:
        """Get cost per token (local is free)."""
        return 0.0


def create_embedding_provider(
    provider: str = "openai", model: str | None = None, **kwargs
) -> EmbeddingProvider:
    """Factory function to create embedding provider.

    Args:
        provider: Provider name ("openai", "gemini", "openrouter", "local")
        model: Model name (optional, uses defaults if not provided)
        **kwargs: Additional provider-specific parameters

    Returns:
        EmbeddingProvider instance
    """
    provider_lower = provider.lower()

    if provider_lower == "openai":
        model = model or "text-embedding-3-small"
        return OpenAIEmbeddingProvider(model=model, **kwargs)

    elif provider_lower == "gemini":
        model = model or "models/embedding-001"
        return GeminiEmbeddingProvider(model=model, **kwargs)

    elif provider_lower == "openrouter":
        model = model or "openai/text-embedding-3-small"
        return OpenRouterEmbeddingProvider(model=model, **kwargs)

    elif provider_lower == "local":
        model = model or "sentence-transformers/all-MiniLM-L6-v2"
        return LocalEmbeddingProvider(model_name=model)

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
