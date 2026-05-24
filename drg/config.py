"""
DSPy LM Configuration Module

Handles DSPy language model configuration from environment variables.
Separated from extract.py to improve testability and follow single responsibility principle.
"""

import logging
import os
import warnings

import dspy

from .errors import LLMConfigError
from .utils.env_loader import load_dotenv

logger = logging.getLogger(__name__)


class LMConfig:
    """Language model configuration handler.

    Handles configuration of DSPy LM from environment variables.
    No global state - each instance is independent.
    """

    def __init__(self):
        """Initialize LM config (does not configure DSPy yet)."""
        self._configured = False

    def configure(self) -> None:
        """Configure DSPy LM from environment variables.

        This method is idempotent - calling it multiple times has no effect
        after the first successful configuration.
        """
        if self._configured:
            return

        # Load local .env if present (keeps API keys out of code)
        load_dotenv(".env", override=False)

        # Environment variable'lardan otomatik oku
        model = os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
        # Normalize common Gemini model formats to LiteLLM-friendly IDs.
        #
        # Users often paste "models/..." from Google docs, but LiteLLM's Gemini adapter already
        # prefixes requests with "models/". If we pass "models/..." through, the URL becomes
        # ".../models/models/<name>:generateContent" and returns 404.
        model_stripped = model.strip()

        # Accept "models/<name>" (Google docs) -> "gemini/<name>"
        if model_stripped.startswith("models/"):
            model_stripped = model_stripped[len("models/") :]

        # Accept "gemini/models/<name>" -> "gemini/<name>"
        if model_stripped.startswith("gemini/models/"):
            model_stripped = "gemini/" + model_stripped[len("gemini/models/") :]

        # Accept plain "gemini-..." without provider prefix -> "gemini/<name>"
        if model_stripped.startswith("gemini-") or model_stripped.startswith("gemini_"):
            model_stripped = f"gemini/{model_stripped}"

        # If after stripping we still have no provider prefix but it's a Gemini model name,
        # add the provider prefix.
        if not model_stripped.startswith("gemini/") and "gemini" in model_stripped.lower():
            # Avoid duplicating when user already passed e.g. "openrouter/..."
            if "/" not in model_stripped:
                model_stripped = f"gemini/{model_stripped}"

        model = model_stripped

        # API key'leri environment'tan oku
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        # Model ve API key uyumunu kontrol et
        model_lower = model.lower()
        api_key = None

        if "openrouter" in model_lower:
            api_key = openrouter_key
            if not api_key:
                warnings.warn(
                    f"OpenRouter model ({model}) seçildi ama OPENROUTER_API_KEY bulunamadı. "
                    "OpenRouter API key'i gerekli.",
                    UserWarning,
                    stacklevel=2,
                )
            # OpenRouter için base URL ayarla
            if not os.getenv("DRG_BASE_URL"):
                base_url = "https://openrouter.ai/api/v1"
        elif "gemini" in model_lower:
            api_key = gemini_key
            if not api_key:
                warnings.warn(
                    f"Gemini model ({model}) seçildi ama GEMINI_API_KEY bulunamadı. "
                    "Gemini API key'i gerekli.",
                    UserWarning,
                    stacklevel=2,
                )
        elif "anthropic" in model_lower or "claude" in model_lower:
            # OpenRouter üzerinden değilse direkt Anthropic
            api_key = anthropic_key
            if not api_key:
                warnings.warn(
                    f"Anthropic model ({model}) seçildi ama ANTHROPIC_API_KEY bulunamadı. "
                    "Anthropic API key'i gerekli.",
                    UserWarning,
                    stacklevel=2,
                )
        elif "perplexity" in model_lower:
            api_key = perplexity_key
            if not api_key:
                warnings.warn(
                    f"Perplexity model ({model}) seçildi ama PERPLEXITY_API_KEY bulunamadı. "
                    "Perplexity API key'i gerekli.",
                    UserWarning,
                    stacklevel=2,
                )
            # Perplexity için base URL ayarla (eğer belirtilmemişse)
            if not os.getenv("DRG_BASE_URL"):
                base_url = "https://api.perplexity.ai"
        elif "ollama" in model_lower:
            # Ollama için API key gerekmez
            api_key = None
        else:
            # OpenAI veya diğer modeller için
            api_key = openai_key
            if not api_key and not model_lower.startswith("ollama"):
                warnings.warn(
                    f"Cloud model ({model}) seçildi ama OPENAI_API_KEY bulunamadı. "
                    "API key gerekli olabilir.",
                    UserWarning,
                    stacklevel=2,
                )

        base_url = os.getenv("DRG_BASE_URL")
        temperature = float(os.getenv("DRG_TEMPERATURE", "0.0"))
        max_tokens = int(os.getenv("DRG_MAX_TOKENS", "1500"))  # Bütçe koruması için sınır

        # DSPy LM'ini konfigüre et
        # OpenRouter için özel base URL (eğer belirtilmemişse)
        if "openrouter" in model_lower and not base_url:
            base_url = "https://openrouter.ai/api/v1"

        # Perplexity için özel base URL (eğer belirtilmemişse)
        if "perplexity" in model_lower and not base_url:
            base_url = "https://api.perplexity.ai"

        # DSPy LM kwargs - temel parametreler
        lm_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # OpenRouter için özel konfigürasyon (LiteLLM üzerinden)
        if "openrouter" in model_lower:
            # OpenRouter için model adını doğrula
            if not model.startswith("openrouter/"):
                # Eğer prefix yoksa ekle (LiteLLM formatı)
                lm_kwargs["model"] = f"openrouter/{model}"
            else:
                # Zaten openrouter/ prefix'i var, direkt kullan
                lm_kwargs["model"] = model

            # LiteLLM OpenRouter için api_key ve api_base kwargs içinde geçilmeli
            if api_key:
                # Environment variable olarak set et (LiteLLM bunu otomatik okur)
                os.environ["OPENROUTER_API_KEY"] = api_key
                # Ayrıca kwargs içinde de geç (bazı durumlarda gerekebilir)
                if "kwargs" not in lm_kwargs:
                    lm_kwargs["kwargs"] = {}
                lm_kwargs["kwargs"]["api_key"] = api_key
                if base_url:
                    lm_kwargs["kwargs"]["api_base"] = base_url
        elif api_key:
            # Diğer servisler için api_key environment variable olarak set et
            if "gemini" in model_lower:
                # Different SDK/adapters use different env var names for Gemini.
                # Keep both to be robust (LiteLLM commonly reads GOOGLE_API_KEY).
                os.environ["GEMINI_API_KEY"] = api_key
                os.environ["GOOGLE_API_KEY"] = api_key
            elif "anthropic" in model_lower or "claude" in model_lower:
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif "perplexity" in model_lower:
                os.environ["PERPLEXITY_API_KEY"] = api_key
            else:
                os.environ["OPENAI_API_KEY"] = api_key

            # kwargs içinde de geç
            if "kwargs" not in lm_kwargs:
                lm_kwargs["kwargs"] = {}
            lm_kwargs["kwargs"]["api_key"] = api_key
            if base_url:
                lm_kwargs["kwargs"]["api_base"] = base_url

        try:
            lm = dspy.LM(**lm_kwargs)
            dspy.configure(lm=lm)
        except Exception as e:
            raise LLMConfigError(
                f"DSPy LM configuration failed for model {model!r}: {e}. "
                "Check your API keys and model identifier."
            ) from e
        self._configured = True
        logger.info(f"DSPy LM configured: {model}")


# Global instance for backward compatibility (but now it's a class, not a global function)
_global_lm_config = LMConfig()


def configure_lm() -> None:
    """Configure DSPy LM from environment variables.

    This is a convenience function that uses a global LMConfig instance.
    For better testability, use LMConfig() directly in your code.
    """
    _global_lm_config.configure()


def reset_lm_config() -> None:
    """Reset global LM config (useful for testing).

    This allows reconfiguration in tests.
    """
    global _global_lm_config
    _global_lm_config = LMConfig()
