"""
DSPy LM Configuration Module

Handles DSPy language model configuration from environment variables.
Separated from extract.py to improve testability and follow single responsibility principle.
"""

import logging
import os
import threading
import warnings

import dspy

from .errors import LLMConfigError
from .utils.env_loader import load_dotenv

logger = logging.getLogger(__name__)


class LMConfig:
    """Language model configuration handler.

    Handles configuration of DSPy LM from environment variables.
    Uses class-level state so that multiple LMConfig instances share the same
    configuration flag, preventing accidental re-configuration of the DSPy
    global LM from different call sites (including async / multi-threaded use).
    """

    # Class-level flag and lock: shared across all instances.
    _configured: bool = False
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize LM config (does not configure DSPy yet)."""
        # Instance owns nothing; state is class-level.

    def configure(self) -> None:
        """Configure DSPy LM from environment variables.

        This method is idempotent — calling it multiple times from any instance
        has no effect after the first successful configuration.
        """
        with LMConfig._lock:
            if LMConfig._configured:
                return
            self._configure_unsafe()

    def _configure_unsafe(self) -> None:
        """Inner configure — must be called only while holding self._lock."""

        # Load local .env if present (keeps API keys out of code)
        load_dotenv(".env", override=False)

        # Read configuration from environment variables
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

        # Read API keys from environment
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        # Validate model/API-key compatibility
        model_lower = model.lower()
        api_key = None

        if "openrouter" in model_lower:
            api_key = openrouter_key
            if not api_key:
                warnings.warn(
                    f"OpenRouter model ({model}) selected but OPENROUTER_API_KEY not found. "
                    "OPENROUTER_API_KEY is required.",
                    UserWarning,
                    stacklevel=2,
                )
            # Set base URL for OpenRouter if not already configured
            if not os.getenv("DRG_BASE_URL"):
                base_url = "https://openrouter.ai/api/v1"
        elif "gemini" in model_lower:
            api_key = gemini_key
            if not api_key:
                warnings.warn(
                    f"Gemini model ({model}) selected but GEMINI_API_KEY not found. "
                    "GEMINI_API_KEY is required.",
                    UserWarning,
                    stacklevel=2,
                )
        elif "anthropic" in model_lower or "claude" in model_lower:
            # Direct Anthropic (not via OpenRouter)
            api_key = anthropic_key
            if not api_key:
                warnings.warn(
                    f"Anthropic model ({model}) selected but ANTHROPIC_API_KEY not found. "
                    "ANTHROPIC_API_KEY is required.",
                    UserWarning,
                    stacklevel=2,
                )
        elif "perplexity" in model_lower:
            api_key = perplexity_key
            if not api_key:
                warnings.warn(
                    f"Perplexity model ({model}) selected but PERPLEXITY_API_KEY not found. "
                    "PERPLEXITY_API_KEY is required.",
                    UserWarning,
                    stacklevel=2,
                )
            # Set base URL for Perplexity if not already configured
            if not os.getenv("DRG_BASE_URL"):
                base_url = "https://api.perplexity.ai"
        elif "ollama" in model_lower:
            # Ollama does not require an API key
            api_key = None
        else:
            # OpenAI or other cloud models
            api_key = openai_key
            if not api_key and not model_lower.startswith("ollama"):
                warnings.warn(
                    f"Cloud model ({model}) selected but OPENAI_API_KEY not found. "
                    "An API key may be required.",
                    UserWarning,
                    stacklevel=2,
                )

        base_url = os.getenv("DRG_BASE_URL")
        temperature = float(os.getenv("DRG_TEMPERATURE", "0.0"))
        max_tokens = int(os.getenv("DRG_MAX_TOKENS", "1500"))  # Budget cap for LLM output

        # Configure DSPy LM
        # Set fallback base URLs for providers that need them
        if "openrouter" in model_lower and not base_url:
            base_url = "https://openrouter.ai/api/v1"

        # Set fallback base URL for Perplexity
        if "perplexity" in model_lower and not base_url:
            base_url = "https://api.perplexity.ai"

        # Build DSPy LM kwargs
        lm_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # OpenRouter: validate/normalise model name and pass credentials
        if "openrouter" in model_lower:
            # Normalise to LiteLLM format: ensure "openrouter/" prefix
            if not model.startswith("openrouter/"):
                lm_kwargs["model"] = f"openrouter/{model}"
            else:
                # Already has openrouter/ prefix
                lm_kwargs["model"] = model

            # LiteLLM needs api_key/api_base in kwargs for OpenRouter
            if api_key:
                # Set env var (LiteLLM reads it automatically)
                os.environ["OPENROUTER_API_KEY"] = api_key
                # Also pass in kwargs as a fallback
                if "kwargs" not in lm_kwargs:
                    lm_kwargs["kwargs"] = {}
                lm_kwargs["kwargs"]["api_key"] = api_key
                if base_url:
                    lm_kwargs["kwargs"]["api_base"] = base_url
        elif api_key:
            # Set env var for other providers
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

            # Pass credentials in kwargs as a fallback
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
        LMConfig._configured = True
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

    Resets the class-level configured flag so that the next call to
    configure() (on any instance) will re-apply DSPy settings.
    """
    global _global_lm_config
    with LMConfig._lock:
        LMConfig._configured = False
    _global_lm_config = LMConfig()
