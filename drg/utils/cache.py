"""Thread-safe LRU cache wrapper for embedding providers.

Motivation
----------

The hot path during knowledge-graph construction calls ``provider.embed(name)``
once per entity, and entities recur heavily within a single document and across
extractions of related documents. Wrapping the provider in an in-memory LRU
cache typically reduces API spend dramatically with no code changes at the
call site.

Design
------

- :class:`EmbeddingCache` is a thin, dependency-injected wrapper that proxies
  ``embed`` / ``embed_batch`` to an underlying provider, caching results
  keyed on a normalized form of the text.
- :func:`cached_provider` is the recommended entry point — it returns the
  same object when called twice with the same provider, so callers can
  safely share caches across modules.
- The cache key normalizes whitespace and case so semantically-identical
  strings hit the same slot. Provider implementations that depend on exact
  casing should disable normalization with ``normalize=False``.

Scope
-----

This module is intentionally minimal. It does **not** persist across
processes; for that, plug a disk-backed store (e.g. ``shelve``,
``sqlite``, Redis) into the same protocol shape.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from .logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CacheStats",
    "EmbeddingCache",
    "cached_provider",
    "clear_cached_providers",
]


def _normalize_key(text: str, normalize: bool) -> str:
    """Return the cache key for ``text``.

    When ``normalize`` is True, the key is the case-folded, whitespace-collapsed
    form — so ``"Apple Inc."`` and ``"  apple inc.  "`` share a slot. When
    False, the raw string is used (useful for case-sensitive providers).
    """
    if not normalize:
        return text
    return " ".join(text.lower().split())


class CacheStats:
    """Tiny atomic counters for hit/miss observability.

    Read via :attr:`EmbeddingCache.stats`. Kept as a plain object (not a
    dataclass) so it's mutable and lock-free for the increment-only path.
    """

    __slots__ = ("capacity", "hits", "misses", "size")

    def __init__(self, capacity: int) -> None:
        self.hits = 0
        self.misses = 0
        self.size = 0
        self.capacity = capacity

    def as_dict(self) -> dict[str, int]:
        total = self.hits + self.misses
        hit_rate_bp = (self.hits * 10000 // total) if total else 0  # basis points
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": self.size,
            "capacity": self.capacity,
            "hit_rate_bp": hit_rate_bp,
        }


class EmbeddingCache:
    """Thread-safe LRU cache wrapper around an embedding provider.

    Args:
        provider: Any object with an ``embed(text) -> list[float]`` method.
            Optionally an ``embed_batch(texts) -> list[list[float]]`` method;
            if absent, batch calls fall back to repeated ``embed`` calls.
        capacity: Maximum number of cached vectors. When exceeded, the
            least-recently-used entry is evicted. Default ``4096``.
        normalize: Whether to case-fold and collapse whitespace before keying.
            Default ``True``.

    Example::

        from drg.utils.cache import EmbeddingCache
        cached = EmbeddingCache(provider, capacity=8192)
        vec = cached.embed("Apple")  # provider hit (miss)
        vec = cached.embed("apple")  # cache hit (normalized)
        cached.stats.as_dict()  # -> {'hits': 1, 'misses': 1, ...}
    """

    def __init__(
        self,
        provider: Any,
        *,
        capacity: int = 4096,
        normalize: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._provider = provider
        self._normalize = normalize
        self._lock = threading.RLock()
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        self.stats = CacheStats(capacity=capacity)

    # ----- Protocol -------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Return the embedding for ``text``, hitting the underlying provider on miss."""
        key = _normalize_key(text, self._normalize)

        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.stats.hits += 1
                return self._store[key]
            self.stats.misses += 1

        # Call the underlying provider OUTSIDE the lock so concurrent
        # embed() calls for different keys don't serialize on network I/O.
        vec = self._provider.embed(text)

        with self._lock:
            # Re-check: another thread may have populated the slot concurrently.
            existing = self._store.get(key)
            if existing is not None:
                self._store.move_to_end(key)
                return existing
            self._store[key] = vec
            self._evict_locked()
            self.stats.size = len(self._store)
        return vec

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Vectorized embed: cache hits are short-circuited; misses are batched."""
        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        with self._lock:
            for i, text in enumerate(texts):
                key = _normalize_key(text, self._normalize)
                if key in self._store:
                    self._store.move_to_end(key)
                    self.stats.hits += 1
                    results[i] = self._store[key]
                else:
                    self.stats.misses += 1
                    miss_indices.append(i)
                    miss_texts.append(text)

        if miss_texts:
            batch_fn = getattr(self._provider, "embed_batch", None)
            if callable(batch_fn):
                vecs = batch_fn(miss_texts)
            else:
                vecs = [self._provider.embed(t) for t in miss_texts]

            with self._lock:
                for idx, text, vec in zip(miss_indices, miss_texts, vecs, strict=False):
                    key = _normalize_key(text, self._normalize)
                    self._store[key] = vec
                    self._store.move_to_end(key)
                    results[idx] = vec
                self._evict_locked()
                self.stats.size = len(self._store)

        return [r for r in results if r is not None]

    # ----- Cache control -------------------------------------------------

    def clear(self) -> None:
        """Empty the cache and reset stats counters."""
        with self._lock:
            self._store.clear()
            self.stats.size = 0
            self.stats.hits = 0
            self.stats.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, text: str) -> bool:
        key = _normalize_key(text, self._normalize)
        with self._lock:
            return key in self._store

    # ----- Delegation -----------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate any non-cached methods to the underlying provider.

        This is how the wrapper preserves provider-specific extensions like
        ``get_dimension()``, ``get_model_name()`` etc. without listing them
        explicitly here.
        """
        return getattr(self._provider, name)

    # ----- Internal -------------------------------------------------------

    def _evict_locked(self) -> None:
        """Evict LRU entries while over capacity. Caller must hold the lock."""
        while len(self._store) > self.stats.capacity:
            self._store.popitem(last=False)


# ---------------------------------------------------------------------------
# Module-level cache registry — opt-in helper so independent modules can share
# the same wrapped provider without re-initializing the cache each time.
# ---------------------------------------------------------------------------

_CACHED_PROVIDERS: dict[int, EmbeddingCache] = {}
_REGISTRY_LOCK = threading.Lock()


def cached_provider(
    provider: Any,
    *,
    capacity: int = 4096,
    normalize: bool = True,
) -> EmbeddingCache:
    """Return an :class:`EmbeddingCache` wrapping ``provider``.

    Calling this multiple times with the same provider returns the same
    cache wrapper, so different modules see the same hit rate. Useful in
    pipelines that build the provider once at startup and pass it to
    several components (entity resolution, semantic search, etc.).
    """
    pid = id(provider)
    with _REGISTRY_LOCK:
        existing = _CACHED_PROVIDERS.get(pid)
        if existing is not None and existing.stats.capacity == capacity:
            return existing
        wrapper = EmbeddingCache(provider, capacity=capacity, normalize=normalize)
        _CACHED_PROVIDERS[pid] = wrapper
        logger.debug(
            "Created shared embedding cache (capacity=%d, normalize=%s)",
            capacity,
            normalize,
        )
        return wrapper


def clear_cached_providers() -> None:
    """Drop every shared cache wrapper. Intended for tests."""
    with _REGISTRY_LOCK:
        _CACHED_PROVIDERS.clear()
