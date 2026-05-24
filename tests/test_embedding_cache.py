"""Tests for drg.utils.cache.EmbeddingCache and the shared-cache registry."""

from __future__ import annotations

import threading

import pytest

from drg.utils.cache import (
    EmbeddingCache,
    cached_provider,
    clear_cached_providers,
)


class _CountingProvider:
    """Minimal provider implementation that counts embed/embed_batch calls."""

    def __init__(self) -> None:
        self.embed_calls = 0
        self.batch_calls = 0
        self.last_batch_size = 0

    def embed(self, text: str):
        self.embed_calls += 1
        return [float(len(text))]

    def embed_batch(self, texts):
        self.batch_calls += 1
        self.last_batch_size = len(texts)
        return [[float(len(t))] for t in texts]

    def get_dimension(self) -> int:
        return 1


@pytest.fixture(autouse=True)
def _clear_registry():
    """Each test starts with an empty shared-cache registry."""
    clear_cached_providers()
    yield
    clear_cached_providers()


def test_embed_returns_provider_result_on_miss():
    provider = _CountingProvider()
    cache = EmbeddingCache(provider, capacity=4)

    vec = cache.embed("Apple")

    assert vec == [5.0]
    assert provider.embed_calls == 1
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0


def test_embed_returns_cached_result_on_hit():
    provider = _CountingProvider()
    cache = EmbeddingCache(provider, capacity=4)

    cache.embed("Apple")
    vec = cache.embed("Apple")

    assert vec == [5.0]
    assert provider.embed_calls == 1
    assert cache.stats.hits == 1


def test_normalize_collapses_whitespace_and_casing():
    provider = _CountingProvider()
    cache = EmbeddingCache(provider, capacity=4, normalize=True)

    cache.embed("Apple")
    cache.embed("  apple  ")

    assert provider.embed_calls == 1
    assert cache.stats.hits == 1


def test_normalize_false_keeps_distinct_keys():
    provider = _CountingProvider()
    cache = EmbeddingCache(provider, capacity=4, normalize=False)

    cache.embed("Apple")
    cache.embed("apple")

    assert provider.embed_calls == 2
    assert cache.stats.hits == 0


def test_lru_eviction_drops_oldest_entry():
    provider = _CountingProvider()
    cache = EmbeddingCache(provider, capacity=2)

    cache.embed("a")
    cache.embed("b")
    cache.embed("c")  # forces eviction of "a"

    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache
    assert len(cache) == 2


def test_embed_batch_separates_hits_and_misses():
    provider = _CountingProvider()
    cache = EmbeddingCache(provider, capacity=8)
    cache.embed("hello")

    result = cache.embed_batch(["hello", "world", "foo"])

    assert len(result) == 3
    assert provider.batch_calls == 1
    assert provider.last_batch_size == 2  # "hello" was cached
    assert cache.stats.hits == 1
    assert cache.stats.misses == 3


def test_embed_batch_falls_back_when_no_batch_method():
    class NoBatchProvider:
        def __init__(self):
            self.calls = 0

        def embed(self, text):
            self.calls += 1
            return [float(len(text))]

    provider = NoBatchProvider()
    cache = EmbeddingCache(provider, capacity=4)

    result = cache.embed_batch(["a", "bb", "ccc"])

    assert [v[0] for v in result] == [1.0, 2.0, 3.0]
    assert provider.calls == 3


def test_clear_resets_state_and_stats():
    provider = _CountingProvider()
    cache = EmbeddingCache(provider, capacity=4)
    cache.embed("a")
    cache.embed("b")

    cache.clear()

    assert len(cache) == 0
    assert cache.stats.hits == 0
    assert cache.stats.misses == 0


def test_invalid_capacity_raises():
    provider = _CountingProvider()
    with pytest.raises(ValueError):
        EmbeddingCache(provider, capacity=0)


def test_attribute_delegation_to_underlying_provider():
    provider = _CountingProvider()
    cache = EmbeddingCache(provider, capacity=4)

    assert cache.get_dimension() == 1


def test_shared_cache_returns_same_wrapper():
    provider = _CountingProvider()

    c1 = cached_provider(provider, capacity=10)
    c2 = cached_provider(provider, capacity=10)

    assert c1 is c2


def test_shared_cache_returns_new_wrapper_when_capacity_differs():
    provider = _CountingProvider()

    c1 = cached_provider(provider, capacity=10)
    c2 = cached_provider(provider, capacity=20)

    assert c1 is not c2
    assert c2.stats.capacity == 20


def test_stats_hit_rate_basis_points():
    provider = _CountingProvider()
    cache = EmbeddingCache(provider, capacity=4)

    cache.embed("a")
    cache.embed("a")
    cache.embed("a")  # 2 hits, 1 miss

    stats = cache.stats.as_dict()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate_bp"] == 6666  # 2/3 ≈ 66.66%


def test_concurrent_access_safe():
    """Smoke check: many threads embedding the same key shouldn't crash."""
    provider = _CountingProvider()
    cache = EmbeddingCache(provider, capacity=64)

    def worker():
        for _ in range(50):
            cache.embed("shared-key")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cache.stats.hits + cache.stats.misses == 8 * 50
    assert "shared-key" in cache
