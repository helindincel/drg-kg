"""Tests for the new :mod:`drg.entity_resolution` package layout and strategies.

These complement ``test_entity_resolution_safety.py`` (which covers the
public ``resolve_entities_and_relations`` API end-to-end).
"""

from __future__ import annotations

import pytest

from drg.entity_resolution import (
    EntityResolver,
    HybridSimilarity,
    SimilarityStrategy,
    StringSimilarity,
    cosine_similarity,
    normalize_entity_name,
    resolve_entities_and_relations,
    similarity_score,
)

# ---------------------------------------------------------------------------
# normalize_entity_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Dr. Elena Vasquez", "elena vasquez"),
        ("Prof. John Smith Jr.", "john smith"),
        ("  Cognitive   Enhancement  ", "cognitive enhancement"),
        ("Mr Smith", "smith"),
        ("Already normalized", "already normalized"),
    ],
)
def test_normalize_entity_name_strips_titles_and_collapses_spaces(raw, expected):
    assert normalize_entity_name(raw) == expected


# ---------------------------------------------------------------------------
# cosine_similarity / similarity_score
# ---------------------------------------------------------------------------


def test_cosine_similarity_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="Vector dimensions must match"):
        cosine_similarity([1.0, 0.0], [1.0])


def test_cosine_similarity_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_similarity_score_safety_rule_single_tokens_distinct():
    """The hard safety rule: two distinct single-token names always score 0."""
    assert similarity_score("Elena", "Selena") == 0.0


def test_similarity_score_substring_boosts():
    """Substring containment (≥3 chars) should produce a high score."""
    score = similarity_score("elena", "dr. elena vasquez")
    assert score >= 0.75


# ---------------------------------------------------------------------------
# Strategy ABC + implementations
# ---------------------------------------------------------------------------


def test_string_similarity_is_a_strategy():
    assert isinstance(StringSimilarity(), SimilarityStrategy)


def test_string_similarity_does_not_use_embeddings():
    assert StringSimilarity().uses_embeddings is False


def test_hybrid_falls_back_to_string_when_embedding_provider_missing():
    """``HybridSimilarity(None)`` must behave like ``StringSimilarity``."""
    hybrid = HybridSimilarity(embedding_provider=None)
    assert hybrid.uses_embeddings is False
    assert hybrid.score("Tesla", "Tesla") == pytest.approx(1.0)


def test_hybrid_degrades_when_embedding_call_raises():
    """If ``embed`` raises, hybrid must downgrade to string similarity, not crash."""

    class _BrokenProvider:
        def embed(self, _text):
            raise RuntimeError("embedding service down")

    hybrid = HybridSimilarity(embedding_provider=_BrokenProvider())
    score = hybrid.score("Dr. Elena Vasquez", "Elena Vasquez")
    # String similarity still works; we just shouldn't crash.
    assert 0.0 <= score <= 1.0


def test_hybrid_caches_embeddings_by_normalized_key():
    """The embedding cache key is the normalized name — repeated lookups
    of cosmetic variants should hit the cache."""

    calls = {"count": 0}

    class _CountingProvider:
        def embed(self, _text):
            calls["count"] += 1
            return [1.0, 0.0]

    hybrid = HybridSimilarity(embedding_provider=_CountingProvider())
    hybrid.score("Dr. Elena Vasquez", "Elena Vasquez")
    initial_calls = calls["count"]
    hybrid.score("dr. elena vasquez", "elena vasquez")  # same after normalization
    assert calls["count"] == initial_calls


# ---------------------------------------------------------------------------
# EntityResolver dispatcher / DI
# ---------------------------------------------------------------------------


def test_resolver_uses_string_strategy_when_embedding_disabled():
    resolver = EntityResolver(use_embedding=False)
    assert isinstance(resolver.similarity_strategy, StringSimilarity)
    assert resolver.use_embedding is False


def test_resolver_accepts_explicit_strategy():
    """``similarity_strategy=`` should bypass the auto-wiring entirely."""

    class _CustomStrategy(SimilarityStrategy):
        def score(self, n1, n2):
            return 0.9 if n1 == n2 else 0.0

    resolver = EntityResolver(similarity_strategy=_CustomStrategy())
    assert isinstance(resolver.similarity_strategy, _CustomStrategy)


def test_resolver_legacy_kwargs_preserved():
    """Construction with the historic kwarg set must still work and the
    ``similarity_threshold`` / ``adaptive_threshold`` / ``min_merge_margin``
    fields must reflect what was passed."""
    resolver = EntityResolver(
        similarity_threshold=0.7,
        use_embedding=False,
        adaptive_threshold=False,
        min_merge_margin=0.05,
    )
    assert resolver.similarity_threshold == 0.7
    assert resolver.adaptive_threshold is False
    assert resolver.min_merge_margin == 0.05


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------


def test_adaptive_threshold_substring_is_very_lenient():
    """Substring-related pairs should clear a very low bar."""
    resolver = EntityResolver(use_embedding=False)
    thr = resolver._get_adaptive_threshold("Elena", "Dr. Elena Vasquez")
    assert thr <= 0.40


def test_adaptive_threshold_disabled_returns_base():
    resolver = EntityResolver(use_embedding=False, adaptive_threshold=False)
    thr = resolver._get_adaptive_threshold("Elena", "Dr. Elena Vasquez")
    assert thr == resolver.base_similarity_threshold


# ---------------------------------------------------------------------------
# Public API end-to-end (smoke; deep coverage lives in test_entity_resolution_safety)
# ---------------------------------------------------------------------------


def test_resolve_entities_and_relations_merges_obvious_duplicates():
    entities = [
        ("Dr. Elena Vasquez", "Person"),
        ("Elena Vasquez", "Person"),
        ("Tesla", "Company"),
    ]
    relations = [
        ("Elena Vasquez", "works_at", "Tesla"),
        ("Dr. Elena Vasquez", "leads", "Tesla"),
    ]
    resolved_entities, resolved_relations = resolve_entities_and_relations(
        entities, relations, use_embedding=False
    )
    names = {n for n, _ in resolved_entities}
    assert len(resolved_entities) == 2  # Person merged + Company
    assert "Tesla" in names
    for s, _, o in resolved_relations:
        assert s in names
        assert o in names
