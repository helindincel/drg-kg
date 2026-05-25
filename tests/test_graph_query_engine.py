"""Unit tests for drg.graph.query_engine.

Covers the deterministic, no-LLM KG query path used by the UI:
  - `_normalize`, `_tokenize`, `_parse_relation_filter`, `_score_entity`
  - `execute_query` end-to-end on a small fixture KG.
"""

from __future__ import annotations

from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode
from drg.graph.query_engine import (
    QueryResult,
    _normalize,
    _parse_relation_filter,
    _score_entity,
    _tokenize,
    execute_query,
)


def _fixture_kg() -> EnhancedKG:
    """Small fixture: 5 nodes, 4 edges, two relation types."""
    kg = EnhancedKG()
    for nid, ntype in [
        ("Alice", "Person"),
        ("Bob", "Person"),
        ("Acme Inc", "Company"),
        ("Globex", "Company"),
        ("Paris", "Place"),
    ]:
        kg.add_node(KGNode(id=nid, type=ntype, properties={}, metadata={}))
    edges = [
        ("Alice", "Acme Inc", "works_at"),
        ("Bob", "Globex", "works_at"),
        ("Alice", "Paris", "lives_in"),
        ("Bob", "Paris", "lives_in"),
    ]
    for s, t, r in edges:
        kg.add_edge(
            KGEdge(
                source=s,
                target=t,
                relationship_type=r,
                relationship_detail=f"{s} {r} {t}",
                metadata={},
            )
        )
    return kg


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_collapses_whitespace():
    assert _normalize("  Hello   WORLD  ") == "hello world"


def test_normalize_handles_empty_and_none_like():
    assert _normalize("") == ""
    assert _normalize("   ") == ""


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


def test_tokenize_drops_stopwords_and_short_tokens():
    tokens = _tokenize("Who is the manager of Acme")
    # "is", "the", "of", "who" are stopwords; single-letter tokens stripped.
    assert "manager" in tokens
    assert "acme" in tokens
    assert "is" not in tokens
    assert "the" not in tokens


def test_tokenize_strips_punctuation():
    tokens = _tokenize("Alice, (Bob), Carol")
    assert "alice" in tokens
    assert "bob" in tokens
    assert "carol" in tokens


def test_tokenize_empty_query_returns_empty():
    assert _tokenize("") == []


# ---------------------------------------------------------------------------
# _parse_relation_filter
# ---------------------------------------------------------------------------


def test_parse_relation_filter_supports_parenthesised_form():
    assert _parse_relation_filter("Alice (works_at)") == "works_at"


def test_parse_relation_filter_supports_relation_prefix():
    assert _parse_relation_filter("Alice relation:works_at neighbours") == "works_at"


def test_parse_relation_filter_supports_rel_equals():
    assert _parse_relation_filter("Alice rel=works_at") == "works_at"


def test_parse_relation_filter_returns_none_when_no_marker():
    assert _parse_relation_filter("Who works with Alice?") is None


def test_parse_relation_filter_rejects_multi_word_parens():
    # Multi-word content inside parens shouldn't be treated as a relation name.
    assert _parse_relation_filter("Alice (works at)") is None


def test_parse_relation_filter_returns_none_for_empty():
    assert _parse_relation_filter("") is None


# ---------------------------------------------------------------------------
# _score_entity
# ---------------------------------------------------------------------------


def test_score_entity_exact_match_is_highest():
    score, reasons = _score_entity("Alice", "alice", ["alice"])
    assert score >= 5.0
    assert "exact_match" in reasons


def test_score_entity_substring_match_scores_positive():
    score, reasons = _score_entity("Alice Wonderland", "alice", ["alice"])
    assert score > 0
    assert "substring_match" in reasons


def test_score_entity_token_overlap_contributes():
    score, reasons = _score_entity("Acme Inc", "acme corp", ["acme", "corp"])
    assert score > 0
    assert any(r.startswith("token_overlap") for r in reasons)


def test_score_entity_returns_zero_for_unrelated():
    score, reasons = _score_entity("Bob", "completely different query", ["completely", "different"])
    assert score == 0
    assert reasons == []


# ---------------------------------------------------------------------------
# execute_query (end-to-end)
# ---------------------------------------------------------------------------


def test_execute_query_returns_QueryResult_type():
    kg = _fixture_kg()
    result = execute_query(kg, "Alice")
    assert isinstance(result, QueryResult)


def test_execute_query_finds_exact_match_as_seed():
    kg = _fixture_kg()
    result = execute_query(kg, "Alice")
    assert "Alice" in result.seed_entities


def test_execute_query_returns_incident_edges():
    kg = _fixture_kg()
    result = execute_query(kg, "Alice")
    # Two edges incident to Alice (works_at, lives_in)
    assert len(result.matched_edges) >= 2
    edge_targets = {e["target"] for e in result.matched_edges}
    assert "Acme Inc" in edge_targets or "Paris" in edge_targets


def test_execute_query_respects_relation_filter():
    kg = _fixture_kg()
    result = execute_query(kg, "Alice (works_at)")
    # Only works_at edges should be returned
    rel_types = {e["relationship_type"] for e in result.matched_edges}
    assert rel_types == {"works_at"}


def test_execute_query_falls_back_to_top_degree_when_no_match():
    kg = _fixture_kg()
    result = execute_query(kg, "completely unrelated nonsense")
    # No name matches → falls back to top-degree entities. Paris has degree 2
    # (the highest in this fixture along with Alice/Bob).
    assert len(result.seed_entities) >= 1


def test_execute_query_caps_edges_at_k_edges():
    kg = _fixture_kg()
    result = execute_query(kg, "Alice", k_edges=1)
    assert len(result.matched_edges) <= 1


def test_execute_query_answer_field_describes_outcome():
    kg = _fixture_kg()
    result = execute_query(kg, "Alice")
    assert "entities" in result.answer.lower() or "edges" in result.answer.lower()


def test_execute_query_matched_entities_are_sorted_case_insensitively():
    kg = _fixture_kg()
    result = execute_query(kg, "Alice")
    lowered = [s.lower() for s in result.matched_entities]
    assert lowered == sorted(lowered)


def test_execute_query_empty_kg_returns_empty_result():
    kg = EnhancedKG()
    result = execute_query(kg, "any query")
    assert result.seed_entities == [] or len(result.seed_entities) == 0
    assert result.matched_edges == []
