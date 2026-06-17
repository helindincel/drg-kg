"""Unit tests for ``drg.coreference_resolution._scoring``.

The three scoring helpers (``semantic_similarity_score``,
``action_based_score``, ``matches_svo_pattern``) are strategy-agnostic
and side-effect free, so we test them with plain stubs — no spaCy
required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from drg.coreference_resolution._scoring import (
    action_based_score,
    matches_svo_pattern,
    semantic_similarity_score,
)


class _FakeDocSlice:
    """Just enough of a spaCy ``Span`` for the slice path under test."""

    def __init__(self, text: str):
        self.text = text


class _FakeDoc:
    """Implements ``doc[a:b].text`` and nothing else."""

    def __init__(self, text: str):
        self._text = text

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Mirror spaCy: slicing returns a Span whose ``.text`` is the
            # joined token text. We approximate with a character slice of the
            # underlying string — the function only inspects ``.text``.
            return _FakeDocSlice(self._text[key.start or 0 : key.stop])
        raise TypeError(f"Unexpected key: {key!r}")


class _StubEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping

    def embed(self, text: str) -> list[float]:
        return self._mapping.get(text, [0.0, 0.0, 0.0])


class _BrokenEmbedder:
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedder offline")


class TestSemanticSimilarityScore:
    def test_returns_zero_when_provider_missing(self):
        doc = _FakeDoc("Some context window text.")
        assert semantic_similarity_score("Alice", 5, doc, None) == 0.0

    def test_returns_zero_when_window_is_empty(self):
        # Position 0 with stop 10 against empty doc → empty window.
        doc = _FakeDoc("")
        embedder = _StubEmbedder({"Alice": [1.0, 0.0, 0.0]})
        assert semantic_similarity_score("Alice", 0, doc, embedder) == 0.0

    def test_returns_zero_when_window_is_whitespace(self):
        doc = _FakeDoc("     \n\t   ")
        embedder = _StubEmbedder({"Alice": [1.0, 0.0, 0.0]})
        assert semantic_similarity_score("Alice", 5, doc, embedder) == 0.0

    def test_returns_one_for_identical_embeddings(self):
        doc = _FakeDoc("Alice met Bob at the lab today.")
        embedder = _StubEmbedder(
            {
                "Alice": [1.0, 0.0, 0.0],
                # The function builds the window from the doc — we make the
                # embedder return the same vector regardless via default.
            }
        )
        # Force matching vectors for both lookups by mapping the exact window text.
        # We don't know the exact slice content; rely on the default fallback
        # being [0,0,0] and only mapping "Alice" to a non-zero vector causes
        # the norm-product to be zero → score 0.
        # To get score 1 we need both embeddings equal — set both via mapping.
        window_text = doc[max(0, 5 - 10) : 5 + 10].text
        embedder._mapping[window_text] = [1.0, 0.0, 0.0]
        score = semantic_similarity_score("Alice", 5, doc, embedder)
        assert score == pytest.approx(1.0, rel=1e-6)

    def test_returns_zero_for_orthogonal_embeddings(self):
        doc = _FakeDoc("Bob discussed something else.")
        window_text = doc[max(0, 5 - 10) : 5 + 10].text
        embedder = _StubEmbedder(
            {
                "Alice": [1.0, 0.0, 0.0],
                window_text: [0.0, 1.0, 0.0],
            }
        )
        score = semantic_similarity_score("Alice", 5, doc, embedder)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_returns_zero_when_norm_is_zero(self):
        doc = _FakeDoc("Hello world today.")
        window_text = doc[max(0, 5 - 10) : 5 + 10].text
        embedder = _StubEmbedder(
            {
                "Alice": [0.0, 0.0, 0.0],
                window_text: [0.0, 0.0, 0.0],
            }
        )
        assert semantic_similarity_score("Alice", 5, doc, embedder) == 0.0

    def test_negative_similarity_clamps_to_zero(self):
        doc = _FakeDoc("Hello world today.")
        window_text = doc[max(0, 5 - 10) : 5 + 10].text
        embedder = _StubEmbedder(
            {
                "Alice": [1.0, 0.0, 0.0],
                window_text: [-1.0, 0.0, 0.0],
            }
        )
        # Cosine similarity would be -1; the scorer clamps to 0.
        assert semantic_similarity_score("Alice", 5, doc, embedder) == 0.0

    def test_returns_zero_when_embedder_raises(self):
        doc = _FakeDoc("Some context window text.")
        assert semantic_similarity_score("Alice", 5, doc, _BrokenEmbedder()) == 0.0


class TestActionBasedScore:
    def test_no_action_keyword_yields_zero(self):
        assert action_based_score("Alice", "She walked to the park.", "Person") == 0.0

    def test_entity_in_context_with_matching_preposition(self):
        # "spoke about iPhone" + "iPhone" present + preposition "about" → 0.5
        score = action_based_score("iPhone", "Tim spoke about iPhone yesterday.", "Product")
        assert score == 0.5

    def test_person_type_without_entity_in_context_gets_smaller_boost(self):
        # "spoke about" matches but "Alice" not in context → 0.3 (Person bias)
        score = action_based_score("Alice", "Tim spoke about iPhones yesterday.", "Person")
        assert score == 0.3

    def test_non_person_type_without_entity_match_returns_zero(self):
        # Action matches but neither entity-in-context nor Person type → 0.0
        score = action_based_score("Tesla", "He spoke about cars.", "Company")
        assert score == 0.0

    def test_is_case_insensitive(self):
        score = action_based_score(
            "iPhone", "He SPOKE ABOUT iPhone yesterday.", "Product"
        )
        assert score == 0.5

    def test_requires_preposition_after_action(self):
        # "spoke" alone (no allowed preposition follows) should not boost.
        score = action_based_score("Alice", "She spoke yesterday.", "Person")
        assert score == 0.0

    def test_recognises_all_action_keywords(self):
        # Each key action keyword should fire when paired with its preposition.
        cases = [
            ("spoke", "about", "Alice spoke about iPhone."),
            ("discussed", "regarding", "He discussed regarding iPhone."),
            ("mentioned", "about", "She mentioned about iPhone."),
            ("wrote", "on", "They wrote on iPhone."),
            ("created", "by", "iPhone created by Tim."),
        ]
        for _action, _prep, sentence in cases:
            score = action_based_score("iPhone", sentence, "Product")
            assert score == 0.5, sentence


def _fake_sentence(text: str, subject_tokens: list[str], other_tokens: list[str] | None = None):
    """Build a stub spaCy ``Span`` exposing ``.text`` and an iterator of tokens."""
    tokens = []
    for tok in subject_tokens:
        tokens.append(SimpleNamespace(text=tok, dep_="nsubj"))
    for tok in other_tokens or []:
        tokens.append(SimpleNamespace(text=tok, dep_="ROOT"))
    return SimpleNamespace(text=text, __iter__=lambda self=None: iter(tokens))


class _FakeSentence:
    def __init__(self, text: str, tokens: list[tuple[str, str]]):
        self.text = text
        self._tokens = [SimpleNamespace(text=tok, dep_=dep) for tok, dep in tokens]

    def __iter__(self):
        return iter(self._tokens)


class TestMatchesSvoPattern:
    def test_subject_with_object_preposition_in_pronoun_context(self):
        prev = _FakeSentence(
            "Alice presented her findings.",
            [("Alice", "nsubj"), ("presented", "ROOT"), ("findings", "dobj")],
        )
        assert matches_svo_pattern("Alice", prev, "She talked about her findings.") is True

    def test_subject_with_subjpass_dep_also_qualifies(self):
        prev = _FakeSentence(
            "The findings were presented by Alice.",
            [("findings", "nsubjpass"), ("presented", "ROOT")],
        )
        assert matches_svo_pattern("findings", prev, "They were discussed about thoroughly.") is True

    def test_entity_not_in_subject_position_returns_false(self):
        prev = _FakeSentence(
            "Bob met Alice yesterday.",
            [("Bob", "nsubj"), ("Alice", "dobj")],
        )
        assert matches_svo_pattern("Alice", prev, "She works about lab matters.") is False

    def test_subject_match_without_object_keyword_returns_false(self):
        prev = _FakeSentence(
            "Alice arrived.",
            [("Alice", "nsubj"), ("arrived", "ROOT")],
        )
        assert matches_svo_pattern("Alice", prev, "She left quickly.") is False

    def test_broken_input_returns_false_silently(self):
        # Passing a non-iterable triggers the bare except branch.
        assert matches_svo_pattern("Alice", object(), "about anything") is False

    def test_case_insensitive_subject_match(self):
        prev = _FakeSentence(
            "ALICE spoke at the event.",
            [("ALICE", "nsubj"), ("spoke", "ROOT")],
        )
        assert matches_svo_pattern("alice", prev, "She talked about the event.") is True

    def test_recognises_multiple_object_keywords(self):
        prev = _FakeSentence(
            "Alice wrote the paper.",
            [("Alice", "nsubj"), ("wrote", "ROOT")],
        )
        for keyword in ("about", "regarding", "concerning", "on", "for", "with"):
            assert matches_svo_pattern("Alice", prev, f"She thought {keyword} it.") is True
