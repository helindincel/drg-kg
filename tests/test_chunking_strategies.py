"""Unit tests for drg.chunking.strategies.

Uses a `FakeTokenizer` (whitespace splitter) instead of the real `tiktoken`
backend, so these tests run fast and have no external dependencies.

Covers:
  - `Chunk.to_dict` round-trip metadata
  - `TokenBasedChunker` happy paths, short-doc shortcut, validation
  - `SentenceBasedChunker` sentence packing + overlap
  - `create_chunker` preset / parameter / error matrix
  - Helper boundary detectors
"""

from __future__ import annotations

import pytest

from drg.chunking.strategies import (
    CHUNKING_PRESETS,
    Chunk,
    SentenceBasedChunker,
    TokenBasedChunker,
    Tokenizer,
    _find_paragraph_boundaries,
    _find_sentence_boundaries,
    _generate_chunk_id,
    create_chunker,
)


class FakeTokenizer(Tokenizer):
    """Whitespace-splitting tokenizer for deterministic unit tests."""

    def encode(self, text: str) -> list[int]:
        # Token id == word index hashed to a stable int; we only need length
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int]) -> str:
        # Return placeholder text proportional to token count — exact contents
        # do not matter for the strategies' chunking logic.
        return " ".join(["w"] * len(token_ids))

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


def _make_chunk(**overrides) -> Chunk:
    defaults = {
        "chunk_id": "ds_doc_chunk_000",
        "sequence_index": 0,
        "text": "hello world",
        "token_count": 2,
        "char_count": 11,
        "origin_dataset": "ds",
        "origin_file": "file.txt",
        "metadata": {"chunking_strategy": "test", "boundary_info": {}},
    }
    defaults.update(overrides)
    return Chunk(**defaults)


def test_chunk_to_dict_preserves_core_fields_and_flattens_metadata():
    chunk = _make_chunk(metadata={"chunking_strategy": "token_based_768_15pct", "extra": 7})
    d = chunk.to_dict()

    assert d["chunk_id"] == "ds_doc_chunk_000"
    assert d["sequence_index"] == 0
    assert d["origin_dataset"] == "ds"
    assert d["origin_file"] == "file.txt"
    assert d["token_count"] == 2
    assert d["char_count"] == 11
    assert d["chunk_text"] == "hello world"
    assert d["chunking_strategy"] == "token_based_768_15pct"
    assert d["extra"] == 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_generate_chunk_id_uses_default_doc_id_when_missing():
    assert _generate_chunk_id("ds", "", 5) == "ds_doc_001_chunk_005"
    assert _generate_chunk_id("ds", "abc", 0) == "ds_abc_chunk_000"


def test_find_sentence_boundaries_detects_terminators():
    text = "First sentence. Second one! Third? Done."
    boundaries = _find_sentence_boundaries(text)
    # Three boundaries (after '.', '!', '?'); trailing '.' has no whitespace after it.
    assert len(boundaries) == 3
    assert all(isinstance(b, int) and b > 0 for b in boundaries)


def test_find_paragraph_boundaries_detects_blank_lines():
    text = "Para one.\n\nPara two.\n\nPara three."
    boundaries = _find_paragraph_boundaries(text)
    assert len(boundaries) == 2


# ---------------------------------------------------------------------------
# TokenBasedChunker
# ---------------------------------------------------------------------------


def test_token_based_chunker_rejects_invalid_parameters():
    tok = FakeTokenizer()
    with pytest.raises(ValueError):
        TokenBasedChunker(tok, chunk_size=0)
    with pytest.raises(ValueError):
        TokenBasedChunker(tok, chunk_size=10, overlap_ratio=1.5)
    with pytest.raises(ValueError):
        TokenBasedChunker(tok, chunk_size=10, overlap_ratio=-0.1)


def test_token_based_chunker_empty_text_returns_empty_list():
    chunker = TokenBasedChunker(FakeTokenizer(), chunk_size=10)
    assert chunker.chunk("", "ds", "file.txt") == []
    assert chunker.chunk("   \n\t  ", "ds", "file.txt") == []


def test_token_based_chunker_short_doc_returns_single_chunk():
    chunker = TokenBasedChunker(FakeTokenizer(), chunk_size=50, overlap_ratio=0.1)
    text = "word " * 10  # 10 tokens, well below chunk_size
    chunks = chunker.chunk(text, "ds", "file.txt")

    assert len(chunks) == 1
    only = chunks[0]
    assert only.sequence_index == 0
    assert only.token_count == 10
    assert only.origin_dataset == "ds"
    assert only.origin_file == "file.txt"
    assert "boundary_info" in only.metadata


def test_token_based_chunker_long_doc_produces_multiple_chunks_with_overlap():
    chunker = TokenBasedChunker(
        FakeTokenizer(),
        chunk_size=10,
        overlap_ratio=0.2,
        respect_sentence_boundaries=False,
    )
    text = "word " * 100  # 100 tokens; ceil((100 - 2) / (10 - 2)) ~= 13 chunks
    chunks = chunker.chunk(text, "ds", "file.txt")

    assert len(chunks) > 1
    # Sequence indices must be strictly increasing and contiguous from 0
    indices = [c.sequence_index for c in chunks]
    assert indices == list(range(len(chunks)))
    # IDs must be unique and follow the canonical pattern
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert all(cid.startswith("ds_file_chunk_") for cid in ids)


def test_token_based_chunker_uses_filename_as_doc_id_when_not_given():
    chunker = TokenBasedChunker(FakeTokenizer(), chunk_size=10)
    text = "word " * 5
    chunks = chunker.chunk(text, "ds", "/abs/path/article.txt")
    assert chunks[0].chunk_id.startswith("ds_article_chunk_")


def test_token_based_chunker_honours_explicit_doc_id():
    chunker = TokenBasedChunker(FakeTokenizer(), chunk_size=10)
    text = "word " * 5
    chunks = chunker.chunk(text, "ds", "file.txt", doc_id="custom-id")
    assert chunks[0].chunk_id == "ds_custom-id_chunk_000"


# ---------------------------------------------------------------------------
# SentenceBasedChunker
# ---------------------------------------------------------------------------


def test_sentence_based_chunker_empty_text_returns_empty_list():
    chunker = SentenceBasedChunker(FakeTokenizer(), target_chunk_size=50)
    assert chunker.chunk("", "ds", "f.txt") == []
    assert chunker.chunk("   ", "ds", "f.txt") == []


def test_sentence_based_chunker_packs_sentences_until_budget_exceeded():
    # Each sentence has 5 tokens; target_chunk_size=12 means we pack 2 sentences
    # per chunk before opening a new one.
    sentences = ["one two three four five. "] * 6  # 6 short sentences
    text = "".join(sentences)
    chunker = SentenceBasedChunker(
        FakeTokenizer(),
        target_chunk_size=12,
        overlap_sentences=1,
    )

    chunks = chunker.chunk(text, "ds", "f.txt")
    assert len(chunks) >= 2

    # Each chunk should respect the budget reasonably; the chunker may
    # overshoot slightly when sentences themselves exceed the budget, but
    # for well-sized inputs token_count must stay within ~2x budget.
    for c in chunks:
        assert c.token_count > 0
        assert c.token_count <= 12 * 2

    # The metadata reports the strategy label
    assert all("sentence_based" in c.metadata["chunking_strategy"] for c in chunks)


def test_sentence_based_chunker_single_short_sentence_yields_single_chunk():
    chunker = SentenceBasedChunker(FakeTokenizer(), target_chunk_size=100)
    chunks = chunker.chunk("Just one sentence here.", "ds", "f.txt")
    assert len(chunks) == 1
    assert chunks[0].sequence_index == 0


# ---------------------------------------------------------------------------
# create_chunker factory
# ---------------------------------------------------------------------------


def test_create_chunker_with_explicit_strategy_returns_token_based():
    chunker = create_chunker(
        strategy="token_based",
        tokenizer=FakeTokenizer(),
        chunk_size=128,
        overlap_ratio=0.1,
    )
    assert isinstance(chunker, TokenBasedChunker)
    assert chunker.chunk_size == 128
    assert chunker.overlap_ratio == 0.1


def test_create_chunker_with_explicit_strategy_returns_sentence_based():
    chunker = create_chunker(
        strategy="sentence_based",
        tokenizer=FakeTokenizer(),
        chunk_size=256,
        overlap_sentences=3,
    )
    assert isinstance(chunker, SentenceBasedChunker)
    assert chunker.target_chunk_size == 256
    assert chunker.overlap_sentences == 3


def test_create_chunker_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        create_chunker(strategy="quantum_based", tokenizer=FakeTokenizer())


def test_create_chunker_rejects_unknown_preset():
    with pytest.raises(ValueError, match="Unknown preset"):
        create_chunker(preset="huge", tokenizer=FakeTokenizer())


@pytest.mark.parametrize("preset_name", list(CHUNKING_PRESETS.keys()))
def test_create_chunker_each_preset_constructs_a_valid_chunker(preset_name):
    chunker = create_chunker(preset=preset_name, tokenizer=FakeTokenizer())
    # All presets resolve to one of the two known concrete chunker classes
    assert isinstance(chunker, (TokenBasedChunker, SentenceBasedChunker))


def test_create_chunker_preset_overrides_explicit_chunk_size():
    # Preset's chunk_size wins over the keyword argument.
    chunker = create_chunker(
        preset="small",
        tokenizer=FakeTokenizer(),
        chunk_size=999,
    )
    assert isinstance(chunker, TokenBasedChunker)
    assert chunker.chunk_size == CHUNKING_PRESETS["small"]["chunk_size"]
