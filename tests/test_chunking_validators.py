"""Unit tests for drg.chunking.validators."""

from __future__ import annotations

from drg.chunking.strategies import Chunk
from drg.chunking.validators import ChunkValidator, validate_chunks


def _make_chunk(
    *,
    chunk_id: str = "ds_doc_chunk_000",
    sequence_index: int = 0,
    text: str = "non-empty body",
    token_count: int = 3,
    char_count: int = 14,
    origin_file: str = "file.txt",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        sequence_index=sequence_index,
        text=text,
        token_count=token_count,
        char_count=char_count,
        origin_dataset="ds",
        origin_file=origin_file,
        metadata={},
    )


# ---------------------------------------------------------------------------
# ChunkValidator.validate_chunks
# ---------------------------------------------------------------------------


def test_validate_chunks_reports_no_chunks_provided_when_empty_input():
    issues = ChunkValidator.validate_chunks([])
    assert issues == ["No chunks provided"]


def test_validate_chunks_passes_for_clean_sequence():
    chunks = [
        _make_chunk(chunk_id="ds_doc_chunk_000", sequence_index=0),
        _make_chunk(chunk_id="ds_doc_chunk_001", sequence_index=1),
        _make_chunk(chunk_id="ds_doc_chunk_002", sequence_index=2),
    ]
    assert ChunkValidator.validate_chunks(chunks) == []


def test_validate_chunks_flags_empty_text():
    chunks = [_make_chunk(chunk_id="ds_doc_chunk_000", text="   ")]
    issues = ChunkValidator.validate_chunks(chunks)
    assert any("is empty" in i for i in issues)


def test_validate_chunks_flags_zero_token_count():
    chunks = [_make_chunk(chunk_id="ds_doc_chunk_000", token_count=0)]
    issues = ChunkValidator.validate_chunks(chunks)
    assert any("zero tokens" in i for i in issues)


def test_validate_chunks_flags_negative_sequence_index():
    chunks = [_make_chunk(chunk_id="ds_doc_chunk_neg", sequence_index=-1)]
    issues = ChunkValidator.validate_chunks(chunks)
    assert any("negative sequence_index" in i for i in issues)


def test_validate_chunks_flags_out_of_order_sequence_indices():
    chunks = [
        _make_chunk(chunk_id="ds_doc_chunk_000", sequence_index=0),
        _make_chunk(chunk_id="ds_doc_chunk_002", sequence_index=2),
        _make_chunk(chunk_id="ds_doc_chunk_001", sequence_index=1),
    ]
    issues = ChunkValidator.validate_chunks(chunks)
    assert any("not in order" in i for i in issues)


def test_validate_chunks_flags_duplicate_ids():
    chunks = [
        _make_chunk(chunk_id="dup", sequence_index=0),
        _make_chunk(chunk_id="dup", sequence_index=1),
    ]
    issues = ChunkValidator.validate_chunks(chunks)
    assert any("Duplicate" in i for i in issues)


def test_validate_chunks_accumulates_multiple_issues():
    chunks = [
        _make_chunk(chunk_id="dup", sequence_index=0, token_count=0),
        _make_chunk(chunk_id="dup", sequence_index=-1, text="  "),
    ]
    issues = ChunkValidator.validate_chunks(chunks)
    # At least: empty text + zero tokens + negative index + duplicate ids
    assert len(issues) >= 4


# ---------------------------------------------------------------------------
# ChunkValidator.check_overlap_consistency
# ---------------------------------------------------------------------------


def test_check_overlap_consistency_empty_or_single_returns_no_issues():
    assert ChunkValidator.check_overlap_consistency([], 0.15) == []
    assert (
        ChunkValidator.check_overlap_consistency([_make_chunk(chunk_id="ds_doc_chunk_000")], 0.15)
        == []
    )


def test_check_overlap_consistency_flags_non_consecutive_indices():
    chunks = [
        _make_chunk(chunk_id="ds_doc_chunk_000", sequence_index=0),
        _make_chunk(chunk_id="ds_doc_chunk_002", sequence_index=2),  # gap
    ]
    issues = ChunkValidator.check_overlap_consistency(chunks, 0.15)
    assert any("Non-consecutive" in i for i in issues)


def test_check_overlap_consistency_skips_cross_document_pairs():
    # Two chunks from different documents should not be compared.
    chunks = [
        _make_chunk(chunk_id="a_doc_chunk_000", sequence_index=0, origin_file="a.txt"),
        _make_chunk(chunk_id="b_doc_chunk_005", sequence_index=5, origin_file="b.txt"),
    ]
    assert ChunkValidator.check_overlap_consistency(chunks, 0.15) == []


def test_check_overlap_consistency_passes_for_consecutive_indices():
    chunks = [
        _make_chunk(chunk_id="ds_doc_chunk_000", sequence_index=0),
        _make_chunk(chunk_id="ds_doc_chunk_001", sequence_index=1),
        _make_chunk(chunk_id="ds_doc_chunk_002", sequence_index=2),
    ]
    assert ChunkValidator.check_overlap_consistency(chunks, 0.15) == []


# ---------------------------------------------------------------------------
# validate_chunks (top-level helper)
# ---------------------------------------------------------------------------


def test_validate_chunks_helper_returns_true_for_valid_input():
    chunks = [_make_chunk(chunk_id="ds_doc_chunk_000", sequence_index=0)]
    assert validate_chunks(chunks) is True


def test_validate_chunks_helper_returns_false_for_invalid_input():
    assert validate_chunks([]) is False
    bad = [_make_chunk(chunk_id="ds_doc_chunk_000", text=" ")]
    assert validate_chunks(bad) is False
