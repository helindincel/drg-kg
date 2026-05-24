from drg.extract import _build_cross_chunk_context_snippets


def test_cross_chunk_snippets_include_related_chunk_by_anchor_entity():
    # Chunk 1 mentions Apple + iPhone
    chunk_texts = [
        "Apple was founded in 1976. Apple produces the iPhone.",
        # Chunk 2 mentions iPhone + Jony Ive but not Apple
        "The iPhone's chief designer was Jony Ive. He left the company in 2019.",
        "Unrelated chunk.",
    ]

    entity_to_chunks = {
        "apple": [0],
        "iphone": [0, 1],
        "jony ive": [1],
    }

    snippets = _build_cross_chunk_context_snippets(
        chunk_texts=chunk_texts,
        entity_to_chunks=entity_to_chunks,
        anchor_entities=["iPhone", "Jony Ive"],
        current_chunk_index=1,
        max_chunks=2,
        snippet_chars=120,
        max_total_chars=500,
        min_anchor_len=3,
    )

    # Should include excerpt from chunk 1 because it shares "iPhone" anchor.
    assert any("Chunk 1 excerpt" in s for s in snippets)
    assert any("Apple produces the iPhone" in s for s in snippets)


def test_cross_chunk_snippets_avoid_substring_collision():
    chunk_texts = [
        "This is about business and economics.",
        "We discussed the US policy yesterday.",
    ]
    entity_to_chunks = {"us": [1]}
    snippets = _build_cross_chunk_context_snippets(
        chunk_texts=chunk_texts,
        entity_to_chunks=entity_to_chunks,
        anchor_entities=["us"],
        current_chunk_index=0,
        max_chunks=2,
        snippet_chars=80,
        max_total_chars=200,
        min_anchor_len=3,  # filters out "us"
    )
    assert snippets == []
