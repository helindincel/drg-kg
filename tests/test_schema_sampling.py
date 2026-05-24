from drg.extract import _sample_text_for_schema_generation


def test_schema_sampling_includes_start_middle_end_markers():
    # Build a deterministic long text with markers spread across the document.
    parts = []
    parts.append("START_MARKER\n" + ("a" * 6000))
    parts.append("\nMID_MARKER\n" + ("b" * 6000))
    parts.append("\nLATE_MARKER\n" + ("c" * 6000))
    text = "".join(parts) * 4  # make it long

    sampled = _sample_text_for_schema_generation(text)
    assert "START_MARKER" in sampled
    assert "MID_MARKER" in sampled
    assert "LATE_MARKER" in sampled


def test_schema_sampling_budget_is_enforced():
    text = ("X" * 300000) + "END_UNIQUE"
    sampled = _sample_text_for_schema_generation(text)
    # Should not exceed default 100k by much (separator overhead is included in enforcement).
    assert len(sampled) <= 100000
    # End should be reachable via last chunk sampling.
    assert "END_UNIQUE" in sampled
