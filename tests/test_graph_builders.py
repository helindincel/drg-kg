"""Unit tests for drg.graph.builders.

Covers `extract_evidence_snippet`, `_relation_docs_from_schema`, and
`build_enhanced_kg` end-to-end. All tests are deterministic and require no
LLM or optional dependencies — they exercise the pure data-transformation
layer that every example and CLI path depends on.
"""

from __future__ import annotations

import pytest

from drg.graph.builders import (
    _relation_docs_from_schema,
    build_enhanced_kg,
    extract_evidence_snippet,
)
from drg.schema import (
    DRGSchema,
    EnhancedDRGSchema,
    Entity,
    EntityType,
    Relation,
    RelationGroup,
)

# ---------------------------------------------------------------------------
# extract_evidence_snippet
# ---------------------------------------------------------------------------


def test_extract_evidence_snippet_returns_none_for_empty_inputs():
    assert extract_evidence_snippet("", "a", "b") is None
    assert extract_evidence_snippet("text", "", "b") is None
    assert extract_evidence_snippet("text", "a", "") is None
    assert extract_evidence_snippet("text", "  ", "b") is None
    assert extract_evidence_snippet("text", "a", "  ") is None


def test_extract_evidence_snippet_returns_none_when_terms_missing():
    text = "The quick brown fox jumps over the lazy dog."
    assert extract_evidence_snippet(text, "cat", "dog") is None
    assert extract_evidence_snippet(text, "fox", "elephant") is None


def test_extract_evidence_snippet_finds_basic_cooccurrence():
    text = "Apple released the iPhone in California."
    snippet = extract_evidence_snippet(text, "Apple", "iPhone")
    assert snippet is not None
    assert "Apple" in snippet
    assert "iPhone" in snippet


def test_extract_evidence_snippet_is_case_insensitive():
    text = "apple makes the IPHONE in California."
    snippet = extract_evidence_snippet(text, "Apple", "iPhone")
    assert snippet is not None
    # Original casing from the text is preserved
    assert "apple" in snippet.lower()
    assert "iphone" in snippet.lower()


def test_extract_evidence_snippet_respects_word_boundaries():
    # 'cat' should NOT match the embedded substring 'catalogue'.
    text = "The catalogue does not mention any pet. The dog is here."
    assert extract_evidence_snippet(text, "cat", "dog") is None


def test_extract_evidence_snippet_returns_none_when_pair_is_too_far():
    # Single occurrence of each, with a long stretch between them so the
    # closest co-occurrence distance exceeds max_pair_distance.
    text = "Alpha begins here. " + ("filler word " * 500) + " Beta ends here."
    assert extract_evidence_snippet(text, "Alpha", "Beta", max_pair_distance=50) is None


def test_extract_evidence_snippet_truncates_with_ellipses_when_too_long():
    long_text = (
        "Alpha is mentioned here. " + ("filler word " * 200) + " Eventually Beta is also mentioned."
    )
    snippet = extract_evidence_snippet(long_text, "Alpha", "Beta", max_chars=80)
    assert snippet is not None
    assert len(snippet) <= 120  # max_chars + ellipses + safety margin
    assert "…" in snippet  # truncation marker present somewhere


def test_extract_evidence_snippet_picks_closest_cooccurrence():
    # Two occurrences of Alpha and Beta — the closer pair (first sentence)
    # should be selected.
    text = (
        "Alpha and Beta are mentioned together in this sentence. "
        + ("middle text " * 30)
        + "Alpha alone appears here. Beta later alone too."
    )
    snippet = extract_evidence_snippet(text, "Alpha", "Beta")
    assert snippet is not None
    # The snippet should come from the first sentence (closest pair)
    assert "together" in snippet


# ---------------------------------------------------------------------------
# _relation_docs_from_schema
# ---------------------------------------------------------------------------


def test_relation_docs_returns_none_pair_when_schema_is_none():
    assert _relation_docs_from_schema(None, "rel", "A", "B") == (None, None)


def test_relation_docs_returns_none_pair_when_relation_not_found():
    schema = EnhancedDRGSchema(
        entity_types=[
            EntityType(name="A", description="a"),
            EntityType(name="B", description="b"),
        ],
        relation_groups=[
            RelationGroup(
                name="g",
                description="g",
                relations=[Relation("knows", "A", "B", description="X")],
            )
        ],
    )
    assert _relation_docs_from_schema(schema, "missing", "A", "B") == (None, None)


def test_relation_docs_prefers_exact_endpoint_match():
    schema = EnhancedDRGSchema(
        entity_types=[
            EntityType(name="A", description="a"),
            EntityType(name="B", description="b"),
            EntityType(name="C", description="c"),
        ],
        relation_groups=[
            RelationGroup(
                name="g",
                description="g",
                relations=[
                    Relation("rel", "A", "C", description="ac", detail="ac-detail"),
                    Relation("rel", "A", "B", description="ab", detail="ab-detail"),
                ],
            )
        ],
    )
    desc, det = _relation_docs_from_schema(schema, "rel", "A", "B")
    assert desc == "ab"
    assert det == "ab-detail"


def test_relation_docs_falls_back_to_name_only_match():
    schema = EnhancedDRGSchema(
        entity_types=[
            EntityType(name="A", description="a"),
            EntityType(name="B", description="b"),
            EntityType(name="C", description="c"),
        ],
        relation_groups=[
            RelationGroup(
                name="g",
                description="g",
                relations=[
                    Relation("rel", "A", "C", description="fallback-desc"),
                ],
            )
        ],
    )
    # Requested (A, B) endpoints do not match any (A, C) variant; fall back.
    desc, _det = _relation_docs_from_schema(schema, "rel", "A", "B")
    assert desc == "fallback-desc"


def test_relation_docs_handles_schema_without_relation_groups_attr():
    class Empty:  # pragma: no cover - trivial
        pass

    assert _relation_docs_from_schema(Empty(), "rel", "A", "B") == (None, None)


def test_relation_docs_skips_empty_strings():
    schema = EnhancedDRGSchema(
        entity_types=[
            EntityType(name="A", description="a"),
            EntityType(name="B", description="b"),
        ],
        relation_groups=[
            RelationGroup(
                name="g",
                description="g",
                relations=[Relation("rel", "A", "B", description="   ", detail="")],
            )
        ],
    )
    # Whitespace-only description / empty detail should be treated as None.
    desc, det = _relation_docs_from_schema(schema, "rel", "A", "B")
    assert desc is None
    assert det is None


# ---------------------------------------------------------------------------
# build_enhanced_kg
# ---------------------------------------------------------------------------


def test_build_enhanced_kg_empty_inputs_produce_empty_graph():
    kg = build_enhanced_kg(entities_typed=[], triples=[])
    assert len(kg.nodes) == 0
    assert len(kg.edges) == 0


def test_build_enhanced_kg_with_entities_only_creates_nodes_no_edges():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[],
    )
    assert set(kg.nodes.keys()) == {"Apple", "iPhone"}
    assert kg.nodes["Apple"].type == "Company"
    assert kg.nodes["iPhone"].type == "Product"
    assert len(kg.edges) == 0


def test_build_enhanced_kg_creates_edges_from_triples():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
    )
    assert len(kg.edges) == 1
    edge = kg.edges[0]
    assert edge.source == "Apple"
    assert edge.target == "iPhone"
    assert edge.relationship_type == "produces"


def test_build_enhanced_kg_auto_adds_nodes_from_triples_if_missing():
    # Triples mention entities not in the entities_typed list.
    kg = build_enhanced_kg(
        entities_typed=[],
        triples=[("Apple", "produces", "iPhone")],
    )
    assert "Apple" in kg.nodes
    assert "iPhone" in kg.nodes
    # Type defaults to None because it wasn't provided in entities_typed.
    assert kg.nodes["Apple"].type is None


def test_build_enhanced_kg_attaches_schema_description_to_edge_metadata():
    schema = EnhancedDRGSchema(
        entity_types=[
            EntityType(name="Company", description="c"),
            EntityType(name="Product", description="p"),
        ],
        relation_groups=[
            RelationGroup(
                name="g",
                description="g",
                relations=[
                    Relation(
                        "produces",
                        "Company",
                        "Product",
                        description="A company makes a product.",
                    )
                ],
            )
        ],
    )
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
        schema=schema,
    )
    assert len(kg.edges) == 1
    md = kg.edges[0].metadata
    assert md["relationship_description"] == "A company makes a product."
    assert md["triple"] == ["Apple", "produces", "iPhone"]


def test_build_enhanced_kg_uses_auto_description_when_schema_missing_relation():
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
        schema=None,
    )
    md = kg.edges[0].metadata
    # Falls back to a deterministic auto-description.
    assert md["relationship_description"].startswith("Auto-extracted relation 'produces'")


def test_build_enhanced_kg_populates_evidence_when_source_text_given():
    text = "Apple released the iPhone in California."
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "released", "iPhone")],
        source_text=text,
    )
    edge = kg.edges[0]
    assert "evidence" in edge.metadata
    assert "Apple" in edge.metadata["evidence"]
    assert "iPhone" in edge.metadata["evidence"]
    # relationship_detail prefers the evidence snippet when available
    assert edge.relationship_detail == edge.metadata["evidence"]


def test_build_enhanced_kg_skips_evidence_when_terms_absent_from_text():
    text = "This text talks about something entirely different."
    kg = build_enhanced_kg(
        entities_typed=[("Apple", "Company"), ("iPhone", "Product")],
        triples=[("Apple", "produces", "iPhone")],
        source_text=text,
    )
    edge = kg.edges[0]
    assert "evidence" not in edge.metadata
    # Falls back to a synthetic "s r o" detail string.
    assert edge.relationship_detail == "Apple produces iPhone"


def test_build_enhanced_kg_handles_legacy_drg_schema_without_crashing():
    # Legacy DRGSchema has no relation_groups attribute path, but the helper
    # still resolves cleanly via _relation_docs_from_schema's hasattr guard.
    schema = DRGSchema(
        entities=[Entity("A"), Entity("B")],
        relations=[Relation("rel", "A", "B", description="legacy-desc")],
    )
    kg = build_enhanced_kg(
        entities_typed=[("a-node", "A"), ("b-node", "B")],
        triples=[("a-node", "rel", "b-node")],
        schema=schema,
    )
    # Legacy schema does not expose relation_groups; the auto-description is
    # used and the build still succeeds.
    assert len(kg.edges) == 1


@pytest.mark.parametrize(
    "env_var,value",
    [
        ("DRG_EVIDENCE_MAX_CHARS", "not-an-int"),
        ("DRG_EVIDENCE_MAX_PAIR_DISTANCE", "garbage"),
    ],
)
def test_build_enhanced_kg_falls_back_when_env_vars_are_garbage(monkeypatch, env_var, value):
    monkeypatch.setenv(env_var, value)
    # Just ensure construction does not raise.
    kg = build_enhanced_kg(
        entities_typed=[("A", "T"), ("B", "T")],
        triples=[("A", "rel", "B")],
    )
    assert len(kg.edges) == 1
